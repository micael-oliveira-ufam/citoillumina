#!/usr/bin/env python3
"""Extrai as imagens do Atlas Digital de Citopatologia do Colo do Útero (IARC) e organiza em pastas para treino.

Fonte: https://screening.iarc.fr/atlascyto.php  (© IARC; ver "Direitos de uso" abaixo)

O site é um aplicativo JavaScript que serve páginas estáticas em /pages/<página>.php/ (índice index.json). O script:
  1. lê o índice de categorias e as listas de cada categoria (inglês e português);
  2. baixa cada imagem em tamanho original (/pic/cytoNNNN.jpg), uma vez só, com pausa entre pedidos;
  3. classifica cada imagem em uma classe de treino a partir do código de categoria do atlas;
  4. grava uma planilha (XLSX e CSV) com uma linha por imagem.

Saída:
    <saida>/dataset/SCC|HSIL|LSIL|NILM/cytoNNNN.jpg      classes do modelo (mesmos nomes de comparar_arquiteturas.py)
    <saida>/dataset/fora_do_treino/<grupo>/cytoNNNN.jpg  ASC-US, ASC-H, LSIL ou HSIL?, glandulares, insatisfatórias, métodos
    <saida>/planilha_atlas_iarc.xlsx e .csv              descrição de cada imagem
    <saida>/brutos/                                      cache (índices JSON) para reexecução

Retomável: imagens já baixadas (e conferidas) não são baixadas de novo.

Uso:
    python extrair_atlas_iarc.py --saida iarc_atlas
    python extrair_atlas_iarc.py --saida iarc_atlas --limite 20     # teste rápido

Direitos de uso: o atlas é protegido por direitos autorais do IARC ("All rights reserved"). Os termos do IARC permitem usar
trechos para pesquisa, estudo privado e fins educacionais não comerciais, com citação da fonte; reproduzir partes substanciais
ou redistribuir exige permissão escrita (publications@iarc.who.int). Os termos não mencionam treino de IA. Use as imagens apenas
localmente, não as publique em repositório nem site, e peça autorização antes de divulgar o conjunto ou modelos treinados com ele.

Limites do mapeamento: a classe vem da categoria do atlas, não de uma revisão célula a célula. Imagens ilustram uma lesão ou um
achado em um campo e podem conter células de outras classes. Conflitos (imagem em categorias de classes diferentes) vão marcados.
"""
import argparse
import csv
import hashlib
import html
import json
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = "https://screening.iarc.fr"
UA = "Mozilla/5.0 (pesquisa academica UFAM; extracao privada do atlas IARC para estudo)"

# classe de treino por código de categoria (testado do mais específico ao mais geral)
MAPA = [
    (r"^F1d", "SCC", ""),
    (r"^F1c1", "fora_do_treino/LSIL_ou_HSIL_indeterminado", "atlas rotula como 'LSIL or HSIL?'"),
    (r"^F1c4", "HSIL", "HSIL com aspectos suspeitos de invasão"),
    (r"^F1c", "HSIL", ""),
    (r"^F1b", "LSIL", ""),
    (r"^F1a1", "fora_do_treino/ASC-US", ""),
    (r"^F1a2", "fora_do_treino/ASC-H", ""),
    (r"^F1a", "fora_do_treino/ASC_geral", ""),
    (r"^F2", "fora_do_treino/glandulares", "células glandulares atípicas/adenocarcinoma (fora das 4 classes)"),
    (r"^G", "fora_do_treino/outras_malignas", "endométrio/linfoma (fora das 4 classes)"),
    (r"^D", "NILM", ""),
    (r"^E", "NILM", "alteração benigna/infecção (NILM no Bethesda)"),
    (r"^A1", "fora_do_treino/insatisfatorio", "amostra insatisfatória"),
    (r"^A", "fora_do_treino/metodos_preparo", "amostragem e preparo"),
]
CLASSES_TREINO = {"SCC", "HSIL", "LSIL", "NILM"}


def classe_de(cat):
    for pad, cl, nota in MAPA:
        if re.match(pad, cat):
            return cl, nota
    return "fora_do_treino/nao_mapeado", "categoria sem regra"


def pedir(url, binario=False, tentativas=4):
    for t in range(tentativas):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=40) as r:
                d = r.read()
            return d if binario else d.decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if t == tentativas - 1:
                raise
            time.sleep(2 ** t)


def limpar(t):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", t))).strip()


def indice(pagina, pausa):
    d = json.loads(pedir(f"{BASE}/pages/{pagina}/index.json"))
    time.sleep(pausa)
    return d


def pagina_html(pagina, idx, chave, pausa):
    if chave not in idx:
        return None
    h = pedir(f"{BASE}/pages/{pagina}/{idx[chave][0]}")
    time.sleep(pausa)
    return h


def hierarquia(h):
    """Nome de cada categoria a partir do índice do atlas (página principal)."""
    nomes, secao = {}, ""
    corpo = h[h.find("panel panel-primary"):]
    for m in re.finditer(r'<h4 class="panel-title">(.*?)</h4>|<div class="list-group-item-(\d)">(.*?)</div>', corpo, flags=re.S):
        if m.group(1):
            secao = limpar(m.group(1))
        else:
            a = re.search(r'href="[^"]*cat=([^&"]+)', m.group(3))
            if a:
                nomes[a.group(1)] = (limpar(m.group(3)), secao)
    return nomes


def linhas_lista(h):
    """(Id, descrição) de cada imagem de uma página de lista."""
    out = []
    for tr in re.split(r"<Tr>", h, flags=re.I)[1:]:
        i = re.search(r"atlascyto_detail\.php\?Id=([A-Za-z0-9_]+)", tr)
        d = re.search(r"<Td valign\s*=\s*top>(.*?)</Td>", tr, flags=re.S | re.I)
        if i and d:
            out.append((i.group(1), limpar(d.group(1))))
    return out


def titulo_h1(h):
    m = re.search(r"<h1>(.*?)<A href", h, flags=re.S | re.I) or re.search(r"<h1>(.*?)</h1>", h, flags=re.S | re.I)
    return limpar(m.group(1)) if m else ""


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--saida", default="iarc_atlas")
    p.add_argument("--pausa", type=float, default=0.25, help="segundos entre pedidos (seja gentil com o servidor)")
    p.add_argument("--limite", type=int, help="processa só N imagens (teste)")
    a = p.parse_args()
    out = Path(a.saida)
    (out / "dataset").mkdir(parents=True, exist_ok=True)
    (out / "brutos").mkdir(exist_ok=True)

    # 1. categorias
    idx_main = indice("atlascyto.php", a.pausa)
    h_main = pagina_html("atlascyto.php", idx_main, "lang=1", a.pausa)
    nomes = hierarquia(h_main)
    idx_lista = indice("atlascyto_list.php", a.pausa)
    cats = sorted({m.group(1) for k in idx_lista for m in [re.match(r"cat=([^&]+)&lang=1$", k)] if m and m.group(1) != "null"})
    print(f"{len(cats)} categorias com lista")

    # 2. imagens por categoria (inglês e português)
    imagens = {}  # Id -> dict
    for c in cats:
        en = linhas_lista(pagina_html("atlascyto_list.php", idx_lista, f"cat={c}&lang=1", a.pausa) or "")
        pt = dict(linhas_lista(pagina_html("atlascyto_list.php", idx_lista, f"cat={c}&lang=4", a.pausa) or ""))
        for i, d in en:
            im = imagens.setdefault(i, {"id": i, "cats": [], "desc_en": d, "desc_pt": pt.get(i, "")})
            im["cats"].append(c)
        print(f"  {c:7s} {len(en):4d} imagens  {nomes.get(c, ('', ''))[0][:60]}")
    ids = sorted(imagens)
    print(f"{len(ids)} imagens únicas")
    if a.limite:
        ids = ids[:a.limite]

    # 3. detalhes (título) e download
    idx_det = indice("atlascyto_detail.php", a.pausa)
    linhas, hashes = [], {}
    for n, i in enumerate(ids, 1):
        im = imagens[i]
        # categoria primária = a mais específica (código mais longo); classes em conflito são sinalizadas
        cats_i = sorted(set(im["cats"]), key=lambda c: (-len(c), c))
        prim = cats_i[0]
        classes = {classe_de(c)[0] for c in cats_i}
        cl, nota = classe_de(prim)
        conflito = len({x for x in classes if x in CLASSES_TREINO or x.startswith("fora")}) > 1
        titulo = ""
        h = pagina_html("atlascyto_detail.php", idx_det, f"Id={i}&cat={prim}&flag=0&lang=1", a.pausa)
        fotos = []
        if h:
            titulo = titulo_h1(h)
            fotos = list(dict.fromkeys(re.findall(r'href="pic/([A-Za-z0-9_]+)\.jpg"', h)))
        if not fotos:
            fotos = [i]          # imagens com painéis (A, B...) aparecem como <Id>a.jpg, <Id>b.jpg
        obj = re.search(r"obj\.?\s*([0-9]+)\s*x", im["desc_en"])
        txt = im["desc_en"].lower()
        preparo = "meio líquido" if re.search(r"liquid|thinprep|surepath", txt) else ("convencional" if "conventional" in txt else "")
        for foto in fotos:
            destino = out / "dataset" / cl / f"{foto}.jpg"
            destino.parent.mkdir(parents=True, exist_ok=True)
            if not (destino.exists() and destino.read_bytes()[:2] == b"\xff\xd8"):
                dados = pedir(f"{BASE}/pic/{foto}.jpg", binario=True)
                if dados[:2] != b"\xff\xd8":      # o servidor devolve a página HTML (200) quando o arquivo não existe
                    print(f"  AVISO: {foto}.jpg não é JPEG; ignorado")
                    continue
                destino.write_bytes(dados)
                time.sleep(a.pausa)
            dados = destino.read_bytes()
            sha = hashlib.sha256(dados).hexdigest()
            try:
                from PIL import Image
                with Image.open(destino) as imo:
                    w, hh = imo.size
            except Exception:
                w = hh = ""
            hashes.setdefault(sha, []).append(foto)
            linhas.append({
                "id_imagem": foto, "id_registro": i, "painel": foto[len(i):] if foto.startswith(i) and foto != i else "",
                "arquivo": str(destino.relative_to(out)), "classe_treino": cl.split("/")[-1],
                "usar_no_treino_4_classes": "sim" if cl in CLASSES_TREINO else "não",
                "pasta": cl, "categoria_codigo": prim, "categoria_nome": nomes.get(prim, ("", ""))[0], "secao_atlas": nomes.get(prim, ("", ""))[1],
                "todas_categorias": ";".join(cats_i), "titulo_diagnostico_en": titulo, "descricao_en": im["desc_en"], "descricao_pt": im["desc_pt"],
                "objetiva": f"{obj.group(1)}x" if obj else "", "preparo_mencionado": preparo, "largura_px": w, "altura_px": hh,
                "tamanho_kb": round(len(dados) / 1024, 1), "sha256": sha, "conflito_de_classe": "sim" if conflito else "",
                "nota_mapeamento": nota, "url_imagem": f"{BASE}/pic/{foto}.jpg", "url_detalhe": f"{BASE}/atlascyto_detail.php?Id={i}&cat={prim}&flag=0&lang=1"})
        if n % 50 == 0 or n == len(ids):
            print(f"  {n}/{len(ids)} registros, {len(linhas)} imagens")
    for r in linhas:
        r["duplicada_de"] = ";".join(x for x in hashes[r["sha256"]] if x != r["id_imagem"])

    # 4. planilha
    campos = list(linhas[0])
    with open(out / "planilha_atlas_iarc.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=campos)
        w.writeheader()
        w.writerows(linhas)
    try:
        from openpyxl import Workbook
        from openpyxl.styles import Alignment, Font, PatternFill
        wb = Workbook()
        ws = wb.active
        ws.title = "Imagens"
        ws.append(campos)
        for r in linhas:
            ws.append([r[c] for c in campos])
        for c in ws[1]:
            c.font = Font(bold=True, color="FFFFFF")
            c.fill = PatternFill("solid", fgColor="4C1D95")
        larg = {"descricao_en": 60, "descricao_pt": 60, "titulo_diagnostico_en": 34, "categoria_nome": 34, "arquivo": 36, "url_imagem": 30, "url_detalhe": 30, "sha256": 14}
        for k, c in enumerate(campos, 1):
            ws.column_dimensions[ws.cell(1, k).column_letter].width = larg.get(c, 16)
        ws.freeze_panes = "B2"
        ws.auto_filter.ref = ws.dimensions
        wc = wb.create_sheet("Resumo")
        wc.append(["pasta", "n_imagens"])
        cont = {}
        for r in linhas:
            cont[r["pasta"]] = cont.get(r["pasta"], 0) + 1
        for k in sorted(cont):
            wc.append([k, cont[k]])
        wc.append(["TOTAL", len(linhas)])
        wk = wb.create_sheet("Categorias")
        wk.append(["codigo", "nome", "secao_atlas", "pasta_de_destino", "n_imagens_listadas"])
        for c in cats:
            n = sum(1 for im in imagens.values() if c in im["cats"])
            wk.append([c, nomes.get(c, ("", ""))[0], nomes.get(c, ("", ""))[1], classe_de(c)[0], n])
        wl = wb.create_sheet("Leia-me")
        for t in ["Atlas Digital de Citopatologia do Colo do Útero, IARC (https://screening.iarc.fr/atlascyto.php).",
                  f"Extraído em {time.strftime('%d/%m/%Y')}. Uma linha por imagem; descrições em inglês e português vêm do atlas.",
                  "Citar: IARC. Cytopathology of the uterine cervix - digital atlas. Lyon: IARC; acesso em " + time.strftime('%d/%m/%Y') + ".",
                  "Direitos: © IARC, todos os direitos reservados. Uso para pesquisa/estudo privado/fins educacionais não comerciais com citação da fonte;",
                  "redistribuição e reprodução substancial exigem permissão escrita (publications@iarc.who.int). Os termos não citam treino de IA.",
                  "A classe de treino é derivada da categoria do atlas, sem revisão célula a célula. Veja a coluna conflito_de_classe e nota_mapeamento.",
                  "ASC-US, ASC-H, 'LSIL ou HSIL?', glandulares, insatisfatórias e métodos ficam fora das 4 classes (SCC, HSIL, LSIL, NILM).",
                  "Alterações benignas e infecções (categorias E) foram mapeadas para NILM, conforme o Bethesda."]:
            wl.append([t])
        wl.column_dimensions["A"].width = 150
        wb.save(out / "planilha_atlas_iarc.xlsx")
    except ImportError:
        print("openpyxl ausente: só o CSV foi gerado")
    print("\nResumo por pasta:")
    cont = {}
    for r in linhas:
        cont[r["pasta"]] = cont.get(r["pasta"], 0) + 1
    for k in sorted(cont):
        print(f"  {k:55s} {cont[k]:4d}")
    dup = sum(1 for v in hashes.values() if len(v) > 1)
    print(f"conflitos de classe: {sum(1 for r in linhas if r['conflito_de_classe'])}; hashes duplicados: {dup}")
    print("Planilha:", out / "planilha_atlas_iarc.xlsx")


if __name__ == "__main__":
    main()
