#!/usr/bin/env python3
"""Treino e validação com RECORTES de células (nível celular), em vez de campos inteiros.

Etapas:
  1. extrair  - localiza núcleos em cada imagem (mesma heurística de anotar_celulas.py), salva cada
                recorte em <saida>/recortes/<classe>/<imagem>_<n>.png e grava manifesto.csv
                (recorte, imagem_origem, classe_fraca, x0,y0,x1,y1).
  2. revisar  - (opcional, recomendado) um especialista rotula os recortes: edite/crie rotulos.csv com
                colunas recorte,classe  (classe = SCC, HSIL, LSIL, NILM ou DESCARTAR). Esses rótulos
                substituem o rótulo fraco.
  2b. planilha - gera planilha_rotulagem.xlsx (cega: recorte + contexto lado a lado, sem a classe da imagem) e
                chave_rotulagem.csv; depois 'planilha-para-csv' converte a planilha preenchida em rotulos.csv.
  3. treinar  - chama comparar_arquiteturas.py sobre os recortes, com validação cruzada AGRUPADA por
                imagem de origem (StratifiedGroupKFold): recortes da mesma imagem nunca ficam em
                treino e validação ao mesmo tempo.

RÓTULO FRACO: sem rótulos manuais, cada recorte herda a classe da imagem inteira. Isso é ruído: uma imagem
HSIL contém muitas células normais, que ficam rotuladas HSIL. A acurácia obtida com rótulo fraco mede a
capacidade de reproduzir o rótulo da imagem, NÃO a de classificar a célula. Só a validação contra recortes
rotulados individualmente por especialista (--rotulos) mede desempenho celular. O script avisa isso no relatório
e informa quantos recortes têm rótulo manual. Recomenda-se rotular ao menos uma amostra estratificada
(ex.: 100 recortes por classe) e relatar as métricas apenas sobre ela (--so-rotulados).

Uso:
    python treinar_recortes.py extrair --dados "Liquid Citology + IARC dataset" --saida recortes_out
    python treinar_recortes.py planilha --dados "Liquid Citology + IARC dataset" --saida recortes_out --por-classe 100
    # o especialista preenche planilha_rotulagem.xlsx; depois:
    python treinar_recortes.py planilha-para-csv --saida recortes_out --xlsx planilha_preenchida.xlsx
    python treinar_recortes.py treinar --saida recortes_out --arquiteturas resnet50 convnext_large \\
        --rotulos recortes_out/rotulos.csv --so-rotulados
Parâmetros não reconhecidos em 'treinar' são repassados a comparar_arquiteturas.py (ex.: --folds 5 --salvar-pesos).
"""
import argparse
import csv
import subprocess
import sys
import warnings
from pathlib import Path

import numpy as np
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)
sys.path.insert(0, str(Path(__file__).parent))
from anotar_celulas import candidatas_nucleos  # noqa: E402
from comparar_arquiteturas import CLASSES, carregar_caminhos  # noqa: E402


def extrair(a):
    caminhos, rotulos = carregar_caminhos(a.dados, a.limite_por_classe)
    saida = Path(a.saida)
    for c in CLASSES:
        (saida / "recortes" / c).mkdir(parents=True, exist_ok=True)
    linhas = []
    for f, r in zip(caminhos, rotulos):
        rgb = np.array(Image.open(f).convert("RGB"))
        for n, (x0, y0, x1, y1) in enumerate(candidatas_nucleos(rgb)[:a.max_por_imagem]):
            nome = f"{CLASSES[r]}__{Path(f).stem}__{n:03d}.png"
            Image.fromarray(rgb[y0:y1, x0:x1]).save(saida / "recortes" / CLASSES[r] / nome)
            linhas.append([nome, f"{CLASSES[r]}__{Path(f).name}", CLASSES[r], x0, y0, x1, y1])
    with open(saida / "manifesto.csv", "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["recorte", "imagem_origem", "classe_fraca", "x0", "y0", "x1", "y1"])
        w.writerows(linhas)
    # planilha de revisão: recorte,classe (vazia = manter rótulo fraco)
    print(f"{len(linhas)} recortes de {len(caminhos)} imagens em {saida / 'recortes'}; manifesto: {saida / 'manifesto.csv'}")
    por = {c: sum(1 for l in linhas if l[2] == c) for c in CLASSES}
    print("recortes por classe fraca:", por)


VALORES = ["SCC", "HSIL", "LSIL", "NILM", "ASC-US/ASC-H (indeterminado)", "DESCARTAR (não avaliável)"]


def planilha(a):
    """Planilha cega de rotulagem: amostra estratificada pela classe fraca, ordem embaralhada, sem revelar a classe da imagem."""
    import io
    import random
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XImg
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.worksheet.datavalidation import DataValidation
    from PIL import ImageDraw

    saida = Path(a.saida)
    man = list(csv.DictReader(open(saida / "manifesto.csv", encoding="utf-8-sig")))
    caminhos, rotulos = carregar_caminhos(a.dados)
    origem = {f"{CLASSES[r]}__{Path(f).name}": f for f, r in zip(caminhos, rotulos)}
    rng = random.Random(a.semente)
    escolhidos = []
    for c in CLASSES:
        por_img = {}
        for m in man:
            if m["classe_fraca"] == c:
                por_img.setdefault(m["imagem_origem"], []).append(m)
        pool = []
        for lista in por_img.values():           # no máximo N recortes por imagem, para diversidade
            rng.shuffle(lista)
            pool += lista[:a.max_por_imagem]
        rng.shuffle(pool)
        escolhidos += pool[:a.por_classe]
        if len(pool) < a.por_classe:
            print(f"AVISO: classe {c}: só {len(pool)} recortes disponíveis (pedidos {a.por_classe})")
    rng.shuffle(escolhidos)                       # ordem aleatória: não agrupa por classe

    wb = Workbook()
    ws = wb.active
    ws.title = "Rotulagem"
    cab = ["ID", "Recorte (célula)", "Contexto (caixa vermelha)", "Rótulo", "Certeza", "Observação"]
    ws.append(cab)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF")
        c.fill = PatternFill("solid", fgColor="4C1D95")
        c.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    for col, w in zip("ABCDEF", (6, 22, 52, 30, 12, 34)):
        ws.column_dimensions[col].width = w
    dv = DataValidation(type="list", formula1='"' + ",".join(VALORES) + '"', allow_blank=True, showErrorMessage=True,
                        errorTitle="Valor inválido", error="Escolha um valor da lista.")
    dc = DataValidation(type="list", formula1='"alta,média,baixa"', allow_blank=True)
    ws.add_data_validation(dv)
    ws.add_data_validation(dc)
    chave = [["id", "recorte", "imagem_origem", "classe_fraca"]]
    for i, m in enumerate(escolhidos, 1):
        r = i + 1
        ws.row_dimensions[r].height = 195
        ws.cell(r, 1, i).alignment = Alignment(horizontal="center", vertical="center")
        rec = Image.open(saida / "recortes" / m["classe_fraca"] / m["recorte"]).convert("RGB")
        rec.thumbnail((150, 150))
        b = io.BytesIO(); rec.save(b, "PNG"); b.seek(0)
        ws.add_image(XImg(b), f"B{r}")
        ctx = Image.open(origem[m["imagem_origem"]]).convert("RGB")
        k = min(340 / ctx.width, 250 / ctx.height)
        ctx = ctx.resize((int(ctx.width * k), int(ctx.height * k)))
        d = ImageDraw.Draw(ctx)
        x0, y0, x1, y1 = (int(float(m[t]) * k) for t in ("x0", "y0", "x1", "y1"))
        d.rectangle([x0, y0, x1, y1], outline=(255, 0, 0), width=2)
        b2 = io.BytesIO(); ctx.save(b2, "PNG"); b2.seek(0)
        ws.add_image(XImg(b2), f"C{r}")
        dv.add(ws.cell(r, 4)); dc.add(ws.cell(r, 5))
        for col in (4, 5, 6):
            ws.cell(r, col).alignment = Alignment(vertical="center", wrap_text=True)
        chave.append([i, m["recorte"], m["imagem_origem"], m["classe_fraca"]])
    ws.freeze_panes = "A2"
    ins = wb.create_sheet("Instruções")
    for linha in [
        "ROTULAGEM CEGA DE CÉLULAS (projeto CitoIllumina)",
        "",
        "Objetivo: obter rótulos celulares de referência para treinar e validar a classificação por célula.",
        "A planilha NÃO mostra a classe da imagem de origem, para não influenciar. Rotule pelo que vê na célula (coluna B), usando o contexto (coluna C) se precisar.",
        "",
        "Rótulo (lista suspensa): SCC, HSIL, LSIL, NILM. O modelo só tem essas 4 classes.",
        "Use 'ASC-US/ASC-H (indeterminado)' quando a célula for atípica mas não permitir LSIL ou HSIL com segurança.",
        "Use 'DESCARTAR' para: fora de foco, artefato, hemácia, leucócito, células sobrepostas, recorte que não é uma célula avaliável.",
        "Certeza: alta, média ou baixa. Observação: livre.",
        "",
        "Referência de critérios: Sistema Bethesda (Nayar e Wilbur, 2015). Os critérios finais são do especialista.",
        "Idealmente duas pessoas rotulam a mesma planilha de forma independente (cópias separadas); o kappa entre elas mede a confiabilidade do rótulo.",
        "Devolva o arquivo preenchido (.xlsx). Não altere a coluna ID nem a ordem das linhas.",
    ]:
        ins.append([linha])
    ins.column_dimensions["A"].width = 140
    ins["A1"].font = Font(bold=True, size=13)
    wb.save(saida / "planilha_rotulagem.xlsx")
    csv.writer(open(saida / "chave_rotulagem.csv", "w", newline="", encoding="utf-8-sig")).writerows(chave)
    print(f"{len(escolhidos)} recortes em {saida / 'planilha_rotulagem.xlsx'}; chave (NÃO entregar ao especialista): {saida / 'chave_rotulagem.csv'}")


def planilha_para_csv(a):
    from openpyxl import load_workbook
    saida = Path(a.saida)
    chave = {int(r["id"]): r for r in csv.DictReader(open(saida / "chave_rotulagem.csv", encoding="utf-8-sig"))}
    ws = load_workbook(a.xlsx, data_only=True)["Rotulagem"]
    linhas, vazios, invalidos = [["recorte", "classe", "certeza", "observacao", "motivo"]], 0, []
    for row in ws.iter_rows(min_row=2, values_only=True):
        id_, _, _, val, cert, obs = (list(row) + [None] * 6)[:6]
        if id_ is None:
            continue
        if not val:
            vazios += 1
            continue
        val = str(val).strip()
        if val not in VALORES:
            invalidos.append((id_, val))
            continue
        classe, motivo = (val, "") if val in CLASSES else ("DESCARTAR", val)
        linhas.append([chave[int(id_)]["recorte"], classe, cert or "", obs or "", motivo])
    csv.writer(open(saida / "rotulos.csv", "w", newline="", encoding="utf-8-sig")).writerows(linhas)
    n = len(linhas) - 1
    por = {c: sum(1 for l in linhas[1:] if l[1] == c) for c in CLASSES + ["DESCARTAR"]}
    print(f"{n} rótulos em {saida / 'rotulos.csv'}; sem rótulo: {vazios}; inválidos: {invalidos[:5]}; por classe: {por}")
    fraca = {r["recorte"]: r["classe_fraca"] for r in chave.values()}
    conc = [fraca[l[0]] == l[1] for l in linhas[1:] if l[1] in CLASSES]
    if conc:
        print(f"Concordância do rótulo manual com o rótulo fraco (classe da imagem): {sum(conc) / len(conc) * 100:.1f}% de {len(conc)}")


def treinar(a, extras):
    saida = Path(a.saida)
    manifesto = list(csv.DictReader(open(saida / "manifesto.csv", encoding="utf-8-sig")))
    manuais = {}
    if a.rotulos:
        for r in csv.DictReader(open(a.rotulos, encoding="utf-8-sig")):
            if r["classe"].strip():
                manuais[r["recorte"]] = r["classe"].strip().upper().replace("DESCARTAR", "DESCARTAR")
        invalidos = {v for v in manuais.values() if v not in CLASSES + ["DESCARTAR"]}
        if invalidos:
            raise SystemExit(f"Classes inválidas em {a.rotulos}: {invalidos}")
    base = saida / "treino_recortes"
    for c in CLASSES:
        (base / c).mkdir(parents=True, exist_ok=True)
    grupos = [["arquivo", "grupo"]]
    n_man = n_desc = 0
    for m in manifesto:
        nome = m["recorte"]
        cl = manuais.get(nome, m["classe_fraca"])
        if cl == "DESCARTAR":
            n_desc += 1
            continue
        if nome in manuais:
            n_man += 1
        elif a.so_rotulados:
            continue
        origem = saida / "recortes" / m["classe_fraca"] / nome
        destino = base / cl / nome
        if destino.exists() or destino.is_symlink():
            destino.unlink()
        destino.symlink_to(origem.resolve())
        grupos.append([nome, m["imagem_origem"]])
    gcsv = base / "grupos.csv"
    csv.writer(open(gcsv, "w", newline="", encoding="utf-8")).writerows(grupos)
    print(f"{len(grupos) - 1} recortes no treino; {n_man} com rótulo manual; {n_desc} descartados; "
          f"modo: {'SÓ rotulados (validação celular)' if a.so_rotulados else 'rótulo fraco herdado da imagem'}")
    if not manuais:
        print("AVISO: nenhum rótulo manual; resultado = reprodução do rótulo da imagem, não classificação celular.")
    cmd = [sys.executable, str(Path(__file__).parent / "comparar_arquiteturas.py"), "--dados", str(base), "--saida", str(saida / "resultados"),
           "--grupos", str(gcsv), "--arquiteturas", *a.arquiteturas] + extras
    print(" ".join(cmd))
    sys.exit(subprocess.call(cmd))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extrair")
    e.add_argument("--dados", required=True)
    e.add_argument("--saida", default="recortes_out")
    e.add_argument("--max-por-imagem", type=int, default=60)
    e.add_argument("--limite-por-classe", type=int)
    pl = sub.add_parser("planilha")
    pl.add_argument("--dados", required=True)
    pl.add_argument("--saida", default="recortes_out")
    pl.add_argument("--por-classe", type=int, default=100)
    pl.add_argument("--max-por-imagem", type=int, default=3)
    pl.add_argument("--semente", type=int, default=42)
    pc = sub.add_parser("planilha-para-csv")
    pc.add_argument("--saida", default="recortes_out")
    pc.add_argument("--xlsx", required=True)
    t = sub.add_parser("treinar")
    t.add_argument("--saida", default="recortes_out")
    t.add_argument("--arquiteturas", nargs="+", default=["resnet50", "convnext_large", "swin_v2_b"])
    t.add_argument("--rotulos", help="CSV recorte,classe com rótulos manuais (DESCARTAR remove o recorte)")
    t.add_argument("--so-rotulados", action="store_true", help="usa apenas recortes com rótulo manual")
    a, extras = p.parse_known_args()
    {"extrair": extrair, "planilha": planilha, "planilha-para-csv": planilha_para_csv}.get(a.cmd, lambda x: treinar(x, extras))(a)


if __name__ == "__main__":
    main()
