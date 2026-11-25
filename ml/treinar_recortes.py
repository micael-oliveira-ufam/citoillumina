#!/usr/bin/env python3
"""Treino e validação com RECORTES de células (nível celular), em vez de campos inteiros.

Etapas:
  1. extrair  - localiza núcleos em cada imagem (mesma heurística de anotar_celulas.py), salva cada
                recorte em <saida>/recortes/<classe>/<imagem>_<n>.png e grava manifesto.csv
                (recorte, imagem_origem, classe_fraca, x0,y0,x1,y1).
  2. revisar  - (opcional, recomendado) um especialista rotula os recortes: edite/crie rotulos.csv com
                colunas recorte,classe  (classe = SCC, HSIL, LSIL, NILM ou DESCARTAR). Esses rótulos
                substituem o rótulo fraco.
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
    # rotular: abra recortes_out/recortes/..., crie recortes_out/rotulos.csv
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
    t = sub.add_parser("treinar")
    t.add_argument("--saida", default="recortes_out")
    t.add_argument("--arquiteturas", nargs="+", default=["resnet50", "convnext_large", "swin_v2_b"])
    t.add_argument("--rotulos", help="CSV recorte,classe com rótulos manuais (DESCARTAR remove o recorte)")
    t.add_argument("--so-rotulados", action="store_true", help="usa apenas recortes com rótulo manual")
    a, extras = p.parse_known_args()
    extrair(a) if a.cmd == "extrair" else treinar(a, extras)


if __name__ == "__main__":
    main()
