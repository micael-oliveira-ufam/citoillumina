#!/usr/bin/env python3
"""Anota imagens de citologia com a classificação individual de cada célula.

Para cada imagem: (1) localiza candidatas a células pelos núcleos (deconvolução de cor
hematoxilina + limiar de Otsu + componentes conectados) ou por grade deslizante; (2) recorta
cada candidata e a classifica com uma rede treinada por comparar_arquiteturas.py;
(3) desenha caixas coloridas com "CLASSE xx%" e uma faixa de título com a classe da lâmina,
a confiança e o número de detecções, no estilo do Cervical AI Viewer.

Classe da lâmina: a mais grave detectada (NILM < LSIL < HSIL < SCC), com a maior confiança dentro dela.

Uso:
    python anotar_celulas.py --imagens pasta_ou_arquivo --arquitetura convnext_large \\
        --pesos resultados_arquiteturas/convnext_large/fold1.pt --saida anotadas
    # modos de localização: --modo nucleos (padrão) | grade
Gera <saida>/<nome>.jpg, <saida>/deteccoes.csv e <saida>/resumo.csv.

LIMITAÇÃO IMPORTANTE: as redes foram treinadas com campos inteiros (imagem -> classe), não com
recortes de células individuais. A classe por célula é, portanto, uma extrapolação (mudança de
domínio) e NÃO foi validada contra rótulos celulares. Para validar, é preciso um conjunto com
células rotuladas individualmente; treinar com recortes (--modo-treino recortes, futuro) melhora isso.
"""
import argparse, csv, glob, sys, warnings
warnings.filterwarnings("ignore", category=FutureWarning)
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage import color, filters, measure, morphology

sys.path.insert(0, str(Path(__file__).parent))
from comparar_arquiteturas import CLASSES, MEDIA, DESVIO, criar_modelo  # noqa: E402

GRAVIDADE = {"NILM": 0, "LSIL": 1, "HSIL": 2, "SCC": 3}
CORES = {"NILM": (22, 163, 74), "LSIL": (234, 179, 8), "HSIL": (255, 140, 0), "SCC": (220, 38, 38)}
EXT = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")


def candidatas_nucleos(rgb, area_min=0.0004, area_max=0.02, fator=2.4, minimo=0.07):
    """Caixas (x0,y0,x1,y1) ao redor de núcleos. Áreas relativas à imagem."""
    h, w = rgb.shape[:2]
    hed = color.rgb2hed(rgb)[..., 0]
    hed = filters.gaussian(hed, sigma=max(1, min(h, w) / 300))
    mask = hed > filters.threshold_otsu(hed)
    mask = morphology.remove_small_objects(morphology.opening(mask, morphology.disk(2)), int(area_min * h * w))
    mask = morphology.closing(mask, morphology.disk(3))
    caixas = []
    for r in measure.regionprops(measure.label(mask)):
        if not (area_min * h * w <= r.area <= area_max * h * w):
            continue
        y0, x0, y1, x1 = r.bbox
        cx, cy, lado = (x0 + x1) / 2, (y0 + y1) / 2, max(x1 - x0, y1 - y0) * fator
        lado = max(lado, minimo * min(h, w))
        caixas.append((int(max(0, cx - lado / 2)), int(max(0, cy - lado / 2)), int(min(w, cx + lado / 2)), int(min(h, cy + lado / 2))))
    return caixas


def candidatas_grade(rgb, lado=0.15, passo=0.10):
    h, w = rgb.shape[:2]
    L, P = int(lado * min(h, w)), int(passo * min(h, w))
    return [(x, y, x + L, y + L) for y in range(0, max(1, h - L + 1), P) for x in range(0, max(1, w - L + 1), P)]


@torch.no_grad()
def classificar(modelo, rgb, caixas, dev, lote=32):
    mean, std = np.array(MEDIA, np.float32), np.array(DESVIO, np.float32)
    out = []
    for i in range(0, len(caixas), lote):
        lotes = []
        for (x0, y0, x1, y1) in caixas[i:i + lote]:
            c = np.array(Image.fromarray(rgb[y0:y1, x0:x1]).resize((224, 224), Image.BILINEAR), np.float32) / 255
            lotes.append(torch.from_numpy(((c - mean) / std).transpose(2, 0, 1)))
        out.append(torch.softmax(modelo(torch.stack(lotes).to(dev)), 1).cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, len(CLASSES)))


def fonte(tam):
    for f in ("DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"):
        try:
            return ImageFont.truetype(f, tam)
        except OSError:
            pass
    return ImageFont.load_default()


def classe_lamina(dets):
    """Mais grave detectada, com a maior probabilidade dentro dela."""
    if not dets:
        return "NILM", 0.0
    g = max(GRAVIDADE[d["classe"]] for d in dets)
    c = [d for d in dets if GRAVIDADE[d["classe"]] == g]
    return c[0]["classe"] if c else "NILM", max(d["prob"] for d in c)


def desenhar(rgb, dets, nome):
    img = Image.fromarray(rgb)
    dr = ImageDraw.Draw(img)
    h, w = rgb.shape[:2]
    fs = max(11, int(min(h, w) / 55))
    f = fonte(fs)
    # menos graves primeiro, para as graves ficarem por cima
    for d in sorted(dets, key=lambda d: GRAVIDADE[d["classe"]]):
        x0, y0, x1, y1 = d["caixa"]
        cor = CORES[d["classe"]]
        dr.rectangle([x0, y0, x1, y1], outline=cor, width=max(2, fs // 5))
        txt = f"{d['classe']} {d['prob'] * 100:.0f}%"
        tb = dr.textbbox((0, 0), txt, font=f)
        ty = max(0, y0 - (tb[3] - tb[1]) - 6)
        dr.rectangle([x0, ty, x0 + tb[2] - tb[0] + 8, ty + tb[3] - tb[1] + 6], fill=cor)
        dr.text((x0 + 4, ty + 1), txt, fill="white", font=f)
    cl, pr = classe_lamina(dets)
    ff = fonte(int(fs * 1.5))
    faixa = f"{nome} · {cl} · {pr * 100:.0f}% · {len(dets)} detecções"
    dr.rectangle([0, 0, w, int(fs * 2.6)], fill=CORES[cl])
    dr.text((12, int(fs * 0.5)), faixa, fill="white", font=ff)
    return img, cl, pr


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--imagens", required=True)
    p.add_argument("--arquitetura", required=True)
    p.add_argument("--pesos", required=True)
    p.add_argument("--saida", default="anotadas")
    p.add_argument("--modo", choices=["nucleos", "grade"], default="nucleos")
    p.add_argument("--limiar", type=float, default=0.0, help="descarta detecções com confiança menor")
    p.add_argument("--max-caixas", type=int, default=120)
    a = p.parse_args()
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = criar_modelo(a.arquitetura, False)
    modelo.load_state_dict(torch.load(a.pesos, map_location="cpu"))
    modelo.to(dev).eval()
    ent = Path(a.imagens)
    arqs = [ent] if ent.is_file() else sorted(f for e in EXT for f in map(Path, glob.glob(str(ent / "**" / e), recursive=True)))
    Path(a.saida).mkdir(parents=True, exist_ok=True)
    fd, fr = open(Path(a.saida) / "deteccoes.csv", "w", newline="", encoding="utf-8-sig"), open(Path(a.saida) / "resumo.csv", "w", newline="", encoding="utf-8-sig")
    wd, wr = csv.writer(fd), csv.writer(fr)
    wd.writerow(["imagem", "x0", "y0", "x1", "y1", "classe", "prob"] + [f"p_{c}" for c in CLASSES])
    wr.writerow(["imagem", "classe_lamina", "confianca", "deteccoes"] + [f"n_{c}" for c in CLASSES])
    for f in arqs:
        rgb = np.array(Image.open(f).convert("RGB"))
        cx = (candidatas_nucleos if a.modo == "nucleos" else candidatas_grade)(rgb)[:a.max_caixas]
        pr = classificar(modelo, rgb, cx, dev)
        dets = [dict(caixa=c, classe=CLASSES[int(q.argmax())], prob=float(q.max()), probs=q) for c, q in zip(cx, pr) if q.max() >= a.limiar]
        img, cl, conf = desenhar(rgb, dets, f.stem)
        img.save(Path(a.saida) / f"{f.stem}.jpg", quality=90)
        for d in dets:
            wd.writerow([f.name, *d["caixa"], d["classe"], f"{d['prob']:.4f}", *[f"{v:.4f}" for v in d["probs"]]])
        wr.writerow([f.name, cl, f"{conf:.4f}", len(dets)] + [sum(d["classe"] == c for d in dets) for c in CLASSES])
        print(f"{f.name}: {cl} {conf * 100:.0f}% ({len(dets)} detecções)")


if __name__ == "__main__":
    main()
