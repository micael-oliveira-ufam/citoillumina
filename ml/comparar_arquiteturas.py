#!/usr/bin/env python3
"""Comparação de arquiteturas de redes neurais para citologia em meio líquido.

Treina e valida, com o MESMO protocolo, redes convolucionais clássicas e modernas e
transformers sobre as imagens de citologia (NILM, LSIL, HSIL, SCC), e documenta tudo:
métricas por fold e globais, matrizes de confusão, curvas de acurácia e perda em função
das épocas, testes estatísticos entre arquiteturas e um relatório em Markdown e DOCX.

O protocolo reproduz o notebook do projeto (kaggle-algoritmo-de-citologia-em-meio-líquido):
  * Adam, taxa de aprendizado 1e-4, decaimento de pesos 1e-5, lote 16, até 50 épocas;
  * parada antecipada com paciência 5 sobre a ACURÁCIA de validação, restaurando a melhor época;
  * validação cruzada estratificada em 5 folds (shuffle, semente 42);
  * entrada 224x224 (Resize do albumentations, INTER_LINEAR); aumentos de dados: giro horizontal
    e vertical, rotação de 90° e ColorJitter (p = 0,3), normalização do ImageNet;
  * cabeça de classificação nova: Dropout(0,5) + Linear(4), sobre pesos pré-treinados no ImageNet.

Arquiteturas (torchvision; ano de publicação original entre parênteses):
  alexnet (2012), vgg16_bn (2014), googlenet (2014), resnet50 (2015), densenet121 (2016),
  mobilenet_v3_large (2019), efficientnet_v2_s (2021), convnext_tiny e convnext_large (2022),
  vit_b_16 (2020, transformer), swin_v2_b (2022, transformer), maxvit_t (2022, híbrido).

Dados: uma pasta por classe. Aceita os nomes do notebook (Carcinoma, HSIL, LSIL, Normal) ou
os de Bethesda (SCC, HSIL, LSIL, NILM).

Uso (Kaggle, GPU):
    python comparar_arquiteturas.py --dados "/kaggle/input/.../Liquid Citology + IARC dataset" \\
        --saida resultados_arquiteturas
    python comparar_arquiteturas.py --dados dados --arquiteturas resnet50 convnext_large swin_v2_b
    python comparar_arquiteturas.py --dados dados --so-relatorio     # refaz gráficos e relatório

O processo é retomável: cada (arquitetura, fold) concluído fica gravado e é reaproveitado.

Atenção sobre vazamento: se várias imagens vêm do mesmo paciente, a divisão por imagem coloca
imagens do mesmo paciente no treino e na validação e superestima o desempenho. Se você tiver o
identificador do paciente, passe --grupos arquivo.csv (colunas: arquivo,grupo) para usar
StratifiedGroupKFold. Sem isso, o relatório avisa que a divisão foi por imagem.
"""
import argparse
import copy
import csv
import glob
import json
import os
import platform
import random
import sys
import time
from pathlib import Path

import albumentations as A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from albumentations.pytorch import ToTensorV2
from PIL import Image
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             cohen_kappa_score, confusion_matrix, f1_score, log_loss,
                             matthews_corrcoef, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from matplotlib.ticker import MaxNLocator
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# ------------------------------------------------------------------ classes
# Ordem do notebook (Carcinoma, HSIL, LSIL, Normal) = ordem de app/tools/export_onnx.py (SCC, HSIL, LSIL, NILM)
CLASSES = ["SCC", "HSIL", "LSIL", "NILM"]
ALIAS_PASTA = {"SCC": ["SCC", "Carcinoma"], "HSIL": ["HSIL"], "LSIL": ["LSIL"], "NILM": ["NILM", "Normal"]}
EXTENSOES = ("*.jpg", "*.jpeg", "*.png", "*.tif", "*.tiff", "*.bmp")
MEDIA, DESVIO = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


# ------------------------------------------------------------------ arquiteturas
def _troca(obj, attr, novo):
    setattr(obj, attr, novo)


def _cabeca(n_in, n_out):
    return nn.Sequential(nn.Dropout(0.5), nn.Linear(n_in, n_out))


def _seq(m, idx, n):
    """Substitui m[idx] (Linear) por Dropout+Linear."""
    m[idx] = _cabeca(m[idx].in_features, n)


def _head_alexnet(m, n): _seq(m.classifier, 6, n)
def _head_vgg(m, n): _seq(m.classifier, 6, n)
def _head_googlenet(m, n): m.fc = _cabeca(m.fc.in_features, n)
def _head_resnet(m, n): m.fc = _cabeca(m.fc.in_features, n)
def _head_densenet(m, n): m.classifier = _cabeca(m.classifier.in_features, n)
def _head_mobilenet(m, n): _seq(m.classifier, 3, n)
def _head_efficientnet(m, n): _seq(m.classifier, 1, n)
def _head_convnext(m, n): _seq(m.classifier, 2, n)
def _head_vit(m, n): m.heads.head = _cabeca(m.heads.head.in_features, n)
def _head_swin(m, n): m.head = _cabeca(m.head.in_features, n)
def _head_maxvit(m, n): _seq(m.classifier, 5, n)


# nome: (construtor, função da cabeça, ano, família)
ARQUITETURAS = {
    "alexnet": (models.alexnet, _head_alexnet, 2012, "CNN clássica"),
    "vgg16_bn": (models.vgg16_bn, _head_vgg, 2014, "CNN clássica"),
    "googlenet": (models.googlenet, _head_googlenet, 2014, "CNN (Inception)"),
    "resnet50": (models.resnet50, _head_resnet, 2015, "CNN residual"),
    "densenet121": (models.densenet121, _head_densenet, 2016, "CNN densa"),
    "mobilenet_v3_large": (models.mobilenet_v3_large, _head_mobilenet, 2019, "CNN leve"),
    "efficientnet_v2_s": (models.efficientnet_v2_s, _head_efficientnet, 2021, "CNN eficiente"),
    "convnext_tiny": (models.convnext_tiny, _head_convnext, 2022, "CNN moderna"),
    "convnext_large": (models.convnext_large, _head_convnext, 2022, "CNN moderna"),
    "vit_b_16": (models.vit_b_16, _head_vit, 2020, "Transformer"),
    "swin_v2_b": (models.swin_v2_b, _head_swin, 2022, "Transformer"),
    "maxvit_t": (models.maxvit_t, _head_maxvit, 2022, "Híbrido conv.+atenção"),
}


def criar_modelo(nome, pretreinado):
    construtor, trocar, _, _ = ARQUITETURAS[nome]
    kwargs = {}
    if nome == "googlenet":
        kwargs = dict(aux_logits=False, init_weights=False) if not pretreinado else {}
    modelo = construtor(weights="DEFAULT" if pretreinado else None, **kwargs)
    if nome == "googlenet" and pretreinado:
        modelo.aux_logits, modelo.aux1, modelo.aux2 = False, None, None
    trocar(modelo, len(CLASSES))
    return modelo


# ------------------------------------------------------------------ dados
def carregar_caminhos(raiz, limite=None):
    raiz = Path(raiz)
    caminhos, rotulos = [], []
    for idx, classe in enumerate(CLASSES):
        pasta = next((raiz / n for n in ALIAS_PASTA[classe] if (raiz / n).is_dir()), None)
        if pasta is None:
            raise SystemExit(f"Pasta da classe {classe} não encontrada em {raiz} (nomes aceitos: {ALIAS_PASTA[classe]})")
        arquivos = sorted(f for ext in EXTENSOES for f in glob.glob(str(pasta / ext)))
        if limite:
            arquivos = arquivos[:limite]
        caminhos += arquivos
        rotulos += [idx] * len(arquivos)
    return np.array(caminhos), np.array(rotulos)


def tf_treino():
    return A.Compose([A.Resize(224, 224), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5), A.RandomRotate90(p=0.5),
                      A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
                      A.Normalize(mean=MEDIA, std=DESVIO), ToTensorV2()])


def tf_val():
    return A.Compose([A.Resize(224, 224), A.Normalize(mean=MEDIA, std=DESVIO), ToTensorV2()])


class Imagens(Dataset):
    def __init__(self, caminhos, rotulos, tf):
        self.c, self.r, self.tf = caminhos, rotulos, tf

    def __len__(self):
        return len(self.c)

    def __getitem__(self, i):
        img = np.array(Image.open(self.c[i]).convert("RGB"))
        return self.tf(image=img)["image"], torch.tensor(int(self.r[i]), dtype=torch.long)


def fixar_sementes(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


# ------------------------------------------------------------------ treino de um fold
@torch.no_grad()
def predizer(modelo, carregador, dev):
    modelo.eval()
    perda, probs, reais = 0.0, [], []
    crit = nn.CrossEntropyLoss(reduction="sum")
    for x, y in carregador:
        x, y = x.to(dev), y.to(dev)
        saida = modelo(x)
        perda += crit(saida, y).item()
        probs.append(torch.softmax(saida, 1).cpu().numpy())
        reais.append(y.cpu().numpy())
    probs, reais = np.concatenate(probs), np.concatenate(reais)
    return perda / len(reais), probs, reais


def treinar_fold(nome, args, dev, tr_x, tr_y, va_x, va_y, rotulo):
    g = torch.Generator().manual_seed(args.semente)
    ca_tr = DataLoader(Imagens(tr_x, tr_y, tf_treino()), batch_size=args.lote, shuffle=True, num_workers=args.trabalhadores,
                       generator=g, pin_memory=dev.type == "cuda")
    ca_va = DataLoader(Imagens(va_x, va_y, tf_val()), batch_size=args.lote, shuffle=False, num_workers=args.trabalhadores)
    modelo = criar_modelo(nome, not args.sem_pretreino).to(dev)
    lr = args.lr_por_arquitetura.get(nome, args.lr)
    otim = torch.optim.Adam(modelo.parameters(), lr=lr, weight_decay=args.decaimento)
    crit = nn.CrossEntropyLoss()
    hist = dict(perda_treino=[], acc_treino=[], perda_val=[], acc_val=[], tempo_epoca=[])
    melhor_acc, melhor_peso, melhor_ep, sem = -1.0, None, 0, 0
    for ep in range(1, args.max_epocas + 1):
        t0 = time.time()
        modelo.train()
        pt, ac, n = 0.0, 0, 0
        for x, y in ca_tr:
            x, y = x.to(dev), y.to(dev)
            otim.zero_grad()
            saida = modelo(x)
            perda = crit(saida, y)
            perda.backward()
            otim.step()
            pt += perda.item() * len(y); ac += (saida.argmax(1) == y).sum().item(); n += len(y)
        pv, probs, reais = predizer(modelo, ca_va, dev)
        acc_v = float(np.mean(probs.argmax(1) == reais))
        for k, v in zip(("perda_treino", "acc_treino", "perda_val", "acc_val", "tempo_epoca"),
                        (pt / n, ac / n, pv, acc_v, time.time() - t0)):
            hist[k].append(float(v))
        print(f"[{rotulo}] época {ep:02d}/{args.max_epocas}  treino {ac / n:.4f}/{pt / n:.4f}  val {acc_v:.4f}/{pv:.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        if acc_v > melhor_acc:
            melhor_acc, melhor_ep, sem = acc_v, ep, 0
            melhor_peso = copy.deepcopy(modelo.state_dict())
        else:
            sem += 1
            if sem >= args.paciencia:
                print(f"[{rotulo}] parada antecipada na época {ep} (melhor: {melhor_ep})")
                break
    modelo.load_state_dict(melhor_peso)
    _, probs, reais = predizer(modelo, ca_va, dev)
    if args.salvar_pesos:
        k = rotulo.split(" fold ")[1].split("/")[0]
        torch.save(melhor_peso, Path(args.saida) / nome / f"fold{k}.pt")
    return modelo, hist, melhor_ep, probs, reais


def medir_modelo(modelo, dev):
    """Parâmetros, tamanho e latência de inferência (lote 1)."""
    n = sum(p.numel() for p in modelo.parameters())
    x = torch.rand(1, 3, 224, 224, device=dev)
    modelo.eval()
    with torch.no_grad():
        for _ in range(3):
            modelo(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(10):
            modelo(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
    return dict(parametros_milhoes=n / 1e6, tamanho_mb=n * 4 / 1e6, latencia_ms=(time.time() - t0) / 10 * 1000)


# ------------------------------------------------------------------ execução de uma arquitetura
def executar_arquitetura(nome, args, dev, caminhos, rotulos, grupos):
    pasta = Path(args.saida) / nome
    pasta.mkdir(parents=True, exist_ok=True)
    if grupos is not None:
        divisor = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.semente).split(caminhos, rotulos, grupos)
    else:
        divisor = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.semente).split(caminhos, rotulos)
    for k, (tr, va) in enumerate(divisor, 1):
        arq = pasta / f"fold{k}.json"
        if arq.exists():
            print(f"[{nome} fold {k}] já concluído, reaproveitando")
            continue
        modelo, hist, melhor_ep, probs, reais = treinar_fold(nome, args, dev, caminhos[tr], rotulos[tr], caminhos[va], rotulos[va],
                                                             f"{nome} fold {k}/{args.folds}")
        info = medir_modelo(modelo, dev) if k == 1 else {}
        json.dump(dict(fold=k, indices_val=va.tolist(), probs=probs.tolist(), reais=reais.tolist(), melhor_epoca=melhor_ep,
                       epocas_executadas=len(hist["acc_val"]), historico=hist, **info), open(arq, "w"))
        del modelo
        if dev.type == "cuda":
            torch.cuda.empty_cache()


# ------------------------------------------------------------------ análise
def carregar_resultados(args, nomes, n_total):
    res = {}
    for nome in nomes:
        folds = sorted(Path(args.saida, nome).glob("fold*.json"), key=lambda p: int(p.stem[4:]))
        if len(folds) < args.folds:
            continue
        dados = [json.load(open(f)) for f in folds]
        probs = np.zeros((n_total, len(CLASSES)))
        reais = np.full(n_total, -1)
        for d in dados:
            probs[d["indices_val"]] = d["probs"]
            reais[d["indices_val"]] = d["reais"]
        res[nome] = dict(folds=dados, probs=probs, reais=reais, pred=probs.argmax(1))
    return res


def metricas(r):
    y, p, pr = r["reais"], r["pred"], r["probs"]
    pr = pr / pr.sum(1, keepdims=True)
    por_fold = []
    for d in r["folds"]:
        yf, pf = np.array(d["reais"]), np.array(d["probs"]).argmax(1)
        por_fold.append(accuracy_score(yf, pf))
    rng = np.random.default_rng(0)
    acertos = (y == p).astype(float)
    boot = [acertos[rng.integers(0, len(y), len(y))].mean() for _ in range(2000)]
    rel = classification_report(y, p, target_names=CLASSES, output_dict=True, zero_division=0)
    info = next((d for d in r["folds"] if "parametros_milhoes" in d), {})
    return dict(
        acc_media=float(np.mean(por_fold)), acc_dp=float(np.std(por_fold)), acc_global=float(accuracy_score(y, p)),
        ic95=(float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))),
        acc_balanceada=float(balanced_accuracy_score(y, p)),
        f1_macro=float(f1_score(y, p, average="macro")), f1_ponderado=float(f1_score(y, p, average="weighted")),
        precisao_macro=float(precision_score(y, p, average="macro", zero_division=0)),
        recall_macro=float(recall_score(y, p, average="macro", zero_division=0)),
        kappa=float(cohen_kappa_score(y, p)), mcc=float(matthews_corrcoef(y, p)),
        auc_macro=float(roc_auc_score(y, pr, multi_class="ovr", average="macro")),
        log_loss=float(log_loss(y, np.clip(pr, 1e-7, 1), labels=range(len(CLASSES)))),
        por_classe={c: rel[c] for c in CLASSES}, por_fold=[float(a) for a in por_fold],
        melhor_epoca_media=float(np.mean([d["melhor_epoca"] for d in r["folds"]])),
        epocas_media=float(np.mean([d["epocas_executadas"] for d in r["folds"]])),
        tempo_epoca_s=float(np.mean([np.mean(d["historico"]["tempo_epoca"]) for d in r["folds"]])),
        matriz=confusion_matrix(y, p, labels=range(len(CLASSES))).tolist(), **info)


def mcnemar(r_a, r_b):
    ok_a, ok_b = r_a["pred"] == r_a["reais"], r_b["pred"] == r_b["reais"]
    b, c = int(np.sum(ok_a & ~ok_b)), int(np.sum(~ok_a & ok_b))
    return b, c, (1.0 if b + c == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue))


# ------------------------------------------------------------------ gráficos
def _media_por_epoca(folds, chave):
    n = max(len(d["historico"][chave]) for d in folds)
    m = np.full((len(folds), n), np.nan)
    for i, d in enumerate(folds):
        v = d["historico"][chave]
        m[i, :len(v)] = v
    return np.nanmean(m, axis=0)


def grafico_curvas(nome, folds, destino):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    cores = plt.cm.tab10.colors
    for i, d in enumerate(folds):
        h, ep = d["historico"], range(1, len(d["historico"]["acc_val"]) + 1)
        a1.plot(ep, h["acc_val"], color=cores[i], marker="o", ms=3, label=f"Fold {d['fold']} (validação)")
        a1.plot(ep, h["acc_treino"], color=cores[i], ls="--", alpha=0.7)
        a2.plot(ep, h["perda_val"], color=cores[i], marker="o", ms=3, label=f"Fold {d['fold']} (validação)")
        a2.plot(ep, h["perda_treino"], color=cores[i], ls="--", alpha=0.7)
        a1.axvline(d["melhor_epoca"], color=cores[i], ls=":", alpha=0.4)
    a1.set_title(f"{nome}: acurácia por época"); a1.set_xlabel("Época"); a1.set_ylabel("Acurácia"); a1.set_ylim(0.3, 1.01)
    a2.set_title(f"{nome}: perda por época"); a2.set_xlabel("Época"); a2.set_ylabel("Perda (entropia cruzada)")
    for a in (a1, a2):
        a.grid(alpha=0.3); a.xaxis.set_major_locator(MaxNLocator(integer=True))
    a1.legend(fontsize=8, title="Contínua: validação · tracejada: treino · pontilhada: melhor época", title_fontsize=7)
    fig.tight_layout(); fig.savefig(destino, dpi=150); plt.close(fig)


def grafico_matriz(nome, cm, destino):
    cm = np.array(cm)
    norm = cm / cm.sum(1, keepdims=True).clip(1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    for a, M, tit, fmt in ((a1, cm, "Contagens", "d"), (a2, norm, "Normalizada por linha (revocação)", ".2f")):
        a.imshow(M, cmap="Blues", vmin=0)
        a.set_xticks(range(4)); a.set_xticklabels(CLASSES); a.set_yticks(range(4)); a.set_yticklabels(CLASSES)
        a.set_xlabel("Classe predita"); a.set_ylabel("Classe verdadeira"); a.set_title(tit)
        for i in range(4):
            for j in range(4):
                a.text(j, i, format(M[i, j], fmt), ha="center", va="center", color="white" if M[i, j] > M.max() * 0.55 else "black")
    fig.suptitle(f"{nome}: matriz de confusão global ({int(cm.sum())} imagens)")
    fig.tight_layout(); fig.savefig(destino, dpi=150); plt.close(fig)


def grafico_comparativo(res, mt, pasta):
    ordem = sorted(res, key=lambda n: -mt[n]["acc_media"])
    rot = [f"{n}\n({ARQUITETURAS[n][2]})" for n in ordem]
    # 1) acurácia ± dp
    fig, ax = plt.subplots(figsize=(max(8, 1.0 * len(ordem) + 3), 4.8))
    v = [mt[n]["acc_media"] * 100 for n in ordem]
    ax.bar(range(len(ordem)), v, yerr=[mt[n]["acc_dp"] * 100 for n in ordem], capsize=3, color="#4C1D95")
    for i, x in enumerate(v):
        ax.text(i, x + 0.4, f"{x:.1f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(ordem))); ax.set_xticklabels(rot, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(max(0, min(v) - 8), 101); ax.set_ylabel("Acurácia (%) média ± desvio entre folds")
    ax.set_title("Acurácia por arquitetura (validação cruzada)"); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(pasta / "comparativo_acuracia.png", dpi=150); plt.close(fig)
    # 2) métricas lado a lado
    fig, ax = plt.subplots(figsize=(max(9, 1.2 * len(ordem) + 3), 4.8))
    w = 0.2
    for k, (chave, nomem, cor) in enumerate((("f1_macro", "F1 macro", "#4C1D95"), ("kappa", "Kappa", "#C084FC"),
                                             ("mcc", "MCC", "#64748B"), ("auc_macro", "AUC macro", "#0E9F6E"))):
        ax.bar(np.arange(len(ordem)) + (k - 1.5) * w, [mt[n][chave] for n in ordem], w, label=nomem, color=cor)
    ax.set_xticks(range(len(ordem))); ax.set_xticklabels(rot, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.02); ax.legend(ncol=4, fontsize=8); ax.grid(axis="y", alpha=0.3)
    ax.set_title("F1 macro, kappa, MCC e AUC por arquitetura"); fig.tight_layout()
    fig.savefig(pasta / "comparativo_metricas.png", dpi=150); plt.close(fig)
    # 3) acurácia x parâmetros e latência
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
    for n in ordem:
        m = mt[n]
        if "parametros_milhoes" not in m:
            continue
        marc = "s" if "Transformer" in ARQUITETURAS[n][3] or "Híbrido" in ARQUITETURAS[n][3] else "o"
        a1.scatter(m["parametros_milhoes"], m["acc_media"] * 100, marker=marc, s=60, color="#4C1D95")
        a1.annotate(n, (m["parametros_milhoes"], m["acc_media"] * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
        a2.scatter(m["latencia_ms"], m["acc_media"] * 100, marker=marc, s=60, color="#4C1D95")
        a2.annotate(n, (m["latencia_ms"], m["acc_media"] * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
    a1.set_xscale("log"); a1.set_xlabel("Parâmetros (milhões, escala log)"); a2.set_xlabel("Latência de inferência (ms/imagem, lote 1)")
    for a in (a1, a2):
        a.set_ylabel("Acurácia (%)"); a.grid(alpha=0.3)
    a1.set_title("Acurácia × tamanho do modelo (quadrado: transformer/híbrido)"); a2.set_title("Acurácia × latência")
    fig.tight_layout(); fig.savefig(pasta / "comparativo_custo.png", dpi=150); plt.close(fig)
    # 4) curva de acurácia de validação por época, todas as arquiteturas
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.8))
    cm_ = plt.cm.tab20.colors
    for i, n in enumerate(ordem):
        a1.plot(range(1, 1 + len(_media_por_epoca(res[n]["folds"], "acc_val"))), _media_por_epoca(res[n]["folds"], "acc_val"),
                color=cm_[i % 20], marker="o", ms=2.5, label=n)
        a2.plot(range(1, 1 + len(_media_por_epoca(res[n]["folds"], "perda_val"))), _media_por_epoca(res[n]["folds"], "perda_val"),
                color=cm_[i % 20], marker="o", ms=2.5, label=n)
    a1.set_title("Acurácia de validação por época (média dos folds ainda em treino)"); a2.set_title("Perda de validação por época")
    for a in (a1, a2):
        a.set_xlabel("Época"); a.grid(alpha=0.3)
    a1.set_ylabel("Acurácia"); a2.set_ylabel("Perda"); a1.set_ylim(0.3, 1.01); a2.set_yscale("log")
    a1.legend(fontsize=7, ncol=2)
    fig.tight_layout(); fig.savefig(pasta / "comparativo_curvas.png", dpi=150); plt.close(fig)
    # 5) revocação por classe
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(ordem) + 1.8))
    M = np.array([[mt[n]["por_classe"][c]["recall"] for c in CLASSES] for n in ordem])
    im = ax.imshow(M, cmap="Purples", vmin=0.5, vmax=1)
    ax.set_xticks(range(4)); ax.set_xticklabels(CLASSES); ax.set_yticks(range(len(ordem))); ax.set_yticklabels(ordem)
    for i in range(len(ordem)):
        for j in range(4):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8, color="white" if M[i, j] > 0.85 else "black")
    ax.set_title("Revocação (sensibilidade) por classe"); fig.colorbar(im, shrink=0.8); fig.tight_layout()
    fig.savefig(pasta / "comparativo_recall_classe.png", dpi=150); plt.close(fig)
    return ordem


# ------------------------------------------------------------------ relatório
def pct(x): return f"{x * 100:.2f}%".replace(".", ",")
def n3(x): return f"{x:.3f}".replace(".", ",")


def montar_relatorio(args, res, mt, ordem, contagem, cfg, grupos_usados):
    P = Path(args.saida)
    b = []  # blocos: ("h1"|"h2"|"p"|"img"|"table", ...)
    b.append(("h1", "Comparação de arquiteturas de redes neurais para citologia em meio líquido"))
    b.append(("p", f"Gerado automaticamente em {time.strftime('%d/%m/%Y %H:%M')} por ml/comparar_arquiteturas.py."))
    b.append(("h2", "1. Protocolo"))
    b.append(("p", f"Imagens: {sum(contagem.values())} ({', '.join(f'{c}: {n}' for c, n in contagem.items())}). "
                   "Todas as arquiteturas usam exatamente os mesmos folds, aumentos de dados, otimizador e critério de parada, "
                   "de modo que as diferenças refletem a arquitetura e não o protocolo."))
    b.append(("table", ["Item", "Valor"], [
        ["Validação", f"{'StratifiedGroupKFold' if grupos_usados else 'StratifiedKFold (divisão por imagem)'}, {args.folds} folds, semente {args.semente}"],
        ["Otimizador", f"Adam, lr {args.lr}, decaimento de pesos {args.decaimento}"],
        ["Lote / épocas", f"{args.lote} / até {args.max_epocas}"],
        ["Parada antecipada", f"paciência {args.paciencia} sobre a acurácia de validação; restaura a melhor época"],
        ["Entrada", "224×224, Resize (INTER_LINEAR), normalização do ImageNet"],
        ["Aumentos (treino)", "giro horizontal e vertical (p=0,5), rotação de 90° (p=0,5), ColorJitter 0,2/0,2/0,2/0,1 (p=0,3)"],
        ["Cabeça", "Dropout(0,5) + Linear(4), sobre pesos pré-treinados no ImageNet" + (" (DESLIGADO neste teste)" if args.sem_pretreino else "")],
        ["Ambiente", f"Python {cfg['python']}, torch {cfg['torch']}, torchvision {cfg['torchvision']}, {cfg['dispositivo']}"]]))
    if not grupos_usados:
        b.append(("p", "AVISO: a divisão dos folds foi por imagem. Se imagens do mesmo paciente aparecem no treino e na validação, "
                       "os números abaixo são otimistas. Use --grupos para dividir por paciente."))
    b.append(("h2", "2. Arquiteturas avaliadas"))
    b.append(("table", ["Arquitetura", "Ano", "Família", "Parâmetros (M)", "Tamanho (MB)", "Latência (ms)"],
              [[n, str(ARQUITETURAS[n][2]), ARQUITETURAS[n][3], f"{mt[n].get('parametros_milhoes', float('nan')):.1f}",
                f"{mt[n].get('tamanho_mb', float('nan')):.0f}", f"{mt[n].get('latencia_ms', float('nan')):.1f}"]
               for n in sorted(ordem, key=lambda n: ARQUITETURAS[n][2])]))
    b.append(("p", "Latência: média de 10 inferências com lote 1 no dispositivo do treino; serve para comparar, não como medida de campo."))
    b.append(("h2", "3. Resultados da validação cruzada"))
    b.append(("table", ["Arquitetura", "Acurácia (média ± dp)", "IC 95% (bootstrap)", "Acurácia balanceada", "F1 macro", "Kappa", "MCC", "AUC macro", "Perda log."],
              [[n, f"{pct(mt[n]['acc_media'])} ± {pct(mt[n]['acc_dp'])}", f"{pct(mt[n]['ic95'][0])} a {pct(mt[n]['ic95'][1])}",
                pct(mt[n]["acc_balanceada"]), n3(mt[n]["f1_macro"]), n3(mt[n]["kappa"]), n3(mt[n]["mcc"]), n3(mt[n]["auc_macro"]),
                n3(mt[n]["log_loss"])] for n in ordem]))
    b.append(("p", "A acurácia média e o desvio são calculados sobre as acurácias dos folds; o intervalo de confiança vem de 2.000 reamostragens "
                   "(bootstrap) das predições fora da amostra. Kappa = kappa de Cohen; MCC = coeficiente de correlação de Matthews; "
                   "AUC = área sob a curva ROC um-contra-todos, média das classes."))
    b.append(("img", P / "comparativo_acuracia.png", "Acurácia por arquitetura."))
    b.append(("img", P / "comparativo_metricas.png", "F1 macro, kappa, MCC e AUC por arquitetura."))
    b.append(("table", ["Arquitetura"] + [f"Fold {i + 1}" for i in range(args.folds)] + ["Melhor época (média)", "Épocas (média)", "s/época"],
              [[n] + [pct(a) for a in mt[n]["por_fold"]] + [f"{mt[n]['melhor_epoca_media']:.1f}", f"{mt[n]['epocas_media']:.1f}",
                                                           f"{mt[n]['tempo_epoca_s']:.0f}"] for n in ordem]))
    b.append(("h2", "4. Acurácia e perda em função das épocas"))
    b.append(("p", "Linhas contínuas: validação; tracejadas: treino; linhas verticais pontilhadas: melhor época de cada fold."))
    b.append(("img", P / "comparativo_curvas.png", "Acurácia e perda de validação por época, todas as arquiteturas (média dos folds ainda em treino em cada época)."))
    for n in ordem:
        b.append(("img", P / n / "curvas.png", f"{n}: acurácia e perda por época nos {args.folds} folds."))
    b.append(("h2", "5. Desempenho por classe"))
    b.append(("table", ["Arquitetura"] + [f"Recall {c}" for c in CLASSES] + [f"Precisão {c}" for c in CLASSES],
              [[n] + [pct(mt[n]["por_classe"][c]["recall"]) for c in CLASSES] + [pct(mt[n]["por_classe"][c]["precision"]) for c in CLASSES]
               for n in ordem]))
    b.append(("p", "Em rastreio, o recall de HSIL e SCC é a medida decisiva: quantos casos graves escapariam."))
    b.append(("img", P / "comparativo_recall_classe.png", "Revocação por classe e arquitetura."))
    b.append(("h2", "6. Matrizes de confusão"))
    for n in ordem:
        b.append(("img", P / n / "matriz_confusao.png", f"{n}: matriz de confusão global (predições fora da amostra de todos os folds)."))
    b.append(("h2", "7. Comparação estatística entre arquiteturas"))
    melhor = ordem[0]
    linhas = []
    for n in ordem[1:]:
        bb, cc, p = mcnemar(res[melhor], res[n])
        linhas.append([n, str(bb), str(cc), f"{p:.4f}".replace(".", ","), "sim" if p < 0.05 else "não"])
    b.append(("p", f"Teste de McNemar exato entre a melhor arquitetura ({melhor}) e cada uma das demais, sobre as mesmas {sum(contagem.values())} predições "
                   "fora da amostra. 'Só a melhor acerta' e 'só a outra acerta' contam as imagens em que apenas uma das duas acerta. "
                   "Sem correção para comparações múltiplas; com muitas arquiteturas, p entre 0,01 e 0,05 deve ser lido com cautela."))
    b.append(("table", ["Arquitetura", "Só a melhor acerta", "Só a outra acerta", "p (McNemar exato)", "Diferença p<0,05"], linhas))
    b.append(("h2", "8. Custo e desempenho"))
    b.append(("img", P / "comparativo_custo.png", "Acurácia em função do número de parâmetros e da latência."))
    b.append(("h2", "9. Limitações"))
    for t in ["Validação interna: os folds saem do mesmo acervo; não há validação externa com lâminas de outro laboratório.",
              "Divisão por imagem (a menos que --grupos tenha sido usado) pode vazar imagens do mesmo paciente entre treino e validação.",
              "Uma única semente de divisão e de inicialização por arquitetura; diferenças pequenas (< 1 ponto percentual) estão dentro do ruído.",
              "Os mesmos hiperparâmetros para todas as arquiteturas favorecem as que gostam de lr 1e-4 com Adam; transformers podem melhorar com outros valores (use --lr-por-arquitetura).",
              "A acurácia usada na parada antecipada é a mesma da validação reportada, o que torna a estimativa levemente otimista; para uma estimativa limpa, reserve um conjunto de teste à parte.",
              "Imagens são campos isolados, não lâminas inteiras; o resultado não se traduz diretamente em desempenho diagnóstico."]:
        b.append(("li", t))
    b.append(("h2", "10. Arquivos gerados"))
    b.append(("p", "resultados.csv (todas as métricas por arquitetura); <arquitetura>/fold*.json (histórico por época, probabilidades e índices de validação de cada fold); "
                   "<arquitetura>/curvas.png e matriz_confusao.png; comparativo_*.png; config.json (parâmetros e ambiente); relatorio.md e relatorio.docx."))
    # Markdown
    md = []
    for blk in b:
        t = blk[0]
        if t == "h1": md.append(f"# {blk[1]}\n")
        elif t == "h2": md.append(f"\n## {blk[1]}\n")
        elif t == "p": md.append(blk[1] + "\n")
        elif t == "li": md.append(f"- {blk[1]}")
        elif t == "img": md.append(f"![{blk[2]}]({Path(blk[1]).relative_to(P).as_posix()})\n\n*{blk[2]}*\n")
        elif t == "table":
            md.append("| " + " | ".join(blk[1]) + " |\n|" + "---|" * len(blk[1]))
            md += ["| " + " | ".join(r) + " |" for r in blk[2]]
            md.append("")
    (P / "relatorio.md").write_text("\n".join(md), encoding="utf-8")
    # DOCX (opcional)
    try:
        from docx import Document
        from docx.shared import Cm
        d = Document()
        for blk in b:
            t = blk[0]
            if t == "h1": d.add_heading(blk[1], 0)
            elif t == "h2": d.add_heading(blk[1], 1)
            elif t == "p": d.add_paragraph(blk[1])
            elif t == "li": d.add_paragraph(blk[1], style="List Bullet")
            elif t == "img":
                d.add_picture(str(blk[1]), width=Cm(16)); c = d.add_paragraph(blk[2]); c.runs[0].italic = True
            elif t == "table":
                tb = d.add_table(rows=1, cols=len(blk[1])); tb.style = "Light Grid Accent 1"
                for i, h in enumerate(blk[1]): tb.rows[0].cells[i].text = h
                for r in blk[2]:
                    cells = tb.add_row().cells
                    for i, v in enumerate(r): cells[i].text = v
                d.add_paragraph()
        d.save(P / "relatorio.docx")
    except ImportError:
        print("python-docx ausente: relatorio.docx não gerado (pip install python-docx)")


def analisar(args, nomes, caminhos, rotulos, contagem, cfg, grupos_usados):
    res = carregar_resultados(args, nomes, len(caminhos))
    if not res:
        raise SystemExit("Nenhuma arquitetura com todos os folds concluídos em " + args.saida)
    P = Path(args.saida)
    mt = {n: metricas(r) for n, r in res.items()}
    for n, r in res.items():
        grafico_curvas(n, r["folds"], P / n / "curvas.png")
        grafico_matriz(n, mt[n]["matriz"], P / n / "matriz_confusao.png")
    ordem = grafico_comparativo(res, mt, P)
    with open(P / "resultados.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["arquitetura", "ano", "familia", "acc_media", "acc_dp", "acc_global", "ic95_inf", "ic95_sup", "acc_balanceada", "f1_macro",
                    "f1_ponderado", "precisao_macro", "recall_macro", "kappa", "mcc", "auc_macro", "log_loss", "melhor_epoca_media",
                    "epocas_media", "tempo_epoca_s", "parametros_milhoes", "tamanho_mb", "latencia_ms"] +
                   [f"recall_{c}" for c in CLASSES] + [f"precisao_{c}" for c in CLASSES])
        for n in ordem:
            m = mt[n]
            w.writerow([n, ARQUITETURAS[n][2], ARQUITETURAS[n][3], m["acc_media"], m["acc_dp"], m["acc_global"], m["ic95"][0], m["ic95"][1],
                        m["acc_balanceada"], m["f1_macro"], m["f1_ponderado"], m["precisao_macro"], m["recall_macro"], m["kappa"], m["mcc"],
                        m["auc_macro"], m["log_loss"], m["melhor_epoca_media"], m["epocas_media"], m["tempo_epoca_s"],
                        m.get("parametros_milhoes", ""), m.get("tamanho_mb", ""), m.get("latencia_ms", "")] +
                       [m["por_classe"][c]["recall"] for c in CLASSES] + [m["por_classe"][c]["precision"] for c in CLASSES])
    montar_relatorio(args, res, mt, ordem, contagem, cfg, grupos_usados)
    print("\nRanking por acurácia média:")
    for n in ordem:
        print(f"  {n:20s} {mt[n]['acc_media'] * 100:6.2f}% ± {mt[n]['acc_dp'] * 100:.2f}   F1 macro {mt[n]['f1_macro']:.4f}   kappa {mt[n]['kappa']:.4f}")
    print("Relatório em", P / "relatorio.md")


# ------------------------------------------------------------------ main
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", required=True)
    p.add_argument("--saida", default="resultados_arquiteturas")
    p.add_argument("--arquiteturas", nargs="+", default=list(ARQUITETURAS), choices=list(ARQUITETURAS))
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--max-epocas", type=int, default=50)
    p.add_argument("--paciencia", type=int, default=5)
    p.add_argument("--lote", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--lr-por-arquitetura", type=json.loads, default={}, help='JSON, ex.: \'{"vit_b_16": 3e-5}\'')
    p.add_argument("--decaimento", type=float, default=1e-5)
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--trabalhadores", type=int, default=2)
    p.add_argument("--grupos", help="CSV arquivo,grupo para dividir por paciente (StratifiedGroupKFold)")
    p.add_argument("--salvar-pesos", action="store_true", help="grava <arquitetura>/fold<N>.pt (melhor época); necessário para anotar_celulas.py")
    p.add_argument("--sem-pretreino", action="store_true", help="não baixa pesos do ImageNet (apenas testes)")
    p.add_argument("--limite-por-classe", type=int, help="usa só N imagens por classe (apenas testes)")
    p.add_argument("--so-relatorio", action="store_true", help="não treina; refaz análise, gráficos e relatório")
    args = p.parse_args()

    fixar_sementes(args.semente)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.saida).mkdir(parents=True, exist_ok=True)
    caminhos, rotulos = carregar_caminhos(args.dados, args.limite_por_classe)
    contagem = {c: int((rotulos == i).sum()) for i, c in enumerate(CLASSES)}
    grupos = None
    if args.grupos:
        mapa = {Path(r[0]).name: r[1] for r in csv.reader(open(args.grupos, encoding="utf-8-sig")) if r and r[0] != "arquivo"}
        faltam = [c for c in caminhos if Path(c).name not in mapa]
        if faltam:
            raise SystemExit(f"{len(faltam)} imagens sem grupo em {args.grupos} (ex.: {faltam[0]})")
        grupos = np.array([mapa[Path(c).name] for c in caminhos])
    cfg = dict(python=platform.python_version(), torch=torch.__version__, torchvision=__import__("torchvision").__version__,
               dispositivo=(torch.cuda.get_device_name(0) if dev.type == "cuda" else "CPU"), argumentos=vars(args), contagem=contagem)
    json.dump(cfg, open(Path(args.saida) / "config.json", "w"), indent=1, ensure_ascii=False, default=str)
    print(f"{len(caminhos)} imagens {contagem}; dispositivo: {cfg['dispositivo']}; arquiteturas: {args.arquiteturas}")
    if not args.so_relatorio:
        for nome in args.arquiteturas:
            print("\n" + "=" * 70 + f"\nARQUITETURA: {nome}\n" + "=" * 70)
            executar_arquitetura(nome, args, dev, caminhos, rotulos, grupos)
    analisar(args, args.arquiteturas, caminhos, rotulos, contagem, cfg, grupos is not None)


if __name__ == "__main__":
    main()
