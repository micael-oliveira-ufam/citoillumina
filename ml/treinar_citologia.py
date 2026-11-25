#!/usr/bin/env python3
"""Treino do classificador de citologia em meio líquido (NILM, LSIL, HSIL, SCC).

ConvNeXt (torchvision) pré-treinado no ImageNet, ajustado em 4 classes de Bethesda com
validação cruzada estratificada em 5 folds, como descrito no relatório do projeto:

  * entrada 224x224, redimensionamento com cv2.INTER_LINEAR (o mesmo do site e do app);
  * aumentos: giro horizontal e vertical, rotação de 90°, ColorJitter;
  * normalização com as estatísticas do ImageNet;
  * parada antecipada após 5 épocas sem melhora na acurácia de validação, restaurando
    os pesos da melhor época em cada fold;
  * cabeça Sequential(Dropout(0,5), Linear), a mesma que app/tools/export_onnx.py espera.

ATENÇÃO: este script é uma reimplementação a partir dos parâmetros documentados do projeto.
O script original do treino (e os hiperparâmetros que não estão documentados: otimizador,
taxa de aprendizado, tamanho do lote, número máximo de épocas) não estão neste repositório.
Os valores padrão abaixo são suposições razoáveis e estão marcados com [suposição]; ajuste-os
se tiver os originais. Resultados novos podem diferir dos publicados.

Organização esperada dos dados (uma pasta por classe, nomes exatos):

    dados/
      NILM/  *.jpg|png
      LSIL/  ...
      HSIL/  ...
      SCC/   ...

Uso:
    python ml/treinar_citologia.py --dados dados --saida saidas_citologia
    python ml/treinar_citologia.py --dados dados --saida saidas_citologia --treinar-final

Saídas (em --saida): fold{N}.pt (state_dict), historico_fold{N}.json, metricas_cv.json,
predicoes_oof.csv, matriz_confusao.csv e, com --treinar-final, citologia_convnext_final.pt,
que pode ser exportado com:  python app/tools/export_onnx.py --checkpoint <arquivo>.pt ...

Dependências: torch, torchvision, albumentations, opencv-python-headless, scikit-learn, pandas.
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import models

# Mesma ordem de app/tools/export_onnx.py (CLASS_NAMES): o índice da saída é o da lista.
CLASSES = ["SCC", "HSIL", "LSIL", "NILM"]
EXTENSOES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEDIA = (0.485, 0.456, 0.406)
DESVIO = (0.229, 0.224, 0.225)
TAMANHO = 224


def fixar_sementes(semente):
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)
    torch.cuda.manual_seed_all(semente)


def listar_imagens(raiz):
    """Devolve (caminhos, rótulos) a partir de raiz/<CLASSE>/*."""
    raiz = Path(raiz)
    caminhos, rotulos = [], []
    for idx, nome in enumerate(CLASSES):
        pasta = raiz / nome
        if not pasta.is_dir():
            raise SystemExit(f"Pasta ausente: {pasta} (esperadas: {', '.join(CLASSES)})")
        arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in EXTENSOES)
        if not arquivos:
            raise SystemExit(f"Nenhuma imagem em {pasta}")
        caminhos += arquivos
        rotulos += [idx] * len(arquivos)
    return caminhos, np.array(rotulos)


def transformacoes(treino):
    base = [A.Resize(TAMANHO, TAMANHO, interpolation=cv2.INTER_LINEAR)]
    if treino:
        base += [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.02, p=0.5),  # [suposição]
        ]
    base += [A.Normalize(mean=MEDIA, std=DESVIO), A.ToTensorV2()] if hasattr(A, "ToTensorV2") else \
            [A.Normalize(mean=MEDIA, std=DESVIO)]
    return A.Compose(base)


class Imagens(Dataset):
    def __init__(self, caminhos, rotulos, treino):
        self.caminhos, self.rotulos, self.tf = caminhos, rotulos, transformacoes(treino)

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.caminhos[i])), cv2.COLOR_BGR2RGB)
        x = self.tf(image=img)["image"]
        if not torch.is_tensor(x):
            x = torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1))).float()
        return x, int(self.rotulos[i])


def criar_modelo(variante, pretreinado):
    """ConvNeXt com a cabeça Sequential(Dropout, Linear) lida por export_onnx.py."""
    pesos = "DEFAULT" if pretreinado else None
    modelo = getattr(models, variante)(weights=pesos)
    entradas = modelo.classifier[2].in_features
    modelo.classifier[2] = nn.Sequential(nn.Dropout(0.5), nn.Linear(entradas, len(CLASSES)))
    return modelo


@torch.no_grad()
def avaliar(modelo, carregador, dispositivo, criterio):
    modelo.eval()
    perdas, preds, reais, probs = 0.0, [], [], []
    for x, y in carregador:
        x, y = x.to(dispositivo), y.to(dispositivo)
        saida = modelo(x)
        perdas += criterio(saida, y).item() * len(y)
        probs.append(torch.softmax(saida, 1).cpu().numpy())
        preds += saida.argmax(1).cpu().tolist()
        reais += y.cpu().tolist()
    n = len(reais)
    return perdas / n, float(np.mean(np.array(preds) == np.array(reais))), np.array(preds), np.array(reais), np.concatenate(probs)


def treinar_fold(args, dispositivo, tr_x, tr_y, va_x, va_y, nome):
    ds_tr, ds_va = Imagens(tr_x, tr_y, True), Imagens(va_x, va_y, False)
    ca_tr = DataLoader(ds_tr, batch_size=args.lote, shuffle=True, num_workers=args.trabalhadores,
                       pin_memory=dispositivo.type == "cuda", drop_last=len(ds_tr) > args.lote)
    ca_va = DataLoader(ds_va, batch_size=args.lote, shuffle=False, num_workers=args.trabalhadores)
    modelo = criar_modelo(args.modelo, not args.sem_pretreino).to(dispositivo)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.decaimento)  # [suposição]
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=args.max_epocas)       # [suposição]
    escala = torch.amp.GradScaler("cuda", enabled=dispositivo.type == "cuda")
    historico, melhor_acc, melhor_peso, melhor_ep, sem_melhora = [], -1.0, None, 0, 0
    for ep in range(1, args.max_epocas + 1):
        t0 = time.time()
        modelo.train()
        perda_tr, acertos, total = 0.0, 0, 0
        for x, y in ca_tr:
            x, y = x.to(dispositivo, non_blocking=True), y.to(dispositivo, non_blocking=True)
            otimizador.zero_grad(set_to_none=True)
            with torch.autocast(device_type=dispositivo.type, enabled=dispositivo.type == "cuda"):
                saida = modelo(x)
                perda = criterio(saida, y)
            escala.scale(perda).backward()
            escala.step(otimizador)
            escala.update()
            perda_tr += perda.item() * len(y)
            acertos += (saida.argmax(1) == y).sum().item()
            total += len(y)
        agendador.step()
        perda_va, acc_va, *_ = avaliar(modelo, ca_va, dispositivo, criterio)
        historico.append(dict(epoca=ep, perda_treino=perda_tr / total, acc_treino=acertos / total,
                              perda_val=perda_va, acc_val=acc_va))
        print(f"[{nome}] época {ep:02d}  treino {acertos / total:.4f}/{perda_tr / total:.4f}  "
              f"val {acc_va:.4f}/{perda_va:.4f}  ({time.time() - t0:.0f}s)", flush=True)
        if acc_va > melhor_acc:
            melhor_acc, melhor_ep, sem_melhora = acc_va, ep, 0
            melhor_peso = copy.deepcopy(modelo.state_dict())
        else:
            sem_melhora += 1
            if sem_melhora >= args.paciencia:
                print(f"[{nome}] parada antecipada: {args.paciencia} épocas sem melhora (melhor: {melhor_ep})")
                break
    modelo.load_state_dict(melhor_peso)  # restaura a melhor época
    return modelo, historico, melhor_ep


def treinar_tudo(args, dispositivo, caminhos, rotulos, epocas):
    """Treino final em 100% dos dados, por um número fixo de épocas (sem conjunto de validação)."""
    modelo = criar_modelo(args.modelo, not args.sem_pretreino).to(dispositivo)
    ds = Imagens(caminhos, rotulos, True)
    carregador = DataLoader(ds, batch_size=args.lote, shuffle=True, num_workers=args.trabalhadores,
                            drop_last=len(ds) > args.lote)
    criterio = nn.CrossEntropyLoss()
    otimizador = torch.optim.AdamW(modelo.parameters(), lr=args.lr, weight_decay=args.decaimento)
    agendador = torch.optim.lr_scheduler.CosineAnnealingLR(otimizador, T_max=max(epocas, 1))
    for ep in range(1, epocas + 1):
        modelo.train()
        for x, y in carregador:
            x, y = x.to(dispositivo), y.to(dispositivo)
            otimizador.zero_grad(set_to_none=True)
            criterio(modelo(x), y).backward()
            otimizador.step()
        agendador.step()
        print(f"[final] época {ep}/{epocas}", flush=True)
    return modelo


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dados", required=True, help="pasta com subpastas NILM, LSIL, HSIL, SCC")
    p.add_argument("--saida", default="saidas_citologia")
    p.add_argument("--modelo", default="convnext_large",
                   choices=["convnext_tiny", "convnext_small", "convnext_base", "convnext_large"])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--paciencia", type=int, default=5, help="épocas sem melhora para parar (documentado: 5)")
    p.add_argument("--max-epocas", type=int, default=30, help="[suposição]")
    p.add_argument("--lote", type=int, default=16, help="[suposição]")
    p.add_argument("--lr", type=float, default=1e-4, help="[suposição]")
    p.add_argument("--decaimento", type=float, default=1e-2, help="[suposição]")
    p.add_argument("--semente", type=int, default=42)
    p.add_argument("--trabalhadores", type=int, default=4)
    p.add_argument("--sem-pretreino", action="store_true", help="não baixa os pesos do ImageNet (só para testes)")
    p.add_argument("--treinar-final", action="store_true",
                   help="ao final, treina em todos os dados pela mediana das melhores épocas dos folds")
    args = p.parse_args()

    fixar_sementes(args.semente)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)
    caminhos, rotulos = listar_imagens(args.dados)
    contagem = {c: int((rotulos == i).sum()) for i, c in enumerate(CLASSES)}
    print(f"{len(caminhos)} imagens; classes: {contagem}; dispositivo: {dispositivo}")

    oof_pred = np.full(len(rotulos), -1)
    oof_prob = np.zeros((len(rotulos), len(CLASSES)))
    por_fold, melhores = [], []
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.semente)
    for k, (tr, va) in enumerate(skf.split(caminhos, rotulos), 1):
        modelo, hist, melhor_ep = treinar_fold(args, dispositivo, [caminhos[i] for i in tr], rotulos[tr],
                                               [caminhos[i] for i in va], rotulos[va], f"fold {k}")
        _, acc, pred, real, prob = avaliar(modelo, DataLoader(Imagens([caminhos[i] for i in va], rotulos[va], False),
                                                              batch_size=args.lote, num_workers=args.trabalhadores),
                                           dispositivo, nn.CrossEntropyLoss())
        oof_pred[va], oof_prob[va] = pred, prob
        por_fold.append(dict(fold=k, acuracia=acc, melhor_epoca=melhor_ep, n_val=len(va)))
        melhores.append(melhor_ep)
        torch.save(modelo.state_dict(), saida / f"fold{k}.pt")
        (saida / f"historico_fold{k}.json").write_text(json.dumps(hist, indent=1))
        print(f"[fold {k}] acurácia de validação {acc:.4f} (melhor época {melhor_ep})", flush=True)

    rel = classification_report(rotulos, oof_pred, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(rotulos, oof_pred, labels=range(len(CLASSES)))
    metricas = dict(
        n=int(len(rotulos)), classes=CLASSES, contagem=contagem,
        acuracia=float(np.mean(oof_pred == rotulos)),
        kappa=float(cohen_kappa_score(rotulos, oof_pred)),
        f1_macro=float(rel["macro avg"]["f1-score"]), precisao_macro=float(rel["macro avg"]["precision"]),
        recall_macro=float(rel["macro avg"]["recall"]), por_classe={c: rel[c] for c in CLASSES}, folds=por_fold,
        hiperparametros=dict(modelo=args.modelo, lote=args.lote, lr=args.lr, decaimento=args.decaimento,
                             max_epocas=args.max_epocas, paciencia=args.paciencia, semente=args.semente))
    (saida / "metricas_cv.json").write_text(json.dumps(metricas, ensure_ascii=False, indent=1))
    pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(saida / "matriz_confusao.csv")
    pd.DataFrame({"arquivo": [str(c) for c in caminhos], "real": [CLASSES[i] for i in rotulos],
                  "predito": [CLASSES[i] for i in oof_pred],
                  **{f"p_{c}": oof_prob[:, i] for i, c in enumerate(CLASSES)}}).to_csv(saida / "predicoes_oof.csv", index=False)
    print(f"\nAcurácia CV: {metricas['acuracia']:.4f}  F1 macro: {metricas['f1_macro']:.4f}  kappa: {metricas['kappa']:.4f}")
    print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES))

    if args.treinar_final:
        epocas = int(np.median(melhores))
        final = treinar_tudo(args, dispositivo, caminhos, rotulos, epocas)
        torch.save(final.state_dict(), saida / "citologia_convnext_final.pt")
        print("Checkpoint final:", saida / "citologia_convnext_final.pt")


if __name__ == "__main__":
    main()
