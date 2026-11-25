# Comparação de arquiteturas e anotação por célula

Scripts: `comparar_arquiteturas.py` (treino, validação, gráficos, relatório) e `anotar_celulas.py` (caixas por célula).

## 1. Comparação com o notebook original (Kaggle)

| Item | Notebook | `comparar_arquiteturas.py` |
|---|---|---|
| Arquiteturas | ResNet50 (divisão única), ConvNeXt-Large e Swin V2-B (5 folds) | 12 arquiteturas, de AlexNet (2012) a ConvNeXt, ViT, Swin V2 e MaxViT, todas com 5 folds |
| Validação | StratifiedKFold(5, semente 42); acurácia por fold, F1 ponderado, relatório de classificação, matriz global | Mesma divisão, mais F1 macro, acurácia balanceada, kappa, MCC, AUC, perda logarítmica, IC 95% por bootstrap, McNemar entre arquiteturas |
| Otimização | Adam 1e-4, decaimento 1e-5, lote 16, 50 épocas, paciência 5 na acurácia | Idêntico |
| Aumentos | Resize 224, giros, RandomRotate90, ColorJitter | Idêntico (ColorJitter 0,2/0,2/0,2/0,1, p=0,3) |
| Cabeça | Dropout(0,5) + Linear(4) | Idêntica em todas as arquiteturas |
| Curvas | Acurácia e perda por fold e época | Por fold e por arquitetura, mais curva comparativa de todas |
| Custo | não mede | parâmetros, tamanho, latência, tempo por época |
| Saídas | gráficos no notebook | JSON por fold, CSV, PNG, `relatorio.md` e `relatorio.docx` |

Diferenças a ter em mente: o notebook usa a perda de validação na parada antecipada da célula ResNet50 e a acurácia nas de k-fold (aqui sempre a acurácia); a ordem de classes do script é SCC, HSIL, LSIL, NILM, a mesma de `app/tools/export_onnx.py` (o notebook usa Carcinoma, HSIL, LSIL, Normal, que é a mesma ordem com outros nomes). O script aceita os dois nomes de pasta.

## 2. Uso

```bash
pip install torch torchvision albumentations scikit-learn scipy matplotlib python-docx scikit-image pillow
python comparar_arquiteturas.py --dados "Liquid Citology + IARC dataset" --saida resultados_arquiteturas --salvar-pesos
python comparar_arquiteturas.py --dados dados --arquiteturas resnet50 convnext_large swin_v2_b
python comparar_arquiteturas.py --dados dados --so-relatorio      # refaz gráficos e relatório
```

Retomável: cada par (arquitetura, fold) concluído é reaproveitado. `--grupos arquivo.csv` (colunas `arquivo,grupo`) divide por paciente. `--lr-por-arquitetura '{"vit_b_16": 3e-5}'` ajusta a taxa por rede.

## 3. Saídas

`relatorio.md/.docx` (protocolo, tabelas, gráficos, limitações), `resultados.csv`, `config.json`, `comparativo_*.png`, e por arquitetura: `fold*.json` (histórico por época, probabilidades), `curvas.png` (acurácia e perda por época), `matriz_confusao.png`, `fold*.pt` (com `--salvar-pesos`).

## 4. Anotação por célula

```bash
python anotar_celulas.py --imagens pasta --arquitetura convnext_large --pesos resultados_arquiteturas/convnext_large/fold1.pt --saida anotadas
```

Localiza núcleos (deconvolução hematoxilina, Otsu, componentes conectados) ou usa grade (`--modo grade`), classifica cada recorte e desenha caixa colorida com "CLASSE xx%" e faixa de título com a classe da lâmina (a mais grave detectada), confiança e número de detecções. Gera `deteccoes.csv` e `resumo.csv`.

**Limitação:** as redes foram treinadas com campos inteiros, não com células individuais. A classe por célula é extrapolação, sem validação contra rótulos celulares; a localização por núcleos é heurística e pode juntar ou perder células. Os testes feitos foram com dados sintéticos; nenhum resultado com as imagens reais foi obtido nesta sessão.

## 5. Rotulagem de recortes por especialista

```bash
python treinar_recortes.py extrair --dados <pasta> --saida recortes_out
python treinar_recortes.py planilha --dados <pasta> --saida recortes_out --por-classe 100
# entregar ao especialista APENAS planilha_rotulagem.xlsx (a chave_rotulagem.csv fica com você)
python treinar_recortes.py planilha-para-csv --saida recortes_out --xlsx planilha_preenchida.xlsx
python treinar_recortes.py treinar --saida recortes_out --rotulos recortes_out/rotulos.csv --so-rotulados
```

A planilha é cega (não mostra a classe da imagem de origem), tem ordem aleatória, no máximo 3 recortes por imagem, recorte e contexto lado a lado (caixa vermelha), listas suspensas de rótulo e de certeza. Para medir a confiabilidade do rótulo, duas pessoas devem preencher cópias separadas; o kappa entre elas é calculado fora do script.
