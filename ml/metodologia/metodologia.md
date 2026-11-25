---
title: "Metodologia de comparação de arquiteturas de redes neurais para citologia cervical em meio líquido"
subtitle: "Projeto CitoIllumina · UFAM"
---

# 1. Objetivo e justificativa

Comparar, sob um protocolo único, arquiteturas de redes neurais convolucionais antigas e atuais e transformers para classificar imagens de citologia em meio líquido em quatro classes do Sistema Bethesda (NILM, LSIL, HSIL, SCC) [1], e permitir a anotação célula a célula das imagens.

O contexto é o do Amazonas. O INCA estimou, para o triênio 2023–2025, 610 casos novos de câncer do colo do útero no Amazonas, taxa bruta de 27,63 e ajustada de 31,71 por 100 mil mulheres, contra 20,48 (bruta) e 16,77 (ajustada) na Região Norte [2]. No mesmo documento, o câncer do colo do útero é a principal causa de morte por câncer entre as mulheres da Região Norte (taxa ajustada de 9,07 por 100 mil em 2021) [2]. A premissa do projeto é que pacientes do interior do estado precisam se deslocar a Manaus para diagnóstico e tratamento, e que uma triagem apoiada por IA pode reduzir essa dependência. **Os dados consultados [2] não separam o interior da capital**; números por município (por exemplo, do SISCAN e da FVS-RCP) ainda precisam ser levantados e não são apresentados aqui.

# 2. Dados

Imagens de citologia em meio líquido do conjunto público de Hussain et al. (Mendeley Data) [3], combinadas com imagens do Atlas Digital do IARC [4]. O conjunto de trabalho tem 1.096 imagens: 637 NILM, 147 LSIL, 216 HSIL e 96 SCC. Nenhuma imagem de pacientes da rede SEMSA foi usada. As pastas são lidas como Carcinoma (SCC), HSIL, LSIL e Normal (NILM).

# 3. Protocolo de treino e validação

O protocolo reproduz o do notebook original do projeto e vale para todas as arquiteturas, de modo que as diferenças refletem a rede e não o procedimento.

| Item | Valor |
|---|---|
| Pré-treino | pesos do ImageNet [5] (torchvision [6]) |
| Cabeça | nova: Dropout(0,5) + Linear(4) |
| Entrada | 224×224 (Resize, interpolação linear), normalização do ImageNet |
| Aumentos (treino) | giro horizontal e vertical (p=0,5), rotação de 90° (p=0,5), ColorJitter 0,2/0,2/0,2/0,1 (p=0,3), via Albumentations [7] |
| Otimizador | Adam [8], taxa 1e-4, decaimento de pesos 1e-5 |
| Lote / épocas | 16 / até 50 |
| Parada antecipada | paciência 5 sobre a acurácia de validação; restaura a melhor época |
| Validação | validação cruzada estratificada em 5 folds, com embaralhamento e semente 42 [9]; agrupada por paciente (StratifiedGroupKFold) quando o identificador existe |

Cada imagem é predita uma única vez, quando está no fold de validação, o que gera predições fora da amostra para todo o conjunto.

# 4. Arquiteturas

Parâmetros conforme torchvision, com a cabeça original de 1.000 classes.

| Arquitetura | Ano | Família | Parâmetros (M) | Ref. |
|---|---|---|---|---|
| AlexNet | 2012 | CNN clássica | 61,1 | [10] |
| VGG-16 (com BatchNorm) | 2014 | CNN clássica | 138,4 | [11] |
| GoogLeNet (Inception v1) | 2014 | CNN com módulos Inception | 6,6 | [12] |
| ResNet-50 | 2015 | CNN residual | 25,6 | [13] |
| DenseNet-121 | 2016 | CNN com conexões densas | 8,0 | [14] |
| MobileNetV3-Large | 2019 | CNN leve | 5,5 | [15] |
| ViT-B/16 | 2020 | Transformer | 86,6 | [16] |
| EfficientNetV2-S | 2021 | CNN eficiente | 21,5 | [17] |
| ConvNeXt-Tiny | 2022 | CNN modernizada | 28,6 | [18] |
| ConvNeXt-Large | 2022 | CNN modernizada | 197,8 | [18] |
| Swin Transformer V2-B | 2022 | Transformer hierárquico | 87,9 | [19, 20] |
| MaxViT-T | 2022 | Híbrido convolução e atenção | 30,9 | [21] |

# 5. Métricas de validação

- **Acurácia**, média e desvio entre os folds, e acurácia global das predições fora da amostra. Intervalo de confiança de 95% por bootstrap (2.000 reamostragens das predições).
- **Matriz de confusão** global, em contagens e normalizada por linha (revocação).
- **Precisão, revocação (sensibilidade) e F1 por classe**, e F1 macro. A revocação de HSIL e SCC é a medida decisiva em triagem, porque mede os casos graves que escapariam.
- **Acurácia balanceada**, média das revocações, por causa do desbalanceamento (NILM é 58% das imagens).
- **Kappa de Cohen** [22], interpretado pela escala de Landis e Koch [23], e **coeficiente de correlação de Matthews** [24].
- **AUC-ROC** um-contra-todos, média das classes, e **perda logarítmica**, que avalia a calibração das probabilidades.
- **Curvas de acurácia e de perda por época**, de treino e validação, por fold e por arquitetura, com a melhor época marcada.
- **Teste de McNemar exato** [25] entre a melhor arquitetura e cada uma das demais, sobre as mesmas predições. Não há correção para comparações múltiplas.
- **Custo**: parâmetros, tamanho, latência de inferência (lote 1) e tempo por época.

# 6. Classificação célula a célula

O script `anotar_celulas.py` localiza candidatas a células pelos núcleos (deconvolução de cor para hematoxilina, limiar de Otsu e componentes conectados) [26], recorta cada candidata, classifica o recorte e desenha uma caixa colorida com a classe e a confiança. A classe da lâmina é a mais grave detectada (NILM < LSIL < HSIL < SCC).

As redes da seção 4 foram treinadas com campos inteiros. Aplicá-las a recortes é uma extrapolação, sem validação contra rótulos celulares. Para tornar a classificação celular validável, o script `treinar_recortes.py` extrai os recortes, aceita rótulos manuais de especialista (`rotulos.csv`) e valida com divisão agrupada por imagem de origem, evitando que recortes da mesma imagem fiquem no treino e na validação. Sem rótulos manuais, cada recorte herda a classe da imagem inteira (rótulo fraco): o resultado então mede a reprodução do rótulo da imagem, não o desempenho celular. A recomendação é rotular ao menos uma amostra estratificada (por exemplo, 100 recortes por classe) e reportar as métricas apenas sobre ela.

# 7. Limitações

- Validação interna: não há validação externa com lâminas de outro laboratório.
- A divisão por imagem pode colocar imagens do mesmo paciente no treino e na validação e superestimar o desempenho; use o agrupamento por paciente quando possível.
- Uma semente por arquitetura: diferenças menores que cerca de 1 ponto percentual estão dentro do ruído.
- Os mesmos hiperparâmetros para todas as redes favorecem as que se adaptam bem a Adam com taxa 1e-4; os transformers podem melhorar com outros valores.
- A acurácia da parada antecipada é a mesma reportada, o que deixa a estimativa levemente otimista; para uma estimativa limpa, reserve um conjunto de teste à parte.
- As imagens são campos isolados, não lâminas inteiras; o resultado não equivale a desempenho diagnóstico.
- Até a data deste documento os scripts foram testados apenas com dados sintéticos; o treino com as imagens reais deve ser feito em GPU (notebook Kaggle fornecido).

# 8. Reprodutibilidade

```
python comparar_arquiteturas.py --dados <pasta> --saida resultados_arquiteturas --salvar-pesos
python anotar_celulas.py --imagens <pasta> --arquitetura <rede> --pesos <fold1.pt> --saida anotadas
python treinar_recortes.py extrair --dados <pasta> --saida recortes_out
python treinar_recortes.py treinar --saida recortes_out --rotulos rotulos.csv --so-rotulados
```

Código em https://github.com/micael-oliveira-ufam/citoillumina (pasta `ml/`). O ambiente (versões de Python, torch e torchvision) é gravado em `config.json`.

# Referências

1. Nayar R, Wilbur DC, eds. *The Bethesda System for Reporting Cervical Cytology*. 3rd ed. Springer; 2015.
2. Instituto Nacional de Câncer (INCA). *Dados e números sobre câncer do colo do útero*. Relatório anual, março de 2023 (estimativas 2023–2025, Tabela 1; mortalidade, p. 7). <https://antigo.inca.gov.br/sites/ufu.sti.inca.local/files/media/document/dados_e_numeros_colo_22marco2023.pdf>
3. Hussain E, Mahanta LB, Das CR, Talukdar RK. A comprehensive study on the multi-class cervical cancer diagnostic prediction on pap smear images using a fusion-based decision from ensemble deep convolutional neural network. *Tissue and Cell*. 2020;65:101347. [Conjunto de dados: Liquid based cytology Pap smear images for multi-class diagnosis of cervical cancer, Mendeley Data.]
4. International Agency for Research on Cancer (IARC). *Digital Atlas of Cervical Cytology* (IARC, Lyon).
5. Russakovsky O, et al. ImageNet large scale visual recognition challenge. *Int J Comput Vis*. 2015;115:211-252.
6. Paszke A, et al. PyTorch: an imperative style, high-performance deep learning library. *NeurIPS*. 2019.
7. Buslaev A, et al. Albumentations: fast and flexible image augmentations. *Information*. 2020;11(2):125.
8. Kingma DP, Ba J. Adam: a method for stochastic optimization. *ICLR*. 2015. arXiv:1412.6980.
9. Pedregosa F, et al. Scikit-learn: machine learning in Python. *J Mach Learn Res*. 2011;12:2825-2830.
10. Krizhevsky A, Sutskever I, Hinton GE. ImageNet classification with deep convolutional neural networks. *NeurIPS*. 2012.
11. Simonyan K, Zisserman A. Very deep convolutional networks for large-scale image recognition. *ICLR*. 2015. arXiv:1409.1556.
12. Szegedy C, et al. Going deeper with convolutions. *CVPR*. 2015. arXiv:1409.4842.
13. He K, Zhang X, Ren S, Sun J. Deep residual learning for image recognition. *CVPR*. 2016. arXiv:1512.03385.
14. Huang G, Liu Z, van der Maaten L, Weinberger KQ. Densely connected convolutional networks. *CVPR*. 2017. arXiv:1608.06993.
15. Howard A, et al. Searching for MobileNetV3. *ICCV*. 2019. arXiv:1905.02244.
16. Dosovitskiy A, et al. An image is worth 16x16 words: transformers for image recognition at scale. *ICLR*. 2021. arXiv:2010.11929.
17. Tan M, Le QV. EfficientNetV2: smaller models and faster training. *ICML*. 2021. arXiv:2104.00298.
18. Liu Z, Mao H, Wu C-Y, Feichtenhofer C, Darrell T, Xie S. A ConvNet for the 2020s. *CVPR*. 2022. arXiv:2201.03545.
19. Liu Z, et al. Swin Transformer: hierarchical vision transformer using shifted windows. *ICCV*. 2021. arXiv:2103.14030.
20. Liu Z, et al. Swin Transformer V2: scaling up capacity and resolution. *CVPR*. 2022. arXiv:2111.09883.
21. Tu Z, et al. MaxViT: multi-axis vision transformer. *ECCV*. 2022. arXiv:2204.01697.
22. Cohen J. A coefficient of agreement for nominal scales. *Educ Psychol Meas*. 1960;20(1):37-46.
23. Landis JR, Koch GG. The measurement of observer agreement for categorical data. *Biometrics*. 1977;33(1):159-174.
24. Matthews BW. Comparison of the predicted and observed secondary structure of T4 phage lysozyme. *Biochim Biophys Acta*. 1975;405(2):442-451.
25. McNemar Q. Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*. 1947;12(2):153-157.
26. Ruifrok AC, Johnston DA. Quantification of histochemical staining by color deconvolution. *Anal Quant Cytol Histol*. 2001;23(4):291-299.
