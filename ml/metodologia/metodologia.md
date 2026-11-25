---
title: "Metodologia de comparação de arquiteturas de redes neurais para citologia cervical em meio líquido"
subtitle: "Projeto CitoIllumina · UFAM"
---

# 1. Objetivo e justificativa

Comparar, sob um protocolo único, arquiteturas de redes neurais convolucionais antigas e atuais e transformers para classificar imagens de citologia em meio líquido em quatro classes do Sistema Bethesda (NILM, LSIL, HSIL, SCC) [1], e permitir a anotação célula a célula das imagens.

O contexto é o do Amazonas. O INCA estimou, para o triênio 2023–2025, 610 casos novos de câncer do colo do útero no Amazonas, taxa bruta de 27,63 e ajustada de 31,71 por 100 mil mulheres, contra 20,48 (bruta) e 16,77 (ajustada) na Região Norte [2]. No mesmo documento, o câncer do colo do útero é a principal causa de morte por câncer entre as mulheres da Região Norte (taxa ajustada de 9,07 por 100 mil em 2021) [2].

## 1.1 O interior do Amazonas

**Rastreamento.** Entre 2016 e 2019, o estado realizou menos exames citopatológicos do que o necessário em todos os anos (por exemplo, 125.018 realizados contra 325.605 a 360.866 esperados, conforme o ano; déficit de 47% a 65%), e a colposcopia ficou 83% a 86% abaixo da necessidade [27]. Em 2010–2014, a cobertura anual média foi de 12,3% a 19,9% e a razão de exames ficou abaixo da meta de 1 [28]. Esses estudos analisam o estado como um todo ou os 62 municípios em conjunto; **nenhuma das fontes que consegui consultar publica a cobertura separada entre interior e capital**. A única cifra desse tipo é o depoimento de uma ginecologista da FCecon a uma reportagem: cobertura de rastreio abaixo de 46% no estado e de cerca de 30% no interior [31]. É uma declaração sem fonte de dados citada, e deve ser tratada como indício, não como estatística.

**Diagnóstico e tratamento concentrados em Manaus.** Segundo a mesma entrevista, as coletas são feitas nos municípios, mas as lâminas são enviadas a Manaus, e alguns municípios esperam oito meses pelo resultado; a colposcopia com biópsia só está disponível em Manaus, e alguns municípios ficam a até sete dias de barco da capital [31]. A FCecon é a única unidade de referência de alta complexidade do SUS para câncer no estado, e mulheres de ao menos 61 municípios precisam ir a Manaus para o tratamento; relatos de pacientes citam viagens de até 8 dias de barco, sem apoio público de transporte, hospedagem ou alimentação [30]. A própria FCecon reconheceu o problema ao propor, em 2019, mutirões semestrais de "Ver e Tratar" em sete municípios-polo (Itacoatiara, Borba, Manacapuru, Tabatinga, Parintins, Tefé e Coari) para evitar o deslocamento a Manaus [32]. No estado, em 28% dos casos oncológicos analisados pelo Ministério Público Federal em 2026, o prazo legal de 60 dias para iniciar o tratamento não foi cumprido [33].

**Mortalidade.** Foram 1.393 óbitos por câncer do colo do útero no Amazonas de 2020 a 2024; 968 notificados em Manaus e 425 nos demais municípios (cerca de 30,5%) [29]. Esse recorte não mede o risco do interior: o artigo não deixa claro se o local é o de residência ou o de ocorrência, e uma paciente do interior que morre em Manaus pode ser contada como da capital. A taxa de incidência estimada para Manaus em 2012–2016 foi de 38,25 por 100 mil, segundo registro de base populacional citado por [34].

**O que falta.** Para sustentar com números a premissa de que o interior depende de Manaus, é preciso extrair, por município de residência, (i) a razão de exames citopatológicos e a proporção de exames insatisfatórios (SISCAN/INCA), (ii) os óbitos (SIM/DATASUS) e (iii) a procedência dos pacientes atendidos na FCecon (registro hospitalar de câncer). Essas bases e os boletins da FVS-RCP não puderam ser consultados nesta pesquisa; os resultados acima vêm de artigos, do relatório do INCA e de reportagens.

A premissa do projeto é que uma triagem apoiada por IA, com leitura de imagens no local, pode reduzir a dependência do deslocamento e do tempo de resposta das lâminas enviadas a Manaus.

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
27. Sousa GA, Viana JN, Souza CSM, Moysés RPC. Linha de cuidado do câncer do colo do útero no Amazonas: uma análise da prevenção ao tratamento de lesões precursoras. *Rev Bras Cancerol*. 2021;67(3):e-161282. <https://rbc.inca.gov.br/index.php/revista/article/download/1282/1404>
28. Viana JN, Moysés RPC, Espir TT, Sousa GA, Barcellos JFM, Alves MGP. Determinantes sociais da saúde e prevenção secundária do câncer do colo do útero no Estado do Amazonas, Brasil. *Medicina (Ribeirão Preto)*. 2019;52(2):110-120. <https://core.ac.uk/download/268328301.pdf>
29. Bento MNS, Lima CR, Cordeiro SC. Saúde da mulher: epidemiologia dos óbitos por neoplasias do colo de útero no Amazonas. *Braz J Implantol Health Sci*. 2025;7(4):851-865. doi:10.36557/2674-8169.2025v7n4p851-865
30. Vieira S. No Amazonas, barreiras à prevenção e ao tratamento de câncer. *Outras Palavras (Outra Saúde)*. 22 out 2025. <https://outraspalavras.net/outrasaude/no-amazonas-barreiras-a-prevencao-e-ao-tratamento-de-cancer/>
31. Vencer o Câncer. Câncer de colo de útero provoca 23 mortes por mês no Amazonas e chama atenção de especialistas (entrevista com a ginecologista Mônica Bandeira de Melo, FCecon). Publicado em 31 jan 2020, atualizado em 1 nov 2025. <https://vencerocancer.org.br/cancer-de-colo-de-utero-provoca-23-mortes-por-mes-no-amazonas-e-chama-atencao-de-especialistas/>
32. Secretaria de Estado de Saúde do Amazonas (SES-AM). Na FCecon, mais de 500 mulheres foram submetidas a procedimento que evita o câncer de colo do útero. 30 dez 2019. <https://www.saude.am.gov.br/na-fcecon-mais-de-500-mulheres-foram-submetidas-a-procedimento-que-evita-o-cancer-de-colo-do-utero/>
33. Ministério Público Federal, Procuradoria da República no Amazonas. MPF recomenda medidas urgentes na atenção a pacientes oncológicos no Amazonas. 23 abr 2026. <https://www.mpf.mp.br/o-mpf/unidades/pr-am/noticias/mpf-recomenda-medidas-urgentes-na-atencao-a-pacientes-oncologicos-no-amazonas>
34. Souza LR, Borges MFSO, Silva IF. Delays in cervical cancer treatment initiation and associated factors in a hospital-based cohort in the Brazilian Western Amazon. *Rev Bras Cancerol*. 2025;71(4):e-105228. (Estudo em Rio Branco, AC; a cifra de Manaus é citada do registro de base populacional.) <https://rbc.inca.gov.br/index.php/revista/article/download/5228/4083/47778>
