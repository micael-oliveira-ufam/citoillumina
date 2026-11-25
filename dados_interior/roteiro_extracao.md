---
title: "Roteiro de extração de dados por município: câncer do colo do útero no interior do Amazonas"
subtitle: "Projeto CitoIllumina · UFAM · SISCAN, SIM, RHC/FCecon, Painel-Oncologia e RCBP"
---

# 1. Para que serve

A justificativa do projeto é que mulheres do interior dependem de Manaus para diagnóstico e tratamento. Hoje isso se apoia em artigos estaduais e em depoimentos, e nenhuma fonte consultada separa o interior da capital [1–6]. Este roteiro diz onde obter, por município, os números que faltam, com quais filtros, e como compará-los.

**Definição de grupos:** Manaus (código IBGE 1302603; 130260 com 6 dígitos) e interior (os 61 demais municípios do Amazonas). Como análise de sensibilidade, pode-se separar o entorno metropolitano.

**Perguntas e fontes**

| # | Pergunta | Fonte principal | Indicador |
|---|---|---|---|
| P1 | O interior é menos rastreado? | SISCAN (TabNet) + população do IBGE | razão de exames citopatológicos, mulheres de 25 a 64 anos, por município de residência |
| P2 | O resultado do exame demora mais no interior? | SISCAN (TabNet) | distribuição do tempo total do exame (até 30, 31–60, mais de 60 dias) e do intervalo de coleta |
| P3 | A qualidade da coleta é pior? | SISCAN (TabNet) | proporção de exames insatisfatórios (parâmetro INCA: menos de 5%) e razão lesão de alto grau/carcinoma (parâmetro: maior que 10) |
| P4 | Morre-se mais no interior? | SIM (TabNet ou microdados) + população | taxa de mortalidade por C53, por município de residência |
| P5 | Quantas pacientes do interior morrem ou se tratam em Manaus? | SIM (microdados), SIH e SIA | cruzamento município de residência × município de ocorrência ou do estabelecimento |
| P6 | Em que estágio chegam e quanto esperam para tratar? | RHC da FCecon; Painel-Oncologia | estadiamento no primeiro atendimento; tempo do diagnóstico ao tratamento (60 dias) |
| P7 | Qual a incidência no interior? | não encontrei base populacional no interior | só estimativas; o RCBP conhecido é o de Manaus |

# 2. Antes de começar

- **Período:** 2019 a 2024 (ou o mais recente disponível em cada base). Faça também uma análise sem 2020 e 2021, anos de pandemia, que reduziram os exames.
- **População-alvo do rastreio:** mulheres de 25 a 64 anos. Para mortalidade, todas as idades e, em separado, 25 a 64.
- **Código do município:** sempre o de **residência**, com 6 dígitos (o SIM e o SISCAN usam 6; o IBGE usa 7; o script aceita ambos).
- **Ética:** dados agregados e de acesso público, sem identificação individual, em geral dispensam avaliação de comitê de ética pela Resolução CNS 510/2016 [7]. Confirme com o orientador e o CEP da UFAM, sobretudo se usar microdados ou dados nominais da FCecon.
- **Guarde tudo:** salve cada tabela exportada (CSV) com data da consulta, filtros e URL em uma planilha de controle. As bases são atualizadas e os números mudam.

# 3. Passo a passo por fonte

## 3.1 SISCAN: rastreamento (P1, P2, P3)

- **Onde:** TabNet do SISCAN, na página do DATASUS: <https://datasus.saude.gov.br/acesso-a-informacao/sistema-de-informacao-do-cancer-siscan-colo-do-utero-e-mama/> [8].
- **Como o INCA usa:** para incluir exames de um ano faturados no seguinte, selecione os dois anos em "períodos disponíveis" e aplique o filtro "ano de resultado" igual ao ano de interesse [8]. Faça o mesmo.
- **Tabulação sugerida:** Linha = município de residência (ou UF e município); Coluna = ano de resultado; Conteúdo = exames citopatológicos; filtros: sexo feminino, faixa etária 25 a 64, tipo de exame citopatológico, motivo "rastreamento". Exporte em CSV.
- **Tabelas a gerar:**
    1. exames em mulheres de 25 a 64 anos, por município de residência e ano (P1);
    2. exames por adequabilidade da amostra, para obter os insatisfatórios (P3);
    3. exames por tempo total do exame e por intervalo de coleta (P2) [8];
    4. exames com lesão de alto grau e com carcinoma epidermoide invasivo (P3).
- **Confirmar na interface:** o TabNet do SISCAN mostra o município de residência e as variáveis de tempo e adequabilidade acima? O informativo do INCA usa essas medidas [8], mas a lista exata de linhas e colunas depende da versão do TabNet. Se uma variável não existir, vá ao item 3.7 (pedido formal).
- **Cobertura por município:** o SISCAN não dá a população. Use a razão de exames e o denominador do IBGE (3.6). Evite chamar de "cobertura" o que é "razão de exames", porque um exame não equivale a uma mulher examinada.
- **Atenção:** o MPF obteve decisão judicial por falhas no registro de exames de prestadores terceirizados no SISCAN no Amazonas (nov/2025) [9]. Subregistro pode gerar a aparência de baixo rastreio. Cite essa limitação.

## 3.2 SIM: mortalidade (P4)

- **Onde:** TabNet do SIM, "Mortalidade desde 1996 pela CID-10": <https://datasus.saude.gov.br/mortalidade-desde-1996-pela-cid-10> (lista dados até 2023) [10].
- **Tabulação:** Óbitos por **residência**, Amazonas; Linha = município; Coluna = ano do óbito; Conteúdo = óbitos por residência; Categoria CID-10 = C53 (neoplasia maligna do colo do útero); Sexo = feminino. Exporte em CSV. Repita com faixa etária de 25 a 64.
- **Cuidado com a classificação:** parte dos óbitos por câncer de útero é registrada como C55 (útero, porção não especificada) em vez de C53. Para comparar interior e capital, rode também C54 e C55 e verifique se a proporção de C55 difere entre os grupos. Há métodos de correção descritos pelo INCA; consulte o manual de mortalidade do instituto.
- **Municípios pequenos:** com poucos óbitos por município, agregue por grupo (script) e por períodos de 3 a 5 anos. Taxas municipais isoladas são instáveis.
- **Mortalidade mal definida:** registre a proporção de causas mal definidas por município; ela é maior onde o acesso é pior e pode subestimar o interior.

## 3.3 SIM, SIH e SIA: fluxo para Manaus (P5)

Este é o dado mais direto para a premissa do projeto, porque mede quantas pacientes do interior morrem ou se tratam na capital.

- **SIM, microdados:** <https://opendatasus.saude.gov.br/dataset/sim> (CSV, 1979 a 2022) ou arquivos .dbc em <https://datasus.saude.gov.br/transferencia-de-arquivos/> [10]. Filtre sexo feminino, causa C53 e UF de residência igual a 13. Cruze o município de residência com o município de ocorrência do óbito. Variável de interesse: proporção de residentes no interior com óbito ocorrido em Manaus.
- **SIH/SUS e SIA/SUS:** os arquivos de internação e de produção ambulatorial trazem o município de residência da paciente e o do estabelecimento. Filtre procedimentos oncológicos e CID C53 e cruze os dois municípios. O volume exige ler os arquivos .dbc com R (pacote `microdatasus`) ou Python (`PySUS`); confirme os nomes das variáveis no dicionário de dados de cada sistema antes de filtrar.
- **Interpretação:** o fluxo para Manaus mede a **utilização** de serviços da capital, não a necessidade. Descreva assim.

## 3.4 RHC da FCecon: estágio e procedência (P6)

- **Onde:** tabulador público do Integrador RHC: <https://irhc.inca.gov.br/RHCNet/visualizaTabNetExterno.action> [11]. A base é consolidada pelo ano da primeira consulta no hospital [11].
- **Filtros:** hospital = Fundação Centro de Controle de Oncologia do Amazonas (FCecon); localização primária = colo do útero (C53); ano do primeiro atendimento; variável de estadiamento; município de residência.
- **Confirmar na interface:** se o tabulador público permite cruzar município de residência com estadiamento. Se não, solicite à FCecon a tabulação (item 3.7). O RHC é obrigatório para hospitais habilitados em oncologia no SUS [11].
- **Indicador:** proporção de casos em estádio III e IV no primeiro atendimento, interior × Manaus; proporção sem informação de estadiamento (informe sempre).

## 3.5 Painel-Oncologia: tempo até o tratamento (P6)

- **O que é:** painel do Ministério da Saúde que usa datas de diagnóstico histopatológico e de tratamento nos sistemas SIA/SUS, SISCAN e SIH/SUS, e agrupa o tempo em 0 a 30, 31 a 60 e mais de 60 dias, além de "sem informação de tratamento" [12]. Veja o endereço atual do painel no site do INCA ou do Ministério da Saúde; o relatório consultado não traz o link [12].
- **Confirmar na interface:** se o painel permite filtrar por município de residência ou por região de saúde do Amazonas.
- **Limitações [12]:** antes de maio de 2018 o CID e o cartão SUS não eram obrigatórios no exame histopatológico, então os períodos não são comparáveis; a fração "sem informação de tratamento" cresce nos anos recentes.

## 3.6 Denominadores: população (IBGE)

- Mulheres de 25 a 64 anos e total de mulheres, por município de residência, do Censo 2022 e das estimativas e projeções do IBGE (SIDRA) ou da tabela de população residente do DATASUS. Use a mesma fonte para todos os anos e municípios, e registre qual.
- Para a razão de exames: exames em 3 anos ÷ (população feminina de 25 a 64 anos ÷ 3), conforme a ficha técnica do indicador do INCA [13]. Confira o parâmetro vigente na ficha.

## 3.7 Pedido formal quando o painel público não bastar

Se uma variável não existir nos painéis (por exemplo, estadiamento por município na FCecon, tempo de resultado por município no SISCAN-AM), faça um pedido de acesso à informação (Lei 12.527/2011) pela plataforma Fala.BR ou pelo e-SIC estadual, dirigido à FVS-RCP, à SES-AM e à FCecon.

> **Modelo.** Solicito, nos termos da Lei nº 12.527/2011, dados agregados (sem identificação individual) sobre câncer do colo do útero (CID-10 C53 e C55) no Estado do Amazonas, por município de residência, de 2019 a 2024, para pesquisa acadêmica da Universidade Federal do Amazonas: (1) número de exames citopatológicos, por faixa etária, adequabilidade da amostra e data de coleta e de liberação do resultado; (2) casos atendidos na FCecon, por município de residência, ano do primeiro atendimento, estadiamento e data de início do tratamento; (3) óbitos por município de residência e de ocorrência. Peço os dados em planilha (CSV ou XLSX) e a indicação do sistema de origem e da data da extração. Declaro que serão usados somente em pesquisa, sem identificar pacientes.

Revise o texto com o orientador e envie em nome do responsável pela pesquisa.

# 4. Padronizar e comparar

1. Salve cada tabela com as colunas `codmun, ano, casos` (e `codmun, ano, pop` para a população).
2. Rode o script `agregar_interior_capital.py` (pasta `dados_interior/`):

```
python agregar_interior_capital.py taxa --dados obitos_c53.csv --populacao pop_fem.csv --por 100000
python agregar_interior_capital.py proporcao --dados insatisfatorios.csv
```

3. O script soma os municípios em Manaus e interior, calcula taxa ou proporção com IC de 95% (Poisson exato; Wilson) e a razão interior/Manaus. Os arquivos `modelos/exemplo_FICTICIO_*.csv` usam **números inventados** só para mostrar o formato.
4. As taxas são brutas. O interior tem estrutura etária diferente; se possível, padronize por idade (população-padrão mundial ou brasileira) e relate a diferença.

**Tabela final a preencher**

| Indicador | Manaus | Interior | Razão (IC 95%) | Fonte e período |
|---|---|---|---|---|
| Razão de exames (25–64 anos) | | | | SISCAN, IBGE |
| Exames com resultado em mais de 60 dias (%) | | | | SISCAN |
| Exames insatisfatórios (%) | | | | SISCAN |
| Mortalidade por C53 (por 100 mil) | | | | SIM, IBGE |
| Óbitos de residentes no interior ocorridos em Manaus (%) | | | | SIM |
| Casos em estádio III/IV na FCecon (%) | | | | RHC |
| Início do tratamento em mais de 60 dias (%) | | | | Painel-Oncologia |

# 5. Como escrever o resultado

- Diga o que cada medida mede. Razão de exames não é cobertura; óbitos em Manaus de residentes do interior medem uso de serviços; estágio avançado depende de quem chega à FCecon.
- O RCBP conhecido no estado é o de Manaus (confirme no tabulador de incidência do INCA [11] se há outro); **não encontrei incidência medida para o interior**. Use apenas as estimativas do INCA, que são do estado inteiro [1].
- Informe subregistro e dados faltantes (SISCAN, estadiamento, causas mal definidas).
- Os depoimentos usados antes (como a cobertura de cerca de 30% no interior [5]) só devem ser mantidos se forem confirmados pelos números do SISCAN; senão, retire ou marque como relato.

# Referências

1. Instituto Nacional de Câncer (INCA). *Dados e números sobre câncer do colo do útero*. Março de 2023. <https://antigo.inca.gov.br/sites/ufu.sti.inca.local/files/media/document/dados_e_numeros_colo_22marco2023.pdf>
2. Sousa GA, et al. Linha de cuidado do câncer do colo do útero no Amazonas. *Rev Bras Cancerol*. 2021;67(3):e-161282. <https://rbc.inca.gov.br/index.php/revista/article/download/1282/1404>
3. Viana JN, et al. Determinantes sociais da saúde e prevenção secundária do câncer do colo do útero no Estado do Amazonas. *Medicina (Ribeirão Preto)*. 2019;52(2):110-120. <https://core.ac.uk/download/268328301.pdf>
4. Bento MNS, Lima CR, Cordeiro SC. Epidemiologia dos óbitos por neoplasias do colo de útero no Amazonas. *Braz J Implantol Health Sci*. 2025;7(4):851-865.
5. Vencer o Câncer. Câncer de colo de útero provoca 23 mortes por mês no Amazonas (entrevista). <https://vencerocancer.org.br/cancer-de-colo-de-utero-provoca-23-mortes-por-mes-no-amazonas-e-chama-atencao-de-especialistas/>
6. Vieira S. No Amazonas, barreiras à prevenção e ao tratamento de câncer. *Outras Palavras*, 22 out 2025. <https://outraspalavras.net/outrasaude/no-amazonas-barreiras-a-prevencao-e-ao-tratamento-de-cancer/>
7. Conselho Nacional de Saúde. Resolução nº 510, de 7 de abril de 2016.
8. INCA. Informativo, n. 1, 2023 (indicadores do SISCAN e uso do TabNet). <https://ninho.inca.gov.br/jspui/bitstream/123456789/14717/1/informativo_numero_1_2023.pdf>
9. Ministério Público Federal, Procuradoria da República no Amazonas. Decisão obriga Estado e FCecon a garantirem o funcionamento do sistema de informação do câncer. 17 nov 2025. <https://www.mpf.mp.br/o-mpf/unidades/pr-am/noticias/mpf-obtem-decisao-que-obriga-estado-e-fcecon-a-garantirem-pleno-funcionamento-do-sistema-de-informacao-do-cancer-no-am>
10. Ministério da Saúde. Mortalidade (SIM): TabNet, microdados e OpenDataSUS. <https://www.gov.br/saude/pt-br/acesso-a-informacao/sic/dados-em-transparencia-ativa/svsa/mortalidade>
11. INCA. Registros Hospitalares de Câncer (tabulador público do RHC, RHCWeb e tabulador de incidência dos RCBP). <https://www.gov.br/inca/pt-br/assuntos/cancer/numeros/registros/rhc>
12. INCA. Relatório do Painel-Oncologia (maio de 2020). <https://www.inca.gov.br/sites/ufu.sti.inca.local/files/media/document/relatorio_painel_oncologia_sus_maio2020.pdf>
13. INCA. Ficha técnica dos indicadores do câncer do colo do útero. <https://www.inca.gov.br/sites/ufu.sti.inca.local/files/media/document/fichatecnicaindicadorescolo14.pdf>
