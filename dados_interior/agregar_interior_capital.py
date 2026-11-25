#!/usr/bin/env python3
"""Agrega dados por município do Amazonas em dois grupos, Manaus e interior (61 demais municípios), e compara.

Dois modos:
  taxa        casos (óbitos, exames, casos novos) por população -> taxa por 100 mil, IC 95% exato de Poisson,
              razão de taxas interior/Manaus com IC 95% (método do log).
  proporcao   numerador e denominador por município (ex.: exames insatisfatórios / exames) -> proporção,
              IC 95% de Wilson, razão de proporções interior/Manaus.

Entrada (CSV com cabeçalho; separador vírgula ou ponto e vírgula; códigos IBGE de 6 ou 7 dígitos):
  taxa:       --dados codmun,ano,casos   --populacao codmun,ano,pop
  proporcao:  --dados codmun,ano,num,den
Manaus = 1302603 (6 dígitos: 130260). Municípios fora do Amazonas (código que não começa com 13) são ignorados.
Os anos usados são os presentes em --dados; a população é somada nos mesmos anos (pessoa-ano).

Exemplo:
    python agregar_interior_capital.py taxa --dados obitos_c53.csv --populacao pop_fem.csv --por 100000
    python agregar_interior_capital.py proporcao --dados insatisfatorios.csv
Aviso: taxas brutas não são padronizadas por idade; a estrutura etária do interior difere da de Manaus.
"""
import argparse
import csv
import math
import sys
from collections import defaultdict

MANAUS = "130260"


def ler(caminho):
    txt = open(caminho, encoding="utf-8-sig").read()
    sep = ";" if txt.split("\n", 1)[0].count(";") > txt.split("\n", 1)[0].count(",") else ","
    return list(csv.DictReader(txt.splitlines(), delimiter=sep))


def cod6(x):
    x = str(x).strip().split(".")[0]
    return x[:6]


def grupo(cod):
    if not cod.startswith("13"):
        return None
    return "Manaus" if cod == MANAUS else "Interior"


def num(x):
    x = str(x).strip().replace(" ", "")
    if x in ("", "-", "..."):
        return 0.0
    return float(x.replace(".", "").replace(",", ".")) if ("," in x) else float(x)


def poisson_ic(k, z=1.959964):
    from scipy.stats import chi2
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2
    return lo, hi


def wilson(k, n, z=1.959964):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def razao(k1, n1, k0, n0, z=1.959964):
    """Razão (k1/n1)/(k0/n0) com IC do log; exige k1,k0 > 0."""
    if min(k1, k0, n1, n0) <= 0:
        return float("nan"), float("nan"), float("nan")
    r = (k1 / n1) / (k0 / n0)
    ep = math.sqrt(1 / k1 + 1 / k0)
    return r, r * math.exp(-z * ep), r * math.exp(z * ep)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modo", choices=["taxa", "proporcao"])
    p.add_argument("--dados", required=True)
    p.add_argument("--populacao")
    p.add_argument("--por", type=float, default=100000)
    a = p.parse_args()
    dados = ler(a.dados)
    ign = 0
    if a.modo == "taxa":
        if not a.populacao:
            sys.exit("--populacao é obrigatório no modo taxa")
        anos = {r["ano"] for r in dados}
        k = defaultdict(float)
        n = defaultdict(float)
        for r in dados:
            g = grupo(cod6(r["codmun"]))
            if g:
                k[g] += num(r["casos"])
            else:
                ign += 1
        for r in ler(a.populacao):
            g = grupo(cod6(r["codmun"]))
            if g and r["ano"] in anos:
                n[g] += num(r["pop"])
        print(f"{'Grupo':10s} {'Casos':>8s} {'Pessoa-ano':>14s} {'Taxa/' + str(int(a.por)):>12s}  IC95%")
        for g in ("Manaus", "Interior"):
            t = k[g] / n[g] * a.por if n[g] else float("nan")
            lo, hi = poisson_ic(int(k[g]))
            print(f"{g:10s} {k[g]:8.0f} {n[g]:14.0f} {t:12.2f}  {lo / n[g] * a.por if n[g] else float('nan'):.2f} a {hi / n[g] * a.por if n[g] else float('nan'):.2f}")
        r, lo, hi = razao(k["Interior"], n["Interior"], k["Manaus"], n["Manaus"])
        print(f"Razão de taxas interior/Manaus: {r:.2f} (IC95% {lo:.2f} a {hi:.2f})")
    else:
        k, n = defaultdict(float), defaultdict(float)
        for r in dados:
            g = grupo(cod6(r["codmun"]))
            if g:
                k[g] += num(r["num"])
                n[g] += num(r["den"])
            else:
                ign += 1
        print(f"{'Grupo':10s} {'Num':>8s} {'Den':>10s} {'Prop%':>8s}  IC95% (Wilson)")
        for g in ("Manaus", "Interior"):
            pr = k[g] / n[g] * 100 if n[g] else float("nan")
            lo, hi = wilson(k[g], n[g])
            print(f"{g:10s} {k[g]:8.0f} {n[g]:10.0f} {pr:8.2f}  {lo * 100:.2f} a {hi * 100:.2f}")
        r, lo, hi = razao(k["Interior"], n["Interior"], k["Manaus"], n["Manaus"])
        print(f"Razão de proporções interior/Manaus: {r:.2f} (IC95% {lo:.2f} a {hi:.2f}; aproximação do log)")
    if ign:
        print(f"({ign} linhas fora do Amazonas ignoradas)")


if __name__ == "__main__":
    main()
