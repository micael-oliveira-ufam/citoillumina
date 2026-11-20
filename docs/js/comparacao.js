// Estatísticas de comparação entre duas classificações (IA × IA ou IA × humano).
// Funções puras, sem DOM: recebem pares { a, b } de siglas Bethesda e devolvem números.
// `a` é a classificação comparada e `b` a de referência (nas métricas por classe e binárias).

export const ORDEM = ['NILM', 'ASC-US', 'LSIL', 'ASC-H', 'HSIL', 'SCC', 'Adenocarcinoma', 'Outro'];
export const ALTO_GRAU = new Set(['ASC-H', 'HSIL', 'SCC']);

/** Agrupa ASC-US em LSIL e ASC-H em HSIL, para alinhar com as 4 classes do modelo. */
export function agrupar(sigla) {
  if (sigla === 'ASC-US') return 'LSIL';
  if (sigla === 'ASC-H') return 'HSIL';
  return sigla;
}

const div = (x, y) => (y > 0 ? x / y : null);

function binarias(pares, positivo) {
  let tp = 0; let fp = 0; let fn = 0; let tn = 0;
  for (const { a, b } of pares) {
    const pa = positivo(a); const pb = positivo(b);
    if (pa && pb) tp++; else if (pa && !pb) fp++; else if (!pa && pb) fn++; else tn++;
  }
  return {
    tp, fp, fn, tn,
    sensibilidade: div(tp, tp + fn),
    especificidade: div(tn, tn + fp),
    vpp: div(tp, tp + fp),
    vpn: div(tn, tn + fn),
    acuracia: div(tp + tn, pares.length),
  };
}

/** Totais por classe de uma lista de siglas (ignora vazios). */
export function totaisPorClasse(siglas, agrupa = false) {
  const t = new Map();
  for (const s0 of siglas) {
    if (!s0) continue;
    const s = agrupa ? agrupar(s0) : s0;
    t.set(s, (t.get(s) || 0) + 1);
  }
  return t;
}

export function comparar(pares0, { agrupa = false } = {}) {
  const pares = pares0.filter((p) => p.a && p.b).map((p) => (agrupa ? { a: agrupar(p.a), b: agrupar(p.b) } : p));
  const n = pares.length;
  const presentes = new Set();
  for (const { a, b } of pares) { presentes.add(a); presentes.add(b); }
  const classes = [...presentes].sort((x, y) => (ORDEM.indexOf(x) + 1 || 99) - (ORDEM.indexOf(y) + 1 || 99));
  const matriz = {};
  for (const c of classes) { matriz[c] = {}; for (const d of classes) matriz[c][d] = 0; }
  for (const { a, b } of pares) matriz[a][b]++;
  const exatos = pares.filter((p) => p.a === p.b).length;
  const po = div(exatos, n);
  let pe = null;
  if (n) {
    pe = 0;
    for (const c of classes) {
      const linha = classes.reduce((s, d) => s + matriz[c][d], 0);
      const col = classes.reduce((s, d) => s + matriz[d][c], 0);
      pe += (linha / n) * (col / n);
    }
  }
  const kappa = n && pe !== null && pe < 1 ? (po - pe) / (1 - pe) : null;
  const porClasse = classes.map((c) => {
    const tp = matriz[c][c];
    const predito = classes.reduce((s, d) => s + matriz[c][d], 0); // a = c
    const real = classes.reduce((s, d) => s + matriz[d][c], 0); // b = c
    const precisao = div(tp, predito);
    const revocacao = div(tp, real);
    const f1 = precisao !== null && revocacao !== null && precisao + revocacao > 0
      ? (2 * precisao * revocacao) / (precisao + revocacao) : null;
    return { classe: c, predito, real, tp, precisao, revocacao, f1 };
  });
  return {
    n, exatos, concordancia: po, kappa, classes, matriz, porClasse,
    lesao: binarias(pares, (s) => s !== 'NILM'),
    altoGrau: binarias(pares, (s) => ALTO_GRAU.has(s)),
  };
}

/** Interpretação usual do kappa (Landis e Koch). */
export function leituraKappa(k) {
  if (k === null || k === undefined || Number.isNaN(k)) return 'indefinido';
  if (k < 0) return 'pior que o acaso';
  if (k <= 0.2) return 'leve';
  if (k <= 0.4) return 'razoável';
  if (k <= 0.6) return 'moderada';
  if (k <= 0.8) return 'substancial';
  return 'quase perfeita';
}
