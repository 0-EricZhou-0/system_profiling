// decimate.js — tools/decimate.py for the Bokeh page: window = window,
// keep = keep (same rules, same indices; tests/python/test_decimate.py
// runs both on the same inputs). The page's CustomJS inlines this file and
// redraws every long series from its full data on x-range changes: per
// pixel column, its first, last, lowest and highest finite sample and its
// first and last NaN (a gap); all samples when the window is sparse; and
// always the series' own first, last, lowest and highest samples
// (decimExtremes = extremes), so the drawn data spans the full data.

const EXACT_PER_COLUMN = 4;

function decimWindow(start, end, cols) {
  const w = end - start;
  return [start - w, end + w, w / cols];
}

function lowerBound(x, v) {            // first i with x[i] >= v
  let a = 0, b = x.length;
  while (a < b) { const m = (a + b) >> 1; if (x[m] < v) a = m + 1; else b = m; }
  return a;
}

function upperBound(x, v) {            // first i with x[i] > v
  let a = 0, b = x.length;
  while (a < b) { const m = (a + b) >> 1; if (x[m] <= v) a = m + 1; else b = m; }
  return a;
}

function decimKeep(x, y, lo, hi, bw) {
  const n = x.length;
  const i0 = Math.max(lowerBound(x, lo) - 1, 0);
  const i1 = Math.min(upperBound(x, hi) + 1, n);
  const out = [];
  if (i1 <= i0) return out;
  if (i1 - i0 <= EXACT_PER_COLUMN * (hi - lo) / bw) {
    for (let i = i0; i < i1; i++) out.push(i);
    return out;
  }
  let i = i0;
  while (i < i1) {
    const c = Math.floor(x[i] / bw);
    let j = i, mn = -1, mx = -1, n0 = -1, n1 = -1;
    for (; j < i1 && Math.floor(x[j] / bw) === c; j++) {
      const v = y[j];
      if (Number.isFinite(v)) {
        if (mn < 0 || v < y[mn]) mn = j;
        if (mx < 0 || v > y[mx]) mx = j;
      } else {                         // NaN (or null, or ±inf): a gap
        if (n0 < 0) n0 = j;
        n1 = j;
      }
    }
    const ks = [i, j - 1, mn, mx, n0, n1].filter((k) => k >= 0).sort((a, b) => a - b);
    for (let k = 0; k < ks.length; k++) if (k === 0 || ks[k] !== ks[k - 1]) out.push(ks[k]);
    i = j;
  }
  return out;
}

function decimExtremes(y) {
  const n = y.length;
  if (n === 0) return [];
  let mn = -1, mx = -1;
  for (let i = 0; i < n; i++) {
    const v = y[i];
    if (!Number.isFinite(v)) continue;
    if (mn < 0 || v < y[mn]) mn = i;
    if (mx < 0 || v > y[mx]) mx = i;
  }
  return [0, n - 1, mn, mx].filter((k) => k >= 0);
}

// ext: decimExtremes(y), computed once per series by the caller.
function decimIndices(x, y, start, end, cols, ext) {
  const [lo, hi, bw] = decimWindow(start, end, cols);
  return [...new Set(decimKeep(x, y, lo, hi, bw).concat(ext || decimExtremes(y)))]
    .sort((a, b) => a - b);
}

function decimReduce(x, y, start, end, cols, ext) {
  const k = decimIndices(x, y, start, end, cols, ext);
  const X = new Float64Array(k.length), Y = new Float64Array(k.length);
  for (let i = 0; i < k.length; i++) { X[i] = x[k[i]]; Y[i] = y[k[i]]; }
  return [X, Y];
}

if (typeof module !== "undefined") module.exports = { decimWindow, decimKeep, decimExtremes, decimIndices, decimReduce, EXACT_PER_COLUMN };
