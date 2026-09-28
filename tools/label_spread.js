// label_spread.js — tools/label_spread.py for the Bokeh page, line for
// line: spread1d = spread_1d, assignRows = assign_rows, placeBarLabels =
// place_bar_labels (same rules, same results; tests/python/
// test_label_spread_js.py runs both on the same inputs). The page's
// CustomJS callbacks inline this file and re-place the process timeline's,
// the region strip's and the event strip's labels on every x-range change.
// textWidth(t) is a function; an item is [wanted centre, width]; a bar is
// [left, right, texts]; a placed label is ["in"|"out", text, x, anchor, row]
// or null.

function spread1d(items, lo, hi, pad) {
  pad = pad || 0.0;
  const n = items.length;
  if (n === 0) return [];
  const order = [...Array(n).keys()].sort((i, j) => items[i][0] - items[j][0] || i - j);
  const offs = new Array(n).fill(0.0), clusters = [];
  const width = (c) => offs[c[1]] + items[order[c[1]]][1] - offs[c[0]] + 0.0;
  const place = (c) => {
    let sum = 0.0;
    for (let j = c[0]; j <= c[1]; j++) sum += items[order[j]][0] - items[order[j]][1] / 2 - offs[j];
    const left = sum / (c[1] - c[0] + 1);
    c[2] = Math.min(Math.max(left, lo), Math.max(lo, hi - width(c)));
  };
  for (let j = 0; j < n; j++) {
    offs[j] = 0.0;
    const c = [j, j, 0.0];
    place(c);
    clusters.push(c);
    while (clusters.length > 1) {
      const p = clusters[clusters.length - 2], q = clusters[clusters.length - 1];
      if (p[2] + width(p) + pad <= q[2]) break;
      const base = offs[p[1]] + items[order[p[1]]][1] + pad;
      const q0 = offs[q[0]];
      for (let k = q[0]; k <= q[1]; k++) offs[k] = base + offs[k] - q0;
      clusters.pop();
      p[1] = q[1];
      place(p);
    }
  }
  const out = new Array(n).fill(0.0);
  for (const [a, b, left] of clusters)
    for (let j = a; j <= b; j++) out[order[j]] = left + offs[j] + items[order[j]][1] / 2;
  return out;
}

function assignRows(items, lo, hi, pad, maxRows, maxShift) {
  pad = pad || 0.0;
  maxRows = maxRows === undefined ? 64 : maxRows;
  const n = items.length;
  if (n === 0) return [[], []];
  const order = [...Array(n).keys()].sort((i, j) => items[i][0] - items[j][0] || i - j);
  const span = hi - lo;
  const layout = (rows) => {
    const row = new Array(n).fill(0);
    order.forEach((i, r) => { row[i] = r % rows; });
    const centres = new Array(n).fill(0.0);
    for (let r = 0; r < rows; r++) {
      const idx = [];
      for (let i = 0; i < n; i++) if (row[i] === r) idx.push(i);
      const cs = spread1d(idx.map((i) => items[i]), lo, hi, pad);
      idx.forEach((i, k) => { centres[i] = cs[k]; });
    }
    return [row, centres];
  };
  let rows = 1;
  for (;;) {
    const loads = new Array(rows).fill(0.0);
    order.forEach((i, r) => { loads[r % rows] += items[i][1] + pad; });
    const fits = Math.max(...loads) <= span;
    if (fits || rows >= maxRows) {
      const [row, centres] = layout(rows);
      if (maxShift === null || maxShift === undefined || rows >= maxRows
          || Math.max(...centres.map((c, i) => Math.abs(c - items[i][0]))) <= maxShift)
        return [row, centres];
    }
    rows += 1;
  }
}

function placeBarLabels(bars, lo, hi, widthUnits, textWidth, pad, maxRows, maxShift) {
  pad = pad || 0.0;
  maxRows = maxRows === undefined ? 64 : maxRows;          // null: no limit
  const per = widthUnits / Math.max(hi - lo, 1e-12);
  const out = new Array(bars.length).fill(null);
  let outside = [];                               // [index, anchor, text, width, visible]
  bars.forEach(([left, right, texts], i) => {
    if (left > hi || right < lo) return;
    const a = Math.max(left, lo), b = Math.min(right, hi);
    const vis = (b - a) * per;
    const fit = texts.find((t) => textWidth(t) + pad < vis);
    if (fit !== undefined) out[i] = ["in", fit, (a + b) / 2, (a + b) / 2, -1];
    else outside.push([i, (a + b) / 2, texts[0], textWidth(texts[0]), vis]);
  });
  if (maxRows !== null && maxRows < 1)
    outside = [];                                 // no label rows
  while (outside.length) {
    const items = outside.map(([_i, anc, _t, w]) => [(anc - lo) * per, w]);
    const cap = maxRows === null ? outside.length : maxRows;
    const [rows, centres] = assignRows(items, 0.0, widthUnits, pad, cap, maxShift);
    const loads = new Array(Math.max(...rows) + 1).fill(0.0);
    const count = new Array(loads.length).fill(0);
    rows.forEach((r, k) => { loads[r] += items[k][1] + pad; count[r] += 1; });
    if (loads.every((ld, r) => ld <= widthUnits || count[r] === 1)) {
      outside.forEach(([i, anc, text], k) => { out[i] = ["out", text, lo + centres[k] / per, anc, rows[k]]; });
      break;
    }
    // Too many for maxRows: leave out the narrowest visible bar's label.
    let m = 0;
    for (let k = 1; k < outside.length; k++)
      if (outside[k][4] < outside[m][4] || (outside[k][4] === outside[m][4] && outside[k][0] < outside[m][0])) m = k;
    outside.splice(m, 1);
  }
  return out;
}

if (typeof module !== "undefined") module.exports = { spread1d, assignRows, placeBarLabels };
