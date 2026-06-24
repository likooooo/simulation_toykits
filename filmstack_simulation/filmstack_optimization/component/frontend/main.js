// Freehand R/T/A editor — curve edit vs box zoom

const METRICS = ["R", "T", "A"];
const METRIC_LABELS = { R: "Reflectance", T: "Transmittance", A: "Absorptance" };
const METRIC_COLORS = { R: "#B91C1C", T: "#047857", A: "#6D28D9" };
const CURRENT_CURVE_COLOR = "#374151";
const HIT_PX = 18;
const MIN_ZOOM_PX = 5;
const MIN_SEGMENT_PX = 2;
const PAD = { top: 14, right: 12, bottom: 40, left: 52 };
const Y_PERCENT_CAP = 100;
const Y_PERCENT_FLOOR = 1;
const Y_FRACTION_CAP = 1;

let args = null;
let localTarget = { R: null, T: null, A: null };
let localTouched = { R: false, T: false, A: false };
let activeMetric = "R";
let viewDomain = {};
let dragMode = null; // "edit" | "zoom"
let dragMetric = null;
let dragStart = null;
let dragLast = null;
let editSnapshot = null;
let activeDragSvg = null;

const elCharts = document.getElementById("charts");
const elStatus = document.getElementById("status");

function clamp(v, lo, hi) {
  return Math.max(lo, Math.min(hi, v));
}

function curveMaxFraction(metric) {
  const current = args.current[metric];
  const target = localTarget[metric] || args.target?.[metric];
  let max = 0;
  if (current) {
    for (let i = 0; i < current.length; i++) {
      max = Math.max(max, current[i]);
    }
  }
  if (target) {
    for (let i = 0; i < target.length; i++) {
      max = Math.max(max, target[i]);
    }
  }
  return max;
}

function curveMinFraction(metric) {
  const current = args.current[metric];
  const target = localTarget[metric] || args.target?.[metric];
  let min = Infinity;
  if (current) {
    for (let i = 0; i < current.length; i++) {
      min = Math.min(min, current[i]);
    }
  }
  if (target) {
    for (let i = 0; i < target.length; i++) {
      min = Math.min(min, target[i]);
    }
  }
  return min === Infinity ? 0 : min;
}

function autoYMinPercent(metric) {
  const minFrac = curveMinFraction(metric);
  return Math.max(0, minFrac * 100 * 0.8);
}

function autoYMaxPercent(metric) {
  const maxFrac = curveMaxFraction(metric);
  return Math.max(Math.min(maxFrac * 100 * 1.2, Y_PERCENT_CAP), Y_PERCENT_FLOOR);
}

function defaultDomain(wl, metric) {
  const xs = wl && wl.length ? wl : [0.4, 0.8];
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  return { x: [x0, x1], y: [autoYMinPercent(metric), autoYMaxPercent(metric)], yAuto: true };
}

function isLegacyFractionDomain(dom) {
  return dom && dom.y && dom.y[1] <= Y_FRACTION_CAP && dom.yAuto !== true;
}

function normalizeDomain(dom, wl, metric) {
  if (!dom || isLegacyFractionDomain(dom)) {
    return defaultDomain(wl, metric);
  }
  const out = { ...dom };
  out.x = [Number(out.x[0]), Number(out.x[1])];
  out.y = [
    Math.max(0, Math.min(Y_PERCENT_CAP, Number(out.y[0]))),
    Math.max(0, Math.min(Y_PERCENT_CAP, Number(out.y[1]))),
  ];
  if (out.y[1] <= out.y[0]) {
    out.y[1] = Math.min(out.y[0] + Y_PERCENT_FLOOR, Y_PERCENT_CAP);
  }
  if (out.yAuto === undefined) {
    out.yAuto = false;
  }
  return out;
}

function refreshAutoYDomain(metric) {
  const dom = viewDomain[metric];
  if (dom && dom.yAuto === false) return;
  const yLo = autoYMinPercent(metric);
  const yHi = autoYMaxPercent(metric);
  if (dom) {
    viewDomain[metric] = { ...dom, y: [yLo, yHi], yAuto: true };
  } else {
    viewDomain[metric] = defaultDomain(args.wl, metric);
  }
}

function ensureDomains(wl, incoming) {
  const out = {};
  for (const m of METRICS) {
    out[m] = normalizeDomain(incoming && incoming[m] ? { ...incoming[m] } : null, wl, m);
  }
  return out;
}

function copyTargetArrays(incoming, wlLen) {
  const out = { R: null, T: null, A: null };
  for (const m of METRICS) {
    if (incoming && incoming[m] && incoming[m].length === wlLen) {
      out[m] = incoming[m].slice();
    }
  }
  return out;
}

function setStatus(text, optimizing) {
  elStatus.textContent = text || "";
  elStatus.classList.toggle("optimizing", !!optimizing);
}

function plotRect(svg) {
  const w = svg.clientWidth || 300;
  const h = svg.clientHeight || 220;
  return {
    w,
    h,
    x0: PAD.left,
    y0: PAD.top,
    x1: w - PAD.right,
    y1: h - PAD.bottom,
  };
}

function xToPx(x, domain, rect) {
  const [lo, hi] = domain.x;
  const t = hi === lo ? 0 : (x - lo) / (hi - lo);
  return rect.x0 + t * (rect.x1 - rect.x0);
}

function yToPx(y, domain, rect) {
  const [lo, hi] = domain.y;
  const t = hi === lo ? 0 : (y - lo) / (hi - lo);
  return rect.y1 - t * (rect.y1 - rect.y0);
}

function pxToX(px, domain, rect) {
  const t = (px - rect.x0) / (rect.x1 - rect.x0);
  const [lo, hi] = domain.x;
  return lo + t * (hi - lo);
}

function pxToY(py, domain, rect) {
  const t = (rect.y1 - py) / (rect.y1 - rect.y0);
  const [lo, hi] = domain.y;
  return lo + t * (hi - lo);
}

function applyStrokeSegment(x0, y0, x1, y1, base, modifiedIndices) {
  const wl = args.wl;
  const src = base.slice();
  const xLo = Math.min(x0, x1);
  const xHi = Math.max(x0, x1);
  const dx = x1 - x0;

  for (let i = 0; i < wl.length; i++) {
    const x = wl[i];
    if (x < xLo || x > xHi) continue;
    const t = dx === 0 ? 0.5 : (x - x0) / dx;
    src[i] = clamp(y0 + t * (y1 - y0), 0, 1);
    if (modifiedIndices) modifiedIndices.add(i);
  }
  return src;
}

function formatTick(v, decimals) {
  if (Math.abs(v) >= 100 || (Math.abs(v) < 0.01 && v !== 0)) {
    return v.toExponential(2);
  }
  return v.toFixed(decimals);
}

function formatPercentTick(v) {
  const abs = Math.abs(v);
  if (abs >= 100 || (abs < 0.01 && v !== 0)) {
    return v.toExponential(2) + "%";
  }
  const decimals = abs >= 10 ? 0 : abs >= 1 ? 1 : 2;
  return v.toFixed(decimals) + "%";
}

function appendSvgText(svg, x, y, text, anchor, cls) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", "text");
  el.setAttribute("x", x);
  el.setAttribute("y", y);
  el.setAttribute("text-anchor", anchor || "middle");
  el.classList.add(cls || "axis-label");
  el.textContent = text;
  svg.appendChild(el);
}

function drawAxes(svg, domain, rect) {
  const nTicks = 5;
  for (let i = 0; i < nTicks; i++) {
    const t = i / (nTicks - 1);
    const yVal = domain.y[0] + t * (domain.y[1] - domain.y[0]);
    const py = yToPx(yVal, domain, rect);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", rect.x0 - 4);
    tick.setAttribute("x2", rect.x0);
    tick.setAttribute("y1", py);
    tick.setAttribute("y2", py);
    tick.classList.add("tick-line");
    svg.appendChild(tick);
    appendSvgText(svg, rect.x0 - 6, py + 3, formatPercentTick(yVal), "end");
  }
  for (let i = 0; i < nTicks; i++) {
    const t = i / (nTicks - 1);
    const xVal = domain.x[0] + t * (domain.x[1] - domain.x[0]);
    const px = xToPx(xVal, domain, rect);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", px);
    tick.setAttribute("x2", px);
    tick.setAttribute("y1", rect.y1);
    tick.setAttribute("y2", rect.y1 + 4);
    tick.classList.add("tick-line");
    svg.appendChild(tick);
    appendSvgText(svg, px, rect.y1 + 16, formatTick(xVal, 4), "middle");
  }
  appendSvgText(
    svg,
    (rect.x0 + rect.x1) / 2,
    rect.y1 + 32,
    "\u03bb (\u03bcm)",
    "middle",
    "axis-title",
  );
}

function curvePoints(wl, ys, domain, rect) {
  return wl.map((x, i) => ({
    x,
    y: ys[i],
    px: xToPx(x, domain, rect),
    py: yToPx(ys[i] * 100, domain, rect),
  }));
}

function hitTestCurve(px, py, wl, ys, domain, rect) {
  const pts = curvePoints(wl, ys, domain, rect);
  let best = Infinity;
  for (let i = 0; i < pts.length; i++) {
    const dx = pts[i].px - px;
    const dy = pts[i].py - py;
    const d = Math.hypot(dx, dy);
    if (d < best) best = d;
  }
  return best;
}

function displayCurve(metric, wl) {
  if (localTarget[metric]) return localTarget[metric];
  return args.current[metric];
}

function sendValue(payload) {
  Streamlit.setComponentValue({ ...payload, ts: Date.now() });
}

function updateFrameHeight() {
  const root = document.getElementById("root");
  const h = root ? root.offsetHeight + 8 : 340;
  Streamlit.setFrameHeight(Math.max(h, 280));
}

function drawMeritAxes(svg, domain, rect) {
  const nTicks = 5;
  for (let i = 0; i < nTicks; i++) {
    const t = i / (nTicks - 1);
    const yVal = domain.y[0] + t * (domain.y[1] - domain.y[0]);
    const py = yToPx(yVal, domain, rect);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", rect.x0 - 4);
    tick.setAttribute("x2", rect.x0);
    tick.setAttribute("y1", py);
    tick.setAttribute("y2", py);
    tick.classList.add("tick-line");
    svg.appendChild(tick);
    appendSvgText(svg, rect.x0 - 6, py + 3, formatTick(yVal, 4), "end");
  }
  for (let i = 0; i < nTicks; i++) {
    const t = i / (nTicks - 1);
    const xVal = domain.x[0] + t * (domain.x[1] - domain.x[0]);
    const px = xToPx(xVal, domain, rect);
    const tick = document.createElementNS("http://www.w3.org/2000/svg", "line");
    tick.setAttribute("x1", px);
    tick.setAttribute("x2", px);
    tick.setAttribute("y1", rect.y1);
    tick.setAttribute("y2", rect.y1 + 4);
    tick.classList.add("tick-line");
    svg.appendChild(tick);
    appendSvgText(svg, px, rect.y1 + 16, formatTick(xVal, 0), "middle");
  }
  appendSvgText(
    svg,
    (rect.x0 + rect.x1) / 2,
    rect.y1 + 32,
    "Iteration",
    "middle",
    "axis-title",
  );
}

function convergenceDomain(history) {
  const n = history.length;
  const xLo = 1;
  const xHi = Math.max(n, 2);
  const yMin = Math.min(...history);
  const yMax = Math.max(...history);
  const ySpan = yMax - yMin;
  const pad = ySpan > 0 ? ySpan * 0.05 : Math.max(Math.abs(yMin), 1) * 0.1;
  return {
    x: [xLo, xHi],
    y: [yMin - pad, yMax + pad],
  };
}

function drawConvergenceSvg(svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const rect = plotRect(svg);
  svg.setAttribute("viewBox", `0 0 ${rect.w} ${rect.h}`);

  const history = args.meritHistory;
  if (!history || !history.length) return;

  const domain = convergenceDomain(history);

  for (let i = 0; i <= 4; i++) {
    const y = domain.y[0] + (i / 4) * (domain.y[1] - domain.y[0]);
    const py = yToPx(y, domain, rect);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", rect.x0);
    line.setAttribute("x2", rect.x1);
    line.setAttribute("y1", py);
    line.setAttribute("y2", py);
    line.classList.add("grid-line");
    svg.appendChild(line);
  }

  const axis = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  axis.setAttribute("x", rect.x0);
  axis.setAttribute("y", rect.y0);
  axis.setAttribute("width", rect.x1 - rect.x0);
  axis.setAttribute("height", rect.y1 - rect.y0);
  axis.setAttribute("fill", "transparent");
  axis.setAttribute("stroke", "#D1D5DB");
  axis.classList.add("axis-line");
  svg.appendChild(axis);

  drawMeritAxes(svg, domain, rect);

  const pts = history.map((yVal, i) => ({
    px: xToPx(i + 1, domain, rect),
    py: yToPx(yVal, domain, rect),
  }));

  if (pts.length >= 2) {
    const d = pts.map((p, i) => `${i ? "L" : "M"} ${p.px} ${p.py}`).join(" ");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.classList.add("curve-convergence");
    path.setAttribute("stroke-width", "2.5");
    svg.appendChild(path);
  }

  for (const p of pts) {
    const dot = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    dot.setAttribute("cx", p.px);
    dot.setAttribute("cy", p.py);
    dot.setAttribute("r", "2");
    dot.classList.add("curve-convergence");
    dot.setAttribute("fill", "#4682B4");
    svg.appendChild(dot);
  }
}

function renderConvergenceChart() {
  const panel = document.createElement("div");
  panel.className = "chart-panel convergence readonly";

  const header = document.createElement("div");
  header.className = "chart-header";

  const title = document.createElement("span");
  title.className = "chart-title";
  title.textContent = "Convergence";

  header.appendChild(title);

  const stage = document.createElement("div");
  stage.className = "chart-stage";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("chart-svg", "readonly");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  stage.appendChild(svg);
  panel.appendChild(header);
  panel.appendChild(stage);
  elCharts.appendChild(panel);

  requestAnimationFrame(() => {
    drawConvergenceSvg(svg);
    updateFrameHeight();
  });
}

function renderChart(metric) {
  const panel = document.createElement("div");
  panel.className = "chart-panel" + (activeMetric === metric ? " active" : "");
  panel.dataset.metric = metric;

  const header = document.createElement("div");
  header.className = "chart-header";

  const title = document.createElement("span");
  title.className = "chart-title";
  title.textContent = METRIC_LABELS[metric];

  const actions = document.createElement("div");
  actions.style.display = "flex";
  actions.style.gap = "0.25rem";
  actions.style.alignItems = "center";

  const legend = document.createElement("button");
  legend.type = "button";
  legend.className = "legend-btn";
  legend.style.color = METRIC_COLORS[metric];
  legend.textContent = metric;
  legend.title = "Select for editing";
  legend.onclick = (ev) => {
    ev.stopPropagation();
    if (args.optimizing) return;
    activeMetric = metric;
    sendValue({ type: "activeMetric", activeMetric: metric });
    renderAll();
  };

  const clearBtn = document.createElement("button");
  clearBtn.type = "button";
  clearBtn.className = "clear-btn";
  clearBtn.textContent = "×";
  clearBtn.title = "Clear target";
  clearBtn.style.visibility = localTouched[metric] ? "visible" : "hidden";
  clearBtn.onclick = (ev) => {
    ev.stopPropagation();
    if (args.optimizing) return;
    localTarget[metric] = null;
    localTouched[metric] = false;
    sendValue({ type: "clearTarget", metric });
    renderAll();
  };

  const resetBtn = document.createElement("button");
  resetBtn.type = "button";
  resetBtn.className = "reset-btn";
  resetBtn.textContent = "重置视图";
  resetBtn.onclick = (ev) => {
    ev.stopPropagation();
    if (args.optimizing) return;
    const dom = defaultDomain(args.wl, metric);
    viewDomain[metric] = dom;
    sendValue({ type: "viewChange", viewDomain: viewDomain });
    renderAll();
  };

  actions.appendChild(legend);
  actions.appendChild(clearBtn);
  actions.appendChild(resetBtn);
  header.appendChild(title);
  header.appendChild(actions);

  const stage = document.createElement("div");
  stage.className = "chart-stage";

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("chart-svg");
  if (args.optimizing) svg.classList.add("disabled");
  svg.setAttribute("preserveAspectRatio", "xMidYMid meet");

  stage.appendChild(svg);
  panel.appendChild(header);
  panel.appendChild(stage);
  elCharts.appendChild(panel);

  requestAnimationFrame(() => {
    drawSvg(metric, svg);
    updateFrameHeight();
  });
  bindSvgEvents(metric, svg);
}

function drawSvg(metric, svg) {
  while (svg.firstChild) svg.removeChild(svg.firstChild);
  const rect = plotRect(svg);
  svg.setAttribute("viewBox", `0 0 ${rect.w} ${rect.h}`);

  const domain = viewDomain[metric] || defaultDomain(args.wl, metric);
  const wl = args.wl;
  const current = args.current[metric];
  const target = localTarget[metric];

  // grid
  for (let i = 0; i <= 4; i++) {
    const y = domain.y[0] + (i / 4) * (domain.y[1] - domain.y[0]);
    const py = yToPx(y, domain, rect);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
    line.setAttribute("x1", rect.x0);
    line.setAttribute("x2", rect.x1);
    line.setAttribute("y1", py);
    line.setAttribute("y2", py);
    line.classList.add("grid-line");
    svg.appendChild(line);
  }

  const axis = document.createElementNS("http://www.w3.org/2000/svg", "rect");
  axis.setAttribute("x", rect.x0);
  axis.setAttribute("y", rect.y0);
  axis.setAttribute("width", rect.x1 - rect.x0);
  axis.setAttribute("height", rect.y1 - rect.y0);
  axis.setAttribute("fill", "transparent");
  axis.setAttribute("stroke", "#D1D5DB");
  axis.classList.add("axis-line");
  svg.appendChild(axis);

  drawAxes(svg, domain, rect);

  function pathFor(ys, opts) {
    const pts = curvePoints(wl, ys, domain, rect);
    if (!pts.length) return;
    const d = pts.map((p, i) => `${i ? "L" : "M"} ${p.px} ${p.py}`).join(" ");
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", d);
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", opts.stroke);
    path.setAttribute("stroke-width", String(opts.width));
    if (opts.dash) path.setAttribute("stroke-dasharray", opts.dash);
    if (opts.opacity != null) path.setAttribute("stroke-opacity", String(opts.opacity));
    svg.appendChild(path);
  }

  const editing = dragMode === "edit" && dragMetric === metric;
  pathFor(current, {
    stroke: CURRENT_CURVE_COLOR,
    width: editing ? 2.5 : 2.5,
    dash: editing ? "4 3" : null,
    opacity: editing ? 0.45 : 1,
  });
  if (target) {
    pathFor(target, {
      stroke: METRIC_COLORS[metric],
      width: editing ? 3.5 : 2.5,
    });
  }

  if (editing && editSnapshot) {
    const pts = editSnapshot.pathPx.slice();
    if (dragLast) {
      const last = pts[pts.length - 1];
      if (!last || last.px !== dragLast.px || last.py !== dragLast.py) {
        pts.push(dragLast);
      }
    }
    if (pts.length >= 2) {
      const d = pts.map((p, i) => `${i ? "L" : "M"} ${p.px} ${p.py}`).join(" ");
      const strokePath = document.createElementNS("http://www.w3.org/2000/svg", "path");
      strokePath.setAttribute("d", d);
      strokePath.setAttribute("fill", "none");
      strokePath.setAttribute("stroke", "#2563EB");
      strokePath.setAttribute("stroke-width", "2");
      strokePath.setAttribute("stroke-dasharray", "6 4");
      svg.appendChild(strokePath);
    }
  }

  if (dragMode === "zoom" && dragMetric === metric && dragStart && dragLast) {
    const x1 = Math.min(dragStart.px, dragLast.px);
    const y1 = Math.min(dragStart.py, dragLast.py);
    const x2 = Math.max(dragStart.px, dragLast.px);
    const y2 = Math.max(dragStart.py, dragLast.py);
    const zr = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    zr.setAttribute("x", x1);
    zr.setAttribute("y", y1);
    zr.setAttribute("width", x2 - x1);
    zr.setAttribute("height", y2 - y1);
    zr.classList.add("zoom-rect");
    svg.appendChild(zr);
  }
}

function svgPoint(svg, clientX, clientY) {
  const box = svg.getBoundingClientRect();
  return { px: clientX - box.left, py: clientY - box.top };
}

function clearDocumentDragListeners() {
  document.removeEventListener("mousemove", onDocumentMove);
  document.removeEventListener("mouseup", onDocumentUp);
  activeDragSvg = null;
}

function onDocumentMove(ev) {
  if (!dragMode || !dragMetric || !activeDragSvg) return;
  const svg = activeDragSvg;
  const { px, py } = svgPoint(svg, ev.clientX, ev.clientY);
  dragLast = { px, py };
  if (dragMode === "edit" && editSnapshot) {
    const rect = plotRect(svg);
    const domain = viewDomain[dragMetric];
    const x1 = pxToX(px, domain, rect);
    const y1 = clamp(pxToY(py, domain, rect) / 100, 0, Y_FRACTION_CAP);
    const lastPx = editSnapshot.pathPx[editSnapshot.pathPx.length - 1];
    const segPx = Math.hypot(px - lastPx.px, py - lastPx.py);
    if (segPx >= MIN_SEGMENT_PX) {
      const next = applyStrokeSegment(
        editSnapshot.x0,
        editSnapshot.y0,
        x1,
        y1,
        editSnapshot.base,
        editSnapshot.modifiedIndices,
      );
      localTarget[dragMetric] = next;
      localTouched[dragMetric] = true;
      editSnapshot.base = next.slice();
      editSnapshot.x0 = x1;
      editSnapshot.y0 = y1;
      editSnapshot.pathPx.push({ px, py });
    }
    setStatus("编辑目标曲线…", false);
    drawSvg(dragMetric, svg);
  } else if (dragMode === "zoom") {
    drawSvg(dragMetric, svg);
  }
}

function onDocumentUp(ev) {
  if (!dragMode || !dragMetric || !activeDragSvg) return;
  const svg = activeDragSvg;
  const metric = dragMetric;
  const rect = plotRect(svg);
  const domain = viewDomain[metric];

  if (dragMode === "edit") {
    if (editSnapshot && dragLast) {
      const x1 = pxToX(dragLast.px, domain, rect);
      const y1 = clamp(pxToY(dragLast.py, domain, rect) / 100, 0, Y_FRACTION_CAP);
      const lastPx = editSnapshot.pathPx[editSnapshot.pathPx.length - 1];
      const segPx = Math.hypot(dragLast.px - lastPx.px, dragLast.py - lastPx.py);
      if (segPx > 0) {
        const next = applyStrokeSegment(
          editSnapshot.x0,
          editSnapshot.y0,
          x1,
          y1,
          editSnapshot.base,
        );
        localTarget[metric] = next;
        localTouched[metric] = true;
      }
    }
    const editWlIndices = {};
    if (editSnapshot && editSnapshot.modifiedIndices && editSnapshot.modifiedIndices.size) {
      editWlIndices[metric] = [...editSnapshot.modifiedIndices].sort((a, b) => a - b);
    }
    sendValue({
      type: "curveDragEnd",
      metric,
      target: { R: localTarget.R, T: localTarget.T, A: localTarget.A },
      touched: { ...localTouched },
      editWlIndices,
      triggerOptimize: true,
    });
    setStatus("已提交目标曲线，等待优化…", false);
  } else if (dragMode === "zoom" && dragStart && dragLast) {
    const dx = Math.abs(dragLast.px - dragStart.px);
    const dy = Math.abs(dragLast.py - dragStart.py);
    if (dx >= MIN_ZOOM_PX && dy >= MIN_ZOOM_PX) {
      const xLo = pxToX(Math.min(dragStart.px, dragLast.px), domain, rect);
      const xHi = pxToX(Math.max(dragStart.px, dragLast.px), domain, rect);
      const yLo = pxToY(Math.max(dragStart.py, dragLast.py), domain, rect);
      const yHi = pxToY(Math.min(dragStart.py, dragLast.py), domain, rect);
      viewDomain[metric] = {
        x: [Math.min(xLo, xHi), Math.max(xLo, xHi)],
        y: [
          Math.max(0, Math.min(yLo, yHi)),
          Math.min(Y_PERCENT_CAP, Math.max(yLo, yHi)),
        ],
        yAuto: false,
      };
      sendValue({ type: "viewChange", viewDomain });
    }
  }

  dragMode = null;
  dragMetric = null;
  dragStart = null;
  dragLast = null;
  editSnapshot = null;
  clearDocumentDragListeners();
  renderAll();
  ev.preventDefault();
}

function bindSvgEvents(metric, svg) {
  svg.addEventListener("mousedown", (ev) => {
    if (args.optimizing || ev.button !== 0) return;
    const rect = plotRect(svg);
    const px = ev.offsetX;
    const py = ev.offsetY;
    if (px < rect.x0 || px > rect.x1 || py < rect.y0 || py > rect.y1) return;

    const domain = viewDomain[metric];

    if (ev.altKey) {
      dragMode = "zoom";
      dragMetric = metric;
      activeDragSvg = svg;
      dragStart = { px, py };
      dragLast = { px, py };
      document.addEventListener("mousemove", onDocumentMove);
      document.addEventListener("mouseup", onDocumentUp);
    } else {
      activeMetric = metric;
      dragMode = "edit";
      dragMetric = metric;
      activeDragSvg = svg;
      if (!localTarget[metric]) {
        localTarget[metric] = args.current[metric].slice();
      }
      const x0 = pxToX(px, domain, rect);
      const y0 = clamp(pxToY(py, domain, rect) / 100, 0, Y_FRACTION_CAP);
      const base = localTarget[metric]
        ? localTarget[metric].slice()
        : args.current[metric].slice();
      editSnapshot = { x0, y0, base, pathPx: [{ px, py }], modifiedIndices: new Set() };
      dragStart = { px, py };
      dragLast = { px, py };
      setStatus("编辑目标曲线…", false);
      drawSvg(metric, svg);
      document.addEventListener("mousemove", onDocumentMove);
      document.addEventListener("mouseup", onDocumentUp);
    }
    ev.preventDefault();
  });

  svg.addEventListener("dblclick", (ev) => {
    if (args.optimizing) return;
    const rect = plotRect(svg);
    const px = ev.offsetX;
    const py = ev.offsetY;
    const domain = viewDomain[metric];
    const dist = hitTestCurve(px, py, args.wl, displayCurve(metric, args.wl), domain, rect);
    if (dist <= HIT_PX) return;
    viewDomain[metric] = defaultDomain(args.wl, metric);
    sendValue({ type: "viewChange", viewDomain });
    renderAll();
  });
}

function renderAll() {
  elCharts.innerHTML = "";
  for (const m of METRICS) renderChart(m);
  if (args.meritHistory && args.meritHistory.length > 0) {
    renderConvergenceChart();
  }
  requestAnimationFrame(updateFrameHeight);
}

function onRender(event) {
  const incoming = event.detail.args;
  if (!incoming || !incoming.wl || !incoming.wl.length) {
    setStatus("请先构建膜系以显示 R/T/A 曲线。");
    return;
  }

  const wlLen = incoming.wl.length;
  args = incoming;

  if (dragMode !== null) {
    return;
  }

  if (incoming.resetTargets) {
    localTarget = copyTargetArrays(incoming.target, wlLen);
    localTouched = { ...(incoming.touched || { R: false, T: false, A: false }) };
    viewDomain = ensureDomains(incoming.wl, incoming.viewDomain || viewDomain);
    activeMetric = incoming.activeMetric || activeMetric || "R";
  } else {
    localTarget = copyTargetArrays(incoming.target, wlLen);
    for (const m of METRICS) {
      localTouched[m] = !!(incoming.touched && incoming.touched[m]);
    }
    viewDomain = ensureDomains(incoming.wl, incoming.viewDomain || viewDomain);
    activeMetric = incoming.activeMetric || activeMetric || "R";
  }

  for (const m of METRICS) {
    refreshAutoYDomain(m);
  }

  if (incoming.optimizing) {
    setStatus("Optimizing… drag disabled.", true);
  } else {
    setStatus("在画布上拖动画线编辑目标（折线链式插值）；松手优化。Alt+拖动缩放视图。");
  }

  renderAll();
}

Streamlit.events.addEventListener(Streamlit.RENDER_EVENT, onRender);
Streamlit.setComponentReady();
updateFrameHeight();

window.addEventListener("resize", () => {
  if (args && dragMode === null) renderAll();
  updateFrameHeight();
});
