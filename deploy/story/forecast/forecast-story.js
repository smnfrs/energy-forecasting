/* Page script for the Stage 10 forecast story. Fetches published JSON (facts,
   narratives, forecasts, SHAP attribution) and builds each section's chart(s). */

const YEARLY_DATA = "data/facts_yearly.json";
const DEPLOY_DATA = "../../data";

// Same fixed categorical order/colors as Stage 9's charts.js GEN_GROUPS, so the
// two pages read consistently. Load gets --orange (not --red) to avoid clashing
// with coal, which already owns --red in this fixed stack order.
const GEN_GROUPS = [
  { key: "wind", label: "Wind", color: "--blue" },
  { key: "solar", label: "Solar", color: "--aqua" },
  { key: "hydro_biomass", label: "Hydro & biomass", color: "--yellow" },
  { key: "nuclear", label: "Nuclear", color: "--green" },
  { key: "gas", label: "Natural gas", color: "--violet" },
  { key: "coal", label: "Coal", color: "--red" },
  { key: "other", label: "Other", color: "--magenta" },
];

const GEN_LOAD_TARGETS = [
  { key: "wind_onshore", label: "Wind onshore", unit: "MW" },
  { key: "wind_offshore", label: "Wind offshore", unit: "MW" },
  { key: "solar", label: "Solar", unit: "MW" },
  { key: "load", label: "Load", unit: "MW" },
];

const SHAP_CATEGORY_LABELS = {
  gas: "Gas (TTF)", carbon: "Carbon (EUA)", oil: "Oil (Brent)",
  neighbour_prices: "Neighbour prices", wind: "Wind", solar: "Solar",
  residual_gen: "Residual generation", conventional_gen: "Conventional generation",
  cross_border: "Cross-border flows", load: "Load", price_momentum: "Price momentum",
  calendar: "Calendar", other: "Other",
};

// ── Scene registry ─────────────────────────────────────────────────────
// Each chart chapter registers a scene: a beat count and an idempotent
// applyBeat(step) that sets the *full* chart state for that step. The page is a
// slide deck (one chapter per slide, stepped with the arrows/buttons in
// index.html), and each chart is shown in its final, fully-annotated state — so
// every scene is initialised to its last beat. applyBeat stays exposed on
// window.SCENES in case a future build wants to re-introduce intermediate reveals.
const SCENES = {};
window.SCENES = SCENES;
// Shared handoff for the price-forecast scene, whose two charts (price + SHAP)
// are built by separate async functions; each populates its slice, then the
// scene registers once both are ready.
const sceneState = { "forecast-price": { priceReady: false, shapReady: false } };

function registerScene(id, steps, applyBeat) {
  SCENES[id] = { steps, applyBeat };
  // Deck mode: land each chart on its final, fully-annotated beat.
  try { applyBeat(steps - 1); } catch (e) { console.warn("scene init", id, e); }
}

/** Set a computed beat's text slot; hide the whole beat when `text` is null, so a
 *  beat whose data is absent is dropped rather than showing a blank/undefined. */
function fillBeat(scene, step, key, text) {
  const beat = document.querySelector(`[data-scene="${scene}"] .beat[data-step="${step}"]`);
  if (!beat) return;
  if (text == null) { beat.hidden = true; return; }
  beat.hidden = false;
  const slot = key ? beat.querySelector(`[data-fill="${key}"]`) : null;
  if (slot) slot.textContent = text;
}

/** The div, only if Plotly has drawn into it — so restyle/relayout can't throw
 *  when a scene beat fires before its chart's async build has finished. */
function plotted(id) { const d = document.getElementById(id); return d && d.data ? d : null; }

// ── Chapter 2: yearly generation & load ────────────────────────────────

/** Resample a daily gen/load record to weekly means (7-day blocks), so the year
 *  reads as ~52 points instead of a 365-point hairball. Averaging within each
 *  block keeps the "MWh/day" scale intact and comparable. Returns week-start
 *  dates plus per-group and load weekly-mean arrays. Pure/testable. */
function weeklyGenLoad(gl) {
  const dates = gl.date || [];
  const keys = GEN_GROUPS.map(g => g.key);
  const out = { date: [], generation: {}, load: [] };
  keys.forEach(k => { out.generation[k] = []; });
  for (let start = 0; start < dates.length; start += 7) {
    const end = Math.min(start + 7, dates.length);
    out.date.push(dates[start]);
    keys.forEach(k => { out.generation[k].push(meanOf((gl.generation?.[k] || []).slice(start, end))); });
    out.load.push(meanOf((gl.load || []).slice(start, end)));
  }
  return out;
}

async function buildYearlyGenLoadChart() {
  const facts = await fetchJSON(YEARLY_DATA);
  if (!facts) return;
  const gl = facts.current_year.gen_load;
  const wk = weeklyGenLoad(gl);
  const genIdx = GEN_GROUPS.map((_, i) => i);
  const traces = GEN_GROUPS.map(g => ({
    x: wk.date, y: wk.generation[g.key], name: g.label, type: "scatter", mode: "lines",
    stackgroup: "gen", line: { width: 0.5, color: cssVar(g.color) },
    fillcolor: hexToRgba(cssVar(g.color), 0.55),
  }));
  // The load line is hidden by default (overlaying it on the stack invites the
  // "gap = imports" misread) and fades in only on its dedicated beat.
  traces.push({
    x: wk.date, y: wk.load, name: "Load", type: "scatter", mode: "lines",
    line: { width: 2, color: cssVar("--orange") }, visible: false,
  });
  const loadIdx = traces.length - 1;
  Plotly.newPlot("chart-yearly-gen-load", traces, baseLayout({
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "MWh/day (weekly mean)" }),
    showlegend: true,
    legend: { orientation: "h", y: 1.12, x: 0 },
  }), plotConfig);
  renderTable("table-chart-yearly-gen-load",
    ["Week of", ...GEN_GROUPS.map(g => g.label), "Load"],
    wk.date.map((d, i) => [d, ...GEN_GROUPS.map(g => Math.round(wk.generation[g.key][i])), Math.round(wk.load[i])]));

  // Beats: 0 = renewable share, 1 = wind/coal shift (dim the rest), 2 = load reveal.
  const mix = gl.fuel_mix_pct || {};
  fillBeat("yearly-gen-load", 0, "renew",
    (isNum(mix.wind) && isNum(mix.solar)) ? `${Math.round(mix.wind + mix.solar)}%` : null);
  const prior = facts?.prior_year?.gen_load?.fuel_mix_pct;
  const shifts = [];
  if (prior) for (const key of ["wind", "coal"]) {
    if (isNum(mix[key]) && isNum(prior[key])) {
      const d = mix[key] - prior[key];
      shifts.push(`${key} ${d >= 0 ? "up" : "down"} ${Math.abs(d).toFixed(1)} points`);
    }
  }
  fillBeat("yearly-gen-load", 1, "shift", shifts.length ? shifts.join(" and ") : null);

  const WIND = 0, COAL = 5;  // indices into GEN_GROUPS
  registerScene("yearly-gen-load", 3, (step) => {
    const div = plotted("chart-yearly-gen-load"); if (!div) return;
    const fills = GEN_GROUPS.map((g, i) => {
      const focused = step !== 1 || i === WIND || i === COAL;
      return hexToRgba(cssVar(g.color), focused ? 0.55 : 0.12);
    });
    Plotly.restyle(div, { fillcolor: fills }, genIdx);
    Plotly.restyle(div, { visible: step >= 2 }, [loadIdx]);
  });
}

// ── Chapter 3: yearly prices ────────────────────────────────────────────

/** Collapse hourly prices to per-day mean / min / max. The band between the daily
 *  low and high is the day's price swing — the page's whole thesis — which the
 *  8,760-point hourly smear buried. Returns parallel arrays. Pure/testable. */
function dailyPriceBands(price) {
  const ts = price?.timestamp || [];
  const px = price?.price || [];
  const byDay = new Map();
  for (let i = 0; i < ts.length; i++) {
    const v = px[i];
    if (!isNum(v)) continue;
    const day = (ts[i] || "").slice(0, 10);
    if (!day) continue;
    if (!byDay.has(day)) byDay.set(day, []);
    byDay.get(day).push(v);
  }
  const date = [], mean = [], lo = [], hi = [];
  for (const [day, vals] of byDay) {
    date.push(day);
    mean.push(meanOf(vals));
    lo.push(Math.min(...vals));
    hi.push(Math.max(...vals));
  }
  return { date, mean, lo, hi };
}

async function buildYearlyPriceChart() {
  const facts = await fetchJSON(YEARLY_DATA);
  if (!facts) return;
  const price = facts.current_year.price;
  const b = dailyPriceBands(price);
  const bandColor = hexToRgba(cssVar("--blue"), 0.15);
  // Trace order fixed for the scene: 0 = daily high (band anchor), 1 = daily low
  // (fills up to the high), 2 = daily-mean line.
  const traces = [
    { x: b.date, y: b.hi, type: "scatter", mode: "lines", line: { width: 0 },
      showlegend: false, hoverinfo: "skip", name: "Daily high" },
    { x: b.date, y: b.lo, type: "scatter", mode: "lines", line: { width: 0 },
      fill: "tonexty", fillcolor: bandColor, showlegend: false, hoverinfo: "skip", name: "Daily low" },
    { x: b.date, y: b.mean, type: "scatter", mode: "lines",
      line: { width: 1.5, color: cssVar("--blue") }, name: "Daily mean",
      hovertemplate: "%{x}: €%{y:.0f}/MWh<extra></extra>" },
  ];
  Plotly.newPlot("chart-yearly-price", traces, baseLayout({
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "EUR/MWh", zeroline: true, zerolinecolor: cssVar("--axis") }),
  }), plotConfig);

  // Beats: 0 = mean line only, 1 = swing band fades in, 2 = mark the deepest trough.
  fillBeat("yearly-price", 0, "mean", isNum(price.mean_price) ? `€${Math.round(price.mean_price)}/MWh` : null);
  fillBeat("yearly-price", 2, "neg",
    (isNum(price.negative_price_hours) && price.negative_price_hours > 0) ? `${price.negative_price_hours} hours` : null);
  let troughX = null, troughY = null;
  for (let i = 0; i < b.lo.length; i++) {
    if (isNum(b.lo[i]) && (troughY === null || b.lo[i] < troughY)) { troughY = b.lo[i]; troughX = b.date[i]; }
  }
  const troughNote = (troughY !== null && troughY < 0) ? [{
    x: troughX, y: troughY, text: `Low: €${Math.round(troughY)}/MWh`, showarrow: true, arrowhead: 3,
    ax: 0, ay: -26, font: { size: 11, color: cssVar("--text-secondary") }, arrowcolor: cssVar("--axis"),
  }] : [];
  registerScene("yearly-price", 3, (step) => {
    const div = plotted("chart-yearly-price"); if (!div) return;
    Plotly.restyle(div, { visible: step >= 1 }, [0, 1]);
    Plotly.relayout(div, { annotations: step >= 2 ? troughNote : [] });
  });
}

// ── Templated narrative (deterministic; replaces the dormant Groq/LLM path) ──
// Each *Sentence(...) is a pure function of already-published JSON: it returns a
// human sentence string, or null when its load-bearing inputs are absent, so the
// box degrades to the muted fallback and never prints undefined/NaN/"null %".
// Kept DOM- and fetch-free so they can be unit-tested against fixture JSON.

function isNum(v) { return typeof v === "number" && isFinite(v); }
function meanOf(arr) {
  const xs = (arr || []).filter(isNum);
  return xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null;
}
function hourLabel(h) { return `${h}:00`; }
// e.g. 10.3 → "10% above"; -6 → "6% below"; 0.4 → "in line with".
function describeDeviation(pct) {
  const a = Math.round(Math.abs(pct));
  if (a < 1) return "in line with";
  return `${a}% ${pct >= 0 ? "above" : "below"}`;
}

/** Yearly generation & load summary from facts_yearly.json. */
function yearlyGenLoadSentence(facts) {
  const cur = facts?.current_year?.gen_load;
  const mix = cur?.fuel_mix_pct;
  if (!mix || !isNum(mix.wind) || !isNum(mix.solar)) return null;
  const parts = [];

  const renew = Math.round(mix.wind + mix.solar);
  let lead = `Over the last 12 months, wind and solar together supplied ${renew}% of Germany's electricity`;
  if (isNum(mix.coal)) lead += `, ahead of coal at ${Math.round(mix.coal)}%`;
  parts.push(lead + ".");

  const prior = facts?.prior_year?.gen_load?.fuel_mix_pct;
  if (prior) {
    const shifts = [];
    for (const key of ["wind", "coal"]) {
      if (isNum(mix[key]) && isNum(prior[key])) {
        const d = mix[key] - prior[key];
        const label = GEN_GROUPS.find(g => g.key === key)?.label ?? key;
        shifts.push(`${label.toLowerCase()} ${d >= 0 ? "up" : "down"} ${Math.abs(d).toFixed(1)} points`);
      }
    }
    if (shifts.length) parts.push(`Versus the 12 months before, ${shifts.join(" and ")}.`);
  }

  if (isNum(cur.imports_pct_of_domestic_gen) && isNum(cur.exports_pct_of_domestic_gen)) {
    parts.push(`Germany imported the equivalent of ${Math.round(cur.imports_pct_of_domestic_gen)}% ` +
      `of its own generation and exported ${Math.round(cur.exports_pct_of_domestic_gen)}%.`);
  }
  return parts.join(" ");
}

/** Yearly price summary from facts_yearly.json. */
function yearlyPriceSentence(facts) {
  const cur = facts?.current_year?.price;
  if (!cur || !isNum(cur.mean_price)) return null;
  const parts = [];

  let lead = `Day-ahead power averaged €${Math.round(cur.mean_price)}/MWh over the last 12 months`;
  const prior = facts?.prior_year?.price;
  if (prior && isNum(prior.mean_price) && Math.abs(cur.mean_price - prior.mean_price) >= 1) {
    lead += `, ${cur.mean_price >= prior.mean_price ? "up" : "down"} from €${Math.round(prior.mean_price)} the year before`;
  }
  parts.push(lead + ".");

  if (isNum(cur.most_expensive_hour) && isNum(cur.most_expensive_hour_avg_price) &&
      isNum(cur.least_expensive_hour) && isNum(cur.least_expensive_hour_avg_price)) {
    parts.push(`On an average day it peaked around ${hourLabel(cur.most_expensive_hour)} ` +
      `(€${Math.round(cur.most_expensive_hour_avg_price)}/MWh) and was cheapest near ` +
      `${hourLabel(cur.least_expensive_hour)} (€${Math.round(cur.least_expensive_hour_avg_price)}/MWh).`);
  }
  if (isNum(cur.negative_price_hours) && cur.negative_price_hours > 0) {
    parts.push(`Prices fell below zero in ${cur.negative_price_hours} hours — moments with more ` +
      `supply than the grid could use.`);
  }
  return parts.join(" ");
}

async function loadYearlyNarrative() {
  const facts = await fetchJSON(YEARLY_DATA);
  renderTemplated("narrative-gen-load-yearly", facts ? yearlyGenLoadSentence(facts) : null);
  renderTemplated("narrative-price-yearly", facts ? yearlyPriceSentence(facts) : null);
}

// ── Chapter 4: gen/load forecast vs recent actuals ─────────────────────

async function buildForecastGenLoadCharts() {
  const actuals = await fetchJSON(`${DEPLOY_DATA}/gen_load_actuals.json`);
  const traces = [];
  const layout = baseLayout({
    grid: { rows: 2, columns: 2, pattern: "independent" },
    margin: { l: 56, r: 16, t: 28, b: 32 },
    showlegend: true,
    legend: { orientation: "h", y: 1.14, x: 0 },
  });

  // Per-target deviation of tomorrow's forecast vs the recent 7-day actual mean,
  // plus each target's subplot axis refs, so the standout beat can annotate it.
  const series = [];
  for (let i = 0; i < GEN_LOAD_TARGETS.length; i++) {
    const t = GEN_LOAD_TARGETS[i];
    const forecast = await fetchJSON(`${DEPLOY_DATA}/gen_load/${t.key}_national.json`);
    if (!forecast) continue;
    const xa = i === 0 ? "x" : `x${i + 1}`;
    const ya = i === 0 ? "y" : `y${i + 1}`;

    const actualDays = actuals?.[t.key]?.days ?? [];
    if (actualDays.length) {
      const actualX = [], actualY = [];
      for (const day of actualDays) {
        day.values.forEach((v, h) => { actualX.push(`${day.date}T${String(h).padStart(2, "0")}:00`); actualY.push(v); });
      }
      traces.push({
        x: actualX, y: actualY, name: "Recent actual", legendgroup: "actual", showlegend: i === 0,
        type: "scatter", mode: "lines", line: { width: 1.5, color: cssVar("--gray-context") }, xaxis: xa, yaxis: ya,
      });
    }
    traces.push({
      x: forecast.forecasts.map(f => f.timestamp), y: forecast.forecasts.map(f => f.forecast),
      name: "Forecast", legendgroup: "forecast", showlegend: i === 0,
      type: "scatter", mode: "lines", line: { width: 2, color: cssVar("--blue") }, xaxis: xa, yaxis: ya,
    });

    const xKey = xa === "x" ? "xaxis" : xa.replace("x", "xaxis");
    const yKey = ya === "y" ? "yaxis" : ya.replace("y", "yaxis");
    layout[xKey] = { linecolor: cssVar("--axis"), gridcolor: "transparent" };
    layout[yKey] = { title: `${t.label} (${t.unit})`, linecolor: cssVar("--axis"), gridcolor: cssVar("--grid") };

    const fMean = meanOf(forecast.forecasts.slice(0, 24).map(f => f.forecast));
    const rMean = meanOf(actualDays.slice(-7).flatMap(d => d.values ?? []));
    series.push({
      label: t.label, axisSuffix: i === 0 ? "" : String(i + 1),
      pct: (isNum(fMean) && isNum(rMean) && rMean !== 0) ? (fMean - rMean) / rMean * 100 : null,
    });
  }

  Plotly.newPlot("chart-forecast-gen-load", traces, layout, plotConfig);

  // Standout beat: biggest non-Load deviation ≥5% (else Load if ≥5%, else "all calm").
  const dev = series.filter(s => isNum(s.pct));
  const nonLoad = dev.filter(s => s.label !== "Load").sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct));
  let standout = (nonLoad.length && Math.abs(nonLoad[0].pct) >= 5) ? nonLoad[0] : null;
  if (!standout) { const load = dev.find(s => s.label === "Load"); if (load && Math.abs(load.pct) >= 5) standout = load; }
  const standoutText = standout
    ? `${standout.label.toLowerCase()} looks ${describeDeviation(standout.pct)} this week's average`
    : (dev.length ? "wind, solar and load all sit close to their recent averages" : null);
  fillBeat("forecast-gen-load", 1, "standout", standoutText);

  const note = standout ? [{
    xref: `x${standout.axisSuffix} domain`, yref: `y${standout.axisSuffix} domain`,
    x: 0.5, y: 1, yanchor: "bottom", showarrow: false,
    text: `${describeDeviation(standout.pct)} recent average`,
    font: { size: 11, color: cssVar("--orange") },
  }] : [];
  registerScene("forecast-gen-load", 2, (step) => {
    const div = plotted("chart-forecast-gen-load"); if (!div) return;
    Plotly.relayout(div, { annotations: step >= 1 ? note : [] });
  });
}

// ── Chapter 5: price forecast + SHAP drivers ───────────────────────────

async function buildForecastPriceChart() {
  const forecast = await fetchJSON(`${DEPLOY_DATA}/price_forecast.json`);
  if (!forecast) return;
  const actuals = await fetchJSON(`${DEPLOY_DATA}/actuals.json`);

  const traces = [];
  if (actuals?.days?.length) {
    const recent = actuals.days.slice(-7);
    const x = [], y = [];
    for (const day of recent) {
      day.prices.forEach((v, h) => { x.push(`${day.date}T${String(h).padStart(2, "0")}:00`); y.push(v); });
    }
    traces.push({ x, y, name: "Recent actual", type: "scatter", mode: "lines", line: { width: 1.5, color: cssVar("--gray-context") } });
  }

  const fx = forecast.forecasts.map(f => f.timestamp);
  const lower = forecast.forecasts.map(f => f.forecast_lower);
  const upper = forecast.forecasts.map(f => f.forecast_upper);
  let bandIdx = null;
  if (lower.some(v => v !== undefined)) {
    const u = traces.length;
    traces.push({ x: fx, y: upper, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
    const l = traces.length;
    traces.push({ x: fx, y: lower, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
      fillcolor: hexToRgba(cssVar("--blue"), 0.15), showlegend: false, hoverinfo: "skip" });
    bandIdx = [u, l];
  }
  traces.push({
    x: fx, y: forecast.forecasts.map(f => f.forecast), name: "Forecast", type: "scatter", mode: "lines",
    line: { width: 2, color: cssVar("--blue") },
  });

  Plotly.newPlot("chart-forecast-price", traces, baseLayout({
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "EUR/MWh" }),
    showlegend: true,
    legend: { orientation: "h", y: 1.12, x: 0 },
  }), plotConfig);

  Object.assign(sceneState["forecast-price"], { priceReady: true, bandIdx });
  tryRegisterForecastPrice();
}

/** Register the price-forecast scene once both its charts (price + SHAP) exist.
 *  Beats: 0 = forecast line only, 1 = PI band fades in, 2 = highlight the top
 *  SHAP driver. Each half degrades independently if its data was missing. */
function tryRegisterForecastPrice() {
  const st = sceneState["forecast-price"];
  if (!st.priceReady || !st.shapReady) return;
  registerScene("forecast-price", 3, (step) => {
    const pdiv = plotted("chart-forecast-price");
    if (pdiv && st.bandIdx) Plotly.restyle(pdiv, { visible: step >= 1 }, st.bandIdx);
    const sdiv = plotted("chart-shap");
    if (sdiv && st.shapCount) {
      const op = new Array(st.shapCount).fill(1);
      if (step >= 2) { op.fill(0.3); op[st.shapTopIdx] = 1; }
      Plotly.restyle(sdiv, { "marker.opacity": [op] });
    }
  });
}

async function buildShapChart() {
  const shap = await fetchJSON(`${DEPLOY_DATA}/price_shap.json`);
  if (!shap) return;
  const categories = shap.categories;
  const meanContribution = categories.map(c => {
    const vals = shap.category_contributions[c];
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  });
  const order = categories
    .map((c, i) => ({ c, v: meanContribution[i] }))
    .sort((a, b) => Math.abs(a.v) - Math.abs(b.v));

  const trace = {
    x: order.map(o => o.v),
    y: order.map(o => SHAP_CATEGORY_LABELS[o.c] ?? o.c),
    type: "bar", orientation: "h",
    marker: { color: order.map(o => o.v >= 0 ? cssVar("--red") : cssVar("--blue")) },
    hovertemplate: "%{y}: %{x:.2f} EUR/MWh<extra></extra>",
  };
  Plotly.newPlot("chart-shap", [trace], baseLayout({
    hovermode: "closest",
    margin: { l: 140, r: 16, t: 8, b: 40 },
    xaxis: Object.assign(baseLayout({}).xaxis, { title: "Mean signed contribution (EUR/MWh)", zeroline: true, zerolinecolor: cssVar("--axis") }),
  }), plotConfig);

  // Top driver = largest |contribution|; bars are sorted ascending, so it's last.
  const top = order.slice().reverse().slice(0, 2).map(o =>
    `${SHAP_CATEGORY_LABELS[o.c] ?? o.c} (${o.v >= 0 ? "+" : "−"}${Math.abs(o.v).toFixed(1)} €/MWh)`);
  fillBeat("forecast-price", 2, "drivers", top.length ? top.join(" and ") : null);
  Object.assign(sceneState["forecast-price"], {
    shapReady: true, shapCount: order.length, shapTopIdx: order.length - 1,
  });
  tryRegisterForecastPrice();
}

/** Gen/load forecast note. `series` is [{label, unit, fMean, rMean, pct}] where
 *  fMean is tomorrow's forecast mean, rMean the recent 7-day actual mean, pct the
 *  signed % deviation (null when it can't be computed). Pure/testable. */
function forecastGenLoadSentenceFrom(series) {
  const valid = (series || []).filter(s => s && isNum(s.fMean));
  if (!valid.length) return null;
  const parts = [];

  const load = valid.find(s => s.label === "Load");
  if (load) {
    let s = `Tomorrow's load is forecast to average about ${Math.round(load.fMean).toLocaleString("en-US")} MW`;
    if (isNum(load.pct)) s += `, ${describeDeviation(load.pct)} the last 7 days`;
    parts.push(s + ".");
  }

  const ranked = valid.filter(s => isNum(s.pct) && s.label !== "Load")
    .sort((a, b) => Math.abs(b.pct) - Math.abs(a.pct));
  if (ranked.length && Math.abs(ranked[0].pct) >= 5) {
    parts.push(`${ranked[0].label} stands out — forecast ${describeDeviation(ranked[0].pct)} its recent average.`);
  } else if (parts.length && ranked.length) {
    parts.push(`Wind and solar are all close to their recent averages.`);
  }
  return parts.length ? parts.join(" ") : null;
}

/** Price-driver note from price_shap.json. Pure/testable. */
function forecastPriceDriverSentence(shap) {
  if (!shap?.categories?.length || !shap.category_contributions) return null;
  const means = shap.categories
    .map(c => ({ label: SHAP_CATEGORY_LABELS[c] ?? c, v: meanOf(shap.category_contributions[c]) }))
    .filter(m => isNum(m.v));
  if (!means.length) return null;
  means.sort((a, b) => Math.abs(b.v) - Math.abs(a.v));
  const clause = means.slice(0, 3).map(m =>
    `${m.label} (${m.v >= 0 ? "+" : "−"}${Math.abs(m.v).toFixed(1)} €/MWh)`);

  let s;
  if (clause.length === 1) {
    s = `Tomorrow's price forecast leans mostly on ${clause[0]}.`;
  } else {
    const last = clause.pop();
    s = `Tomorrow's price forecast leans most on ${clause.join(", ")}, then ${last}.`;
  }
  return s + " The SHAP attribution explains this specific model's own reasoning, " +
    "not verified real-world market causality.";
}

/** Assemble the gen/load series inputs by fetching forecast + recent-actual JSON. */
async function loadForecastNarrative() {
  const price = await fetchJSON(`${DEPLOY_DATA}/price_forecast.json`);
  const deliveryDate = price?.forecasts?.[0]?.timestamp?.slice(0, 10) ?? null;
  if (deliveryDate) {
    document.querySelectorAll(".delivery-date").forEach(el => { el.textContent = deliveryDate; });
  }

  const actuals = await fetchJSON(`${DEPLOY_DATA}/gen_load_actuals.json`);
  const series = [];
  for (const t of GEN_LOAD_TARGETS) {
    const fc = await fetchJSON(`${DEPLOY_DATA}/gen_load/${t.key}_national.json`);
    if (!fc?.forecasts?.length) continue;
    // Prefer the delivery day; fall back to the first 24 forecast hours if the
    // snapshots don't line up (e.g. stale committed data).
    let day = deliveryDate
      ? fc.forecasts.filter(f => (f.timestamp || "").slice(0, 10) === deliveryDate) : [];
    if (!day.length) day = fc.forecasts.slice(0, 24);
    const fMean = meanOf(day.map(f => f.forecast));
    const rMean = meanOf((actuals?.[t.key]?.days ?? []).slice(-7).flatMap(d => d.values ?? []));
    series.push({
      label: t.label, unit: t.unit, fMean, rMean,
      pct: (isNum(fMean) && isNum(rMean) && rMean !== 0) ? (fMean - rMean) / rMean * 100 : null,
    });
  }
  renderTemplated("narrative-gen-load-forecast", forecastGenLoadSentenceFrom(series));

  const shap = await fetchJSON(`${DEPLOY_DATA}/price_shap.json`);
  renderTemplated("narrative-price-driver", forecastPriceDriverSentence(shap));
}

// ── Chapter 6: track record (closing) ──────────────────────────────────

/** Closing summary from errors_summary.json (rolling daily MAE/RMSE). Pure. */
function trackRecordSentence(errors) {
  const mae = meanOf(errors?.mae);
  if (!isNum(mae)) return null;
  const n = Array.isArray(errors?.dates) ? errors.dates.length : null;
  const rmse = meanOf(errors?.rmse);
  let s = `Over the last ${n ? `${n} days` : "week"}, the day-ahead price forecast was off ` +
    `by about €${Math.round(mae)}/MWh on average`;
  if (isNum(rmse)) s += ` (RMSE €${Math.round(rmse)})`;
  return s + ". Good enough to plan around — which is the whole point.";
}

async function buildTrackRecordChart() {
  const errors = await fetchJSON(`${DEPLOY_DATA}/errors_summary.json`);
  const maeMean = meanOf(errors?.mae);
  const nDays = Array.isArray(errors?.dates) ? errors.dates.length : null;
  fillBeat("track-record", 0, "window", nDays ? `${nDays} days` : null);
  fillBeat("track-record", 0, "mae", isNum(maeMean) ? `about €${Math.round(maeMean)}/MWh` : null);
  renderTemplated("narrative-track-record", trackRecordSentence(errors));

  const hist = await fetchJSON(`${DEPLOY_DATA}/forecast_history.json`);
  if (!hist?.forecasts?.length) return;
  const actuals = await fetchJSON(`${DEPLOY_DATA}/actuals.json`);

  // Concatenate the issued forecasts (each a 24h delivery day) into one track.
  const fx = [], fy = [], flo = [], fhi = [];
  const issues = hist.forecasts.slice().sort((a, b) => (a.issued_at > b.issued_at ? 1 : -1));
  for (const iss of issues) {
    for (const p of (iss.forecasts || [])) {
      fx.push(p.timestamp); fy.push(p.forecast); flo.push(p.forecast_lower); fhi.push(p.forecast_upper);
    }
  }
  // Realised prices for exactly those delivery dates, overlaid.
  const dates = [...new Set(fx.map(t => (t || "").slice(0, 10)))];
  const dayMap = new Map((actuals?.days || []).map(d => [d.date, d.prices]));
  const ax = [], ay = [];
  for (const date of dates) {
    const prices = dayMap.get(date);
    if (!prices) continue;
    prices.forEach((v, h) => { ax.push(`${date}T${String(h).padStart(2, "0")}:00`); ay.push(v); });
  }

  const traces = [];
  if (ax.length) traces.push({ x: ax, y: ay, name: "Realised", type: "scatter", mode: "lines",
    line: { width: 1.5, color: cssVar("--gray-context") } });
  if (fhi.some(v => isNum(v))) {
    traces.push({ x: fx, y: fhi, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
    traces.push({ x: fx, y: flo, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
      fillcolor: hexToRgba(cssVar("--blue"), 0.15), showlegend: false, hoverinfo: "skip" });
  }
  traces.push({ x: fx, y: fy, name: "Forecast (issued day before)", type: "scatter", mode: "lines",
    line: { width: 2, color: cssVar("--blue") } });

  Plotly.newPlot("chart-track-record", traces, baseLayout({
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "EUR/MWh" }),
    showlegend: true, legend: { orientation: "h", y: 1.12, x: 0 },
  }), plotConfig);
}

buildYearlyGenLoadChart();
buildYearlyPriceChart();
loadYearlyNarrative();
buildForecastGenLoadCharts();
buildForecastPriceChart();
buildShapChart();
loadForecastNarrative();
buildTrackRecordChart();
