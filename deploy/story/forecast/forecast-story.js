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

// ── Chapter 2: yearly generation & load ────────────────────────────────

async function buildYearlyGenLoadChart() {
  const facts = await fetchJSON(YEARLY_DATA);
  if (!facts) return;
  const gl = facts.current_year.gen_load;
  const traces = GEN_GROUPS.map(g => ({
    x: gl.date, y: gl.generation[g.key], name: g.label, type: "scatter", mode: "lines",
    stackgroup: "gen", line: { width: 0.5, color: cssVar(g.color) },
    fillcolor: hexToRgba(cssVar(g.color), 0.55),
  }));
  traces.push({
    x: gl.date, y: gl.load, name: "Load", type: "scatter", mode: "lines",
    line: { width: 2, color: cssVar("--orange") },
  });
  Plotly.newPlot("chart-yearly-gen-load", traces, baseLayout({
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "MWh/day" }),
    showlegend: true,
    legend: { orientation: "h", y: 1.12, x: 0 },
  }), plotConfig);
  renderTable("table-chart-yearly-gen-load",
    ["Date", ...GEN_GROUPS.map(g => g.label), "Load"],
    gl.date.map((d, i) => [d, ...GEN_GROUPS.map(g => gl.generation[g.key][i]), gl.load[i]]));
}

// ── Chapter 3: yearly prices ────────────────────────────────────────────

async function buildYearlyPriceChart() {
  const facts = await fetchJSON(YEARLY_DATA);
  if (!facts) return;
  const price = facts.current_year.price;
  const trace = {
    x: price.timestamp, y: price.price, type: "scattergl", mode: "lines",
    line: { width: 1, color: cssVar("--blue") }, name: "DE-LU price",
  };
  Plotly.newPlot("chart-yearly-price", [trace], baseLayout({
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "EUR/MWh", zeroline: true, zerolinecolor: cssVar("--axis") }),
  }), plotConfig);
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
  }

  Plotly.newPlot("chart-forecast-gen-load", traces, layout, plotConfig);
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
  if (lower.some(v => v !== undefined)) {
    traces.push({ x: fx, y: upper, type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" });
    traces.push({ x: fx, y: lower, type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
      fillcolor: hexToRgba(cssVar("--blue"), 0.15), showlegend: false, hoverinfo: "skip" });
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

buildYearlyGenLoadChart();
buildYearlyPriceChart();
loadYearlyNarrative();
buildForecastGenLoadCharts();
buildForecastPriceChart();
buildShapChart();
loadForecastNarrative();
