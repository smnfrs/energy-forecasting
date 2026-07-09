/* Page script for the Stage 10 forecast story. Fetches published JSON (facts,
   narratives, forecasts, SHAP attribution) and builds each section's chart(s). */

const YEARLY_DATA = "data/facts_yearly.json";
const YEARLY_NARRATIVE = "data/narrative_yearly.json";
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

  const mix = gl.fuel_mix_pct;
  document.getElementById("stat-fuel-mix").textContent =
    Object.entries(mix).map(([k, v]) => `${GEN_GROUPS.find(g => g.key === k)?.label ?? k}: ${v}%`).join(" · ");
  document.getElementById("stat-imports-exports").textContent =
    `Imports ${gl.imports_pct_of_domestic_gen}% · Exports ${gl.exports_pct_of_domestic_gen}% of domestic generation`;
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

  document.getElementById("stat-price-mean").textContent = `Mean: ${price.mean_price} EUR/MWh`;
  document.getElementById("stat-price-extremes").textContent =
    `Most expensive hour on average: ${price.most_expensive_hour}:00 (${price.most_expensive_hour_avg_price} EUR/MWh) · ` +
    `Least expensive: ${price.least_expensive_hour}:00 (${price.least_expensive_hour_avg_price} EUR/MWh)`;
  document.getElementById("stat-negative-hours").textContent =
    `${price.negative_price_hours} negative-price hours in the trailing year`;
}

async function loadYearlyNarrative() {
  const n = await fetchJSON(YEARLY_NARRATIVE);
  renderNarrative("narrative-gen-load-yearly", n?.gen_load_yearly_summary, n?.status);
  renderNarrative("narrative-price-yearly", n?.price_yearly_summary, n?.status);
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

async function loadForecastNarrative() {
  const n = await fetchJSON(`${DEPLOY_DATA}/narrative_forecast.json`);
  renderNarrative("narrative-gen-load-forecast", n?.gen_load_forecast_note, n?.status);
  renderNarrative("narrative-price-driver", n?.price_driver_explanation, n?.status);
  if (n?.delivery_date) {
    document.querySelectorAll(".delivery-date").forEach(el => { el.textContent = n.delivery_date; });
  }
}

buildYearlyGenLoadChart();
buildYearlyPriceChart();
loadYearlyNarrative();
buildForecastGenLoadCharts();
buildForecastPriceChart();
buildShapChart();
loadForecastNarrative();
