/* Shared Plotly chart builders for the Stage 9 story site.
   Used by both index.html (the real page) and chart_prototypes.html (QA harness).
   Reads data from a <script id="story-data" type="application/json"> element on
   the including page — not fetch() — so the page works from a plain file:// open. */

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

const font = { family: "system-ui, -apple-system, 'Segoe UI', sans-serif", color: "var(--text-secondary)" };

function baseLayout(overrides) {
  return Object.assign({
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    font: { family: font.family, color: cssVar("--text-secondary"), size: 12 },
    margin: { l: 56, r: 16, t: 8, b: 40 },
    hovermode: "x unified",
    hoverlabel: { bgcolor: cssVar("--surface-1"), bordercolor: cssVar("--border"), font: { color: cssVar("--text-primary") } },
    xaxis: { gridcolor: "transparent", linecolor: cssVar("--axis"), tickcolor: cssVar("--axis"), zeroline: false },
    yaxis: { gridcolor: cssVar("--grid"), linecolor: cssVar("--axis"), tickcolor: cssVar("--axis"), zeroline: false, gridwidth: 1 },
    showlegend: false,
  }, overrides);
}

const plotConfig = { displayModeBar: false, responsive: true };

function renderTable(containerId, headers, rows) {
  const el = document.getElementById(containerId);
  if (!el) return;
  let html = "<table><thead><tr>" + headers.map(h => `<th>${h}</th>`).join("") + "</tr></thead><tbody>";
  for (const row of rows) {
    html += "<tr>" + row.map(v => `<td>${v === null || v === undefined ? "—" : v}</td>`).join("") + "</tr>";
  }
  html += "</tbody></table>";
  el.textContent = "";
  const template = document.createElement("template");
  template.innerHTML = html;
  el.appendChild(template.content);
}

document.addEventListener("click", (e) => {
  const btn = e.target.closest(".table-toggle");
  if (!btn) return;
  const id = "table-" + btn.dataset.tableFor;
  document.getElementById(id).classList.toggle("open");
});

const STORY_DATA = JSON.parse(
  document.getElementById("story-data").textContent
    .replace(/<!--\s*STORY_DATA_(START|END)\s*-->/g, "")
    .trim()
);

async function loadJSON(name) {
  return STORY_DATA[name];
}

function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

// --- Chapter 2: price history ---
async function buildPriceChart() {
  const d = await loadJSON("price_history.json");
  const blue = cssVar("--blue");
  const traces = [
    { x: d.month, y: d.max, name: "range", type: "scatter", mode: "lines", line: { width: 0 }, showlegend: false, hoverinfo: "skip" },
    { x: d.month, y: d.min, name: "range", type: "scatter", mode: "lines", line: { width: 0 }, fill: "tonexty",
      fillcolor: hexToRgba(blue, 0.10), showlegend: false, hoverinfo: "skip" },
    { x: d.month, y: d.mean, name: "Monthly mean price", type: "scatter", mode: "lines",
      line: { width: 2, color: blue, shape: "spline", smoothing: 0.3 } },
  ];
  Plotly.newPlot("chart-price", traces, baseLayout({
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "EUR/MWh" }),
  }), plotConfig);
  renderTable("table-chart-price", ["Month", "Mean", "Min", "Max"],
    d.month.map((m, i) => [m, d.mean[i], d.min[i], d.max[i]]));
}

// --- Chapter 3: gas shock, small multiples (stacked panels, shared x) ---
async function buildGasChart() {
  const d = await loadJSON("gas_shock.json");
  const orange = cssVar("--orange");
  const blue = cssVar("--blue");
  const traces = [
    { x: d.month, y: d.gas_ttf_eur_mwh, name: "Gas (TTF)", type: "scatter", mode: "lines",
      line: { width: 2, color: orange }, xaxis: "x", yaxis: "y" },
    { x: d.month, y: d.power_price_eur_mwh, name: "DE-LU power price", type: "scatter", mode: "lines",
      line: { width: 2, color: blue }, xaxis: "x2", yaxis: "y2" },
  ];
  const layout = baseLayout({
    grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
    yaxis:  { title: "Gas EUR/MWh", gridcolor: cssVar("--grid"), linecolor: cssVar("--axis"), zeroline: false },
    yaxis2: { title: "Power EUR/MWh", gridcolor: cssVar("--grid"), linecolor: cssVar("--axis"), zeroline: false },
    xaxis:  { matches: "x2", showticklabels: false, linecolor: cssVar("--axis"), gridcolor: "transparent" },
    xaxis2: { linecolor: cssVar("--axis"), gridcolor: "transparent" },
    margin: { l: 64, r: 16, t: 8, b: 40 },
  });
  Plotly.newPlot("chart-gas", traces, layout, plotConfig);
  renderTable("table-chart-gas", ["Month", "Gas EUR/MWh", "Power EUR/MWh"],
    d.month.map((m, i) => [m, d.gas_ttf_eur_mwh[i], d.power_price_eur_mwh[i]]));
}

// --- Chapter 4: bidding zones, emphasis small multiples ---
async function buildZonesChart() {
  const d = await loadJSON("bidding_zones.json");
  const blue = cssVar("--blue");
  const gray = cssVar("--gray-context");
  const countries = [
    { code: "AT", name: "Austria" },
    { code: "FR", name: "France" },
    { code: "NL", name: "Netherlands" },
    { code: "PL", name: "Poland" },
  ];
  const traces = [];
  const layout = baseLayout({
    grid: { rows: 2, columns: 2, pattern: "independent" },
    margin: { l: 56, r: 16, t: 24, b: 32 },
    annotations: [],
  });
  countries.forEach((c, i) => {
    const xa = i === 0 ? "x" : `x${i + 1}`;
    const ya = i === 0 ? "y" : `y${i + 1}`;
    traces.push({ x: d.month, y: d.DE_LU, name: "DE-LU", legendgroup: "de", showlegend: i === 0,
      type: "scatter", mode: "lines", line: { width: 2, color: blue }, xaxis: xa, yaxis: ya });
    traces.push({ x: d.month, y: d[c.code], name: "Neighbour", legendgroup: "nb", showlegend: i === 0,
      type: "scatter", mode: "lines", line: { width: 2, color: gray }, xaxis: xa, yaxis: ya });
    layout[xa === "x" ? "xaxis" : xa.replace("x", "xaxis")] = { linecolor: cssVar("--axis"), gridcolor: "transparent" };
    layout[ya === "y" ? "yaxis" : ya.replace("y", "yaxis")] = { title: c.name, linecolor: cssVar("--axis"), gridcolor: cssVar("--grid") };
  });
  layout.showlegend = true;
  layout.legend = { orientation: "h", y: 1.08, x: 0 };
  Plotly.newPlot("chart-zones", traces, layout, plotConfig);
  renderTable("table-chart-zones", ["Month", "DE-LU", "AT", "FR", "NL", "PL"],
    d.month.map((m, i) => [m, d.DE_LU[i], d.AT[i], d.FR[i], d.NL[i], d.PL[i]]));
}

// --- Chapter 5: generation mix (stacked composition) + renewable share ---
const GEN_GROUPS = [
  { key: "wind", label: "Wind", color: "--blue" },
  { key: "solar", label: "Solar", color: "--aqua" },
  { key: "hydro_biomass", label: "Hydro & biomass", color: "--yellow" },
  { key: "nuclear", label: "Nuclear", color: "--green" },
  { key: "gas", label: "Natural gas", color: "--violet" },
  { key: "coal", label: "Coal", color: "--red" },
  { key: "other", label: "Other", color: "--magenta" },
];

async function buildMixChart() {
  const d = await loadJSON("generation_mix.json");
  const traces = GEN_GROUPS.map(g => ({
    x: d.month, y: d[g.key], name: g.label, type: "scatter", mode: "lines",
    stackgroup: "gen", line: { width: 0.5, color: cssVar(g.color) },
    fillcolor: hexToRgba(cssVar(g.color), 0.55), xaxis: "x", yaxis: "y",
  }));
  traces.push({
    x: d.month, y: d.renewable_share_pct, name: "Renewable share", type: "scatter", mode: "lines",
    line: { width: 2, color: cssVar("--blue") }, xaxis: "x2", yaxis: "y2", showlegend: false,
  });
  const layout = baseLayout({
    grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
    yaxis:  { title: "MW (avg)", gridcolor: cssVar("--grid"), linecolor: cssVar("--axis"), zeroline: false },
    yaxis2: { title: "Renewable %", gridcolor: cssVar("--grid"), linecolor: cssVar("--axis"), zeroline: false, range: [0, 100] },
    xaxis:  { matches: "x2", showticklabels: false, linecolor: cssVar("--axis"), gridcolor: "transparent" },
    xaxis2: { linecolor: cssVar("--axis"), gridcolor: "transparent" },
    margin: { l: 64, r: 16, t: 8, b: 40 },
    showlegend: true,
    legend: { orientation: "h", y: 1.1, x: 0 },
  });
  Plotly.newPlot("chart-mix", traces, layout, plotConfig);
  renderTable("table-chart-mix",
    ["Month", ...GEN_GROUPS.map(g => g.label), "Renewable %"],
    d.month.map((m, i) => [m, ...GEN_GROUPS.map(g => d[g.key][i]), d.renewable_share_pct[i]]));
}

// --- Chapter 6a: negative price hours per year ---
async function buildNegYearsChart() {
  const d = await loadJSON("negative_prices.json");
  const blue = cssVar("--blue");
  const trace = {
    x: d.year, y: d.hours, type: "bar", marker: { color: blue },
    hovertemplate: "%{y} hours<extra></extra>",
  };
  Plotly.newPlot("chart-neg-years", [trace], baseLayout({
    hovermode: "closest",
    bargap: 0.3,
    yaxis: Object.assign(baseLayout({}).yaxis, { title: "Negative-price hours / year" }),
    xaxis: Object.assign(baseLayout({}).xaxis, { dtick: 1 }),
  }), plotConfig);
}

// --- Chapter 6b: example duck-curve day ---
async function buildNegDayChart() {
  const d = await loadJSON("negative_price_example_day.json");
  const blue = cssVar("--blue");
  const aqua = cssVar("--aqua");
  const yellow = cssVar("--yellow");
  const red = cssVar("--red");
  const genTraces = [
    { x: d.hour, y: d.solar, name: "Solar", type: "scatter", mode: "lines", stackgroup: "gen",
      line: { width: 0.5, color: yellow }, fillcolor: hexToRgba(yellow, 0.55), xaxis: "x", yaxis: "y" },
    { x: d.hour, y: d.wind_onshore, name: "Wind onshore", type: "scatter", mode: "lines", stackgroup: "gen",
      line: { width: 0.5, color: blue }, fillcolor: hexToRgba(blue, 0.55), xaxis: "x", yaxis: "y" },
    { x: d.hour, y: d.wind_offshore, name: "Wind offshore", type: "scatter", mode: "lines", stackgroup: "gen",
      line: { width: 0.5, color: aqua }, fillcolor: hexToRgba(aqua, 0.55), xaxis: "x", yaxis: "y" },
  ];
  const priceColors = d.price.map(v => v < 0 ? red : blue);
  const priceTrace = {
    x: d.hour, y: d.price, type: "bar", marker: { color: priceColors }, xaxis: "x2", yaxis: "y2",
    showlegend: false, hovertemplate: "EUR %{y}/MWh<extra></extra>",
  };
  const layout = baseLayout({
    grid: { rows: 2, columns: 1, pattern: "independent", roworder: "top to bottom" },
    yaxis:  { title: "Generation (MW)", gridcolor: cssVar("--grid"), linecolor: cssVar("--axis"), zeroline: false },
    yaxis2: { title: "Price (EUR/MWh)", gridcolor: cssVar("--grid"), linecolor: cssVar("--axis"), zeroline: true, zerolinecolor: cssVar("--axis") },
    xaxis:  { matches: "x2", showticklabels: false, linecolor: cssVar("--axis"), gridcolor: "transparent" },
    xaxis2: { title: "Hour (11 May 2025)", linecolor: cssVar("--axis"), dtick: 2, gridcolor: "transparent" },
    margin: { l: 64, r: 16, t: 8, b: 40 },
    showlegend: true,
    legend: { orientation: "h", y: 1.12, x: 0 },
    hovermode: "x",
  });
  Plotly.newPlot("chart-neg-day", [...genTraces, priceTrace], layout, plotConfig);
  renderTable("table-chart-neg-day", ["Hour", "Solar", "Wind onshore", "Wind offshore", "Price EUR/MWh"],
    d.hour.map((h, i) => [h, d.solar[i], d.wind_onshore[i], d.wind_offshore[i], d.price[i]]));
}
