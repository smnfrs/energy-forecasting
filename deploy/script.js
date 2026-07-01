(function () {
  "use strict";

  // When a live API is available, set window.API_DATA_BASE before loading this script.
  const DATA = (typeof window !== "undefined" && window.API_DATA_BASE) || "data/";
  let currentLang = "en";
  let translations = {};

  // Per-plan §7.3: gen/load card config. Stage 7a renders national only;
  // per-TSO traces are skipped gracefully when files return null.
  const GEN_LOAD_CARDS = [
    {
      id: "card-wind-onshore", target: "wind_onshore",
      label: "Wind Onshore", color: "#2266CC",
      tsos: ["national", "50hz", "amprion", "tennet", "transnetbw"],
      tsoLabels: { national: "National", "50hz": "50Hertz", amprion: "Amprion", tennet: "TenneT", transnetbw: "TransnetBW" },
    },
    {
      id: "card-wind-offshore", target: "wind_offshore",
      label: "Wind Offshore", color: "#44AADD",
      tsos: ["national", "50hz", "tennet"],
      tsoLabels: { national: "National", "50hz": "50Hertz", tennet: "TenneT" },
    },
    {
      id: "card-solar", target: "solar",
      label: "Solar", color: "#DDAA00",
      tsos: ["national", "50hz", "amprion", "tennet", "transnetbw"],
      tsoLabels: { national: "National", "50hz": "50Hertz", amprion: "Amprion", tennet: "TenneT", transnetbw: "TransnetBW" },
    },
    {
      id: "card-load", target: "load",
      label: "Load", color: "#EE0000",
      tsos: ["national", "50hz", "amprion", "tennet", "transnetbw", "creos"],
      tsoLabels: { national: "National", "50hz": "50Hertz", amprion: "Amprion", tennet: "TenneT", transnetbw: "TransnetBW", creos: "Creos" },
    },
  ];

  // ── Utilities ─────────────────────────────────────────────────────────────

  async function fetchJSON(path) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) return null;
      return await resp.json();
    } catch { return null; }
  }

  // Convert Stage 6 history format → EP-compatible [{date, prices}]
  function adaptHistory(history) {
    if (!history || !history.forecasts) return [];
    return history.forecasts
      .filter(e => e.forecasts && e.forecasts.length === 24)
      .map(e => ({
        date: e.forecasts[0].timestamp.slice(0, 10),
        prices: e.forecasts.map(f => f.forecast),
      }));
  }

  function colorWithAlpha(hex, alpha) {
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  function showNoData(el, key) {
    const div = document.createElement("div");
    div.className = "no-data";
    div.textContent = t(key || "no_data");
    el.appendChild(div);
  }

  // ── §1 Price Forecast (24h bar + PI band) ─────────────────────────────────

  function renderPriceChart(forecast) {
    const el = document.getElementById("forecast-chart");
    if (!forecast || !forecast.forecasts || !forecast.forecasts.length) {
      showNoData(el); return;
    }

    const hours = forecast.forecasts.map(f => f.timestamp.slice(11, 16));
    const values = forecast.forecasts.map(f => f.forecast);
    const lower = forecast.forecasts.map(f => f.forecast_lower ?? null);
    const upper = forecast.forecasts.map(f => f.forecast_upper ?? null);
    const hasPI = lower.some(v => v !== null);

    const delivEl = document.getElementById("delivery-date");
    if (delivEl) delivEl.textContent = forecast.forecasts[0].timestamp.slice(0, 10);

    const traces = [];
    if (hasPI) {
      traces.push({
        x: hours, y: lower, type: "scatter", mode: "none",
        fill: null, showlegend: false, hoverinfo: "skip",
      });
      traces.push({
        x: hours, y: upper, type: "scatter", mode: "none",
        fill: "tonexty", fillcolor: "rgba(59,130,246,0.15)",
        name: t("pi_band"), showlegend: true, hoverinfo: "skip",
      });
    }
    traces.push({
      x: hours, y: values, type: "bar",
      name: t("forecast") + " (EUR/MWh)",
      marker: { color: "#3B82F6" },
    });

    Plotly.newPlot(el, traces, {
      xaxis: { title: "" },
      yaxis: { title: "EUR/MWh" },
      legend: { orientation: "h", y: -0.18 },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 300,
    }, { responsive: true, displayModeBar: false });
  }

  // ── §2 Gen/Load National Overview (168h stacked area + load line) ─────────

  async function renderGenLoadSummary() {
    const el = document.getElementById("summary-chart");

    const [wo, woff, sol, load, actuals] = await Promise.all([
      fetchJSON(DATA + "gen_load/wind_onshore_national.json"),
      fetchJSON(DATA + "gen_load/wind_offshore_national.json"),
      fetchJSON(DATA + "gen_load/solar_national.json"),
      fetchJSON(DATA + "gen_load/load_national.json"),
      fetchJSON(DATA + "gen_load_actuals.json"),
    ]);

    if (!wo && !woff && !sol && !load) {
      showNoData(el, "no_gen_load_data"); return;
    }

    const traces = [];
    const genSeries = [
      { data: wo,   name: t("wind_onshore"),  color: "#2266CC", fill: "tozeroy" },
      { data: woff, name: t("wind_offshore"), color: "#44AADD", fill: "tonexty" },
      { data: sol,  name: t("solar"),         color: "#DDAA00", fill: "tonexty" },
    ];

    for (const s of genSeries) {
      if (!s.data) continue;
      traces.push({
        x: s.data.forecasts.map(f => f.timestamp),
        y: s.data.forecasts.map(f => f.forecast),
        type: "scatter", mode: "lines", name: s.name,
        stackgroup: "gen", fill: s.fill,
        fillcolor: colorWithAlpha(s.color, 0.65),
        line: { color: s.color, width: 1 },
      });
    }

    if (load) {
      traces.push({
        x: load.forecasts.map(f => f.timestamp),
        y: load.forecasts.map(f => f.forecast),
        type: "scatter", mode: "lines",
        name: t("load") + " (" + t("forecast") + ")",
        yaxis: "y2",
        line: { color: "#EE0000", width: 2, dash: "dash" },
      });
    }

    // Actuals overlay (Stage 7b — gracefully skipped when file absent)
    if (actuals) {
      const ACTUALS_COLORS = { wind_onshore: "#2266CC", wind_offshore: "#44AADD", solar: "#DDAA00", load: "#EE0000" };
      for (const target of ["wind_onshore", "wind_offshore", "solar"]) {
        const td = actuals[target];
        if (!td || !td.days) continue;
        const xs = [], ys = [];
        for (const day of td.days) {
          for (let h = 0; h < day.values.length; h++) {
            xs.push(`${day.date}T${String(h).padStart(2, "0")}:00:00Z`);
            ys.push(day.values[h]);
          }
        }
        traces.push({
          x: xs, y: ys, type: "scatter", mode: "lines",
          name: t(target) + " (" + t("actual") + ")",
          stackgroup: "gen",
          line: { color: ACTUALS_COLORS[target], width: 1, dash: "dot" },
          opacity: 0.5, showlegend: false,
        });
      }
      if (actuals.load && load) {
        const td = actuals.load;
        const xs = [], ys = [];
        for (const day of (td.days || [])) {
          for (let h = 0; h < day.values.length; h++) {
            xs.push(`${day.date}T${String(h).padStart(2, "0")}:00:00Z`);
            ys.push(day.values[h]);
          }
        }
        if (xs.length) {
          traces.push({
            x: xs, y: ys, type: "scatter", mode: "lines",
            name: t("load") + " (" + t("actual") + ")",
            yaxis: "y2",
            line: { color: "#EE0000", width: 1 },
            opacity: 0.5, showlegend: false,
          });
        }
      }
    }

    Plotly.newPlot(el, traces, {
      xaxis: { type: "date", title: "" },
      yaxis: { title: "MW (" + t("gen_load_section").split("&")[0].trim() + ")", rangemode: "tozero" },
      yaxis2: { title: "MW (" + t("load") + ")", overlaying: "y", side: "right", rangemode: "tozero" },
      legend: { orientation: "h", y: -0.22 },
      margin: { t: 20, r: 80, b: 80, l: 65 },
      height: 350,
    }, { responsive: true, displayModeBar: false });
  }

  // ── §3–6 Individual Gen/Load Cards ────────────────────────────────────────

  function renderGenLoadCard(container, cfg, dataArr) {
    const nationalData = dataArr[0];
    if (!nationalData) {
      showNoData(container, "no_gen_load_data"); return;
    }

    // Build trace list and checkbox controls for available TSOs
    const traces = [];
    const availableTSOs = cfg.tsos
      .map((tso, i) => ({ tso, data: dataArr[i] }))
      .filter(({ data }) => data !== null);

    // Checkbox controls
    const controls = document.createElement("div");
    controls.className = "tso-controls";

    // Track which trace indices belong to each TSO (for restyle)
    const tsoTraceMap = [];

    for (const { tso, data } of availableTSOs) {
      const isNational = tso === "national";
      const forecasts = data.forecasts || [];
      const ts = forecasts.map(f => f.timestamp);
      const ys = forecasts.map(f => f.forecast);
      const lower = forecasts.map(f => f.forecast_lower ?? null);
      const upper = forecasts.map(f => f.forecast_upper ?? null);
      const hasPI = isNational && lower.some(v => v !== null);

      const startIdx = traces.length;
      if (hasPI) {
        traces.push({
          x: ts, y: lower, type: "scatter", mode: "none",
          fill: null, showlegend: false, hoverinfo: "skip", visible: true,
        });
        traces.push({
          x: ts, y: upper, type: "scatter", mode: "none",
          fill: "tonexty", fillcolor: colorWithAlpha(cfg.color, 0.15),
          name: t("pi_band"), showlegend: true, hoverinfo: "skip", visible: true,
        });
      }
      traces.push({
        x: ts, y: ys, type: "scatter", mode: "lines",
        name: cfg.tsoLabels[tso] || tso,
        line: { color: cfg.color, width: isNational ? 2 : 1.5, dash: isNational ? "solid" : "dash" },
        visible: isNational,
      });

      const endIdx = traces.length;
      tsoTraceMap.push({ tso, startIdx, endIdx, hasPI });

      // Checkbox label
      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = isNational;
      cb.dataset.tso = tso;
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + (cfg.tsoLabels[tso] || tso)));
      controls.appendChild(label);
    }

    const chartDiv = document.createElement("div");
    container.appendChild(controls);
    container.appendChild(chartDiv);

    Plotly.newPlot(chartDiv, traces, {
      xaxis: { type: "date", title: "" },
      yaxis: { title: t("unit_mw"), rangemode: "tozero" },
      legend: { orientation: "h", y: -0.22 },
      margin: { t: 20, r: 20, b: 80, l: 65 },
      height: 280,
    }, { responsive: true, displayModeBar: false });

    // Wire checkbox → restyle visibility
    controls.querySelectorAll("input[type=checkbox]").forEach(cb => {
      cb.addEventListener("change", () => {
        const checkedTSOs = new Set(
          [...controls.querySelectorAll("input[type=checkbox]")]
            .filter(c => c.checked)
            .map(c => c.dataset.tso)
        );
        const visibilities = new Array(traces.length).fill(false);
        for (const { tso, startIdx, endIdx, hasPI } of tsoTraceMap) {
          const show = checkedTSOs.has(tso);
          for (let i = startIdx; i < endIdx; i++) {
            // PI helper traces (no legend): show if national is shown
            visibilities[i] = show;
          }
        }
        Plotly.restyle(chartDiv, { visible: visibilities });
      });
    });
  }

  function setupGenLoadCards() {
    for (const cfg of GEN_LOAD_CARDS) {
      const el = document.getElementById(cfg.id);
      if (!el) continue;
      el.addEventListener("toggle", async function onFirst() {
        if (!el.open) return;
        el.removeEventListener("toggle", onFirst);
        const files = cfg.tsos.map(tso => `${DATA}gen_load/${cfg.target}_${tso}.json`);
        const dataArr = await Promise.all(files.map(fetchJSON));
        renderGenLoadCard(el.querySelector(".chart-container"), cfg, dataArr);
      });
    }
  }

  // ── §7 Forecast vs Actual — Price (30 days) ───────────────────────────────

  function renderHistoryChart(actuals, history) {
    const el = document.getElementById("history-chart");
    if (!actuals || !history || !history.length) { showNoData(el); return; }

    // Daily mean actual prices
    const actualsMap = {};
    for (const day of (actuals.days || [])) {
      if (day.prices && day.prices.length) {
        actualsMap[day.date] = day.prices.reduce((a, b) => a + b, 0) / day.prices.length;
      }
    }

    const matched = history
      .filter(e => actualsMap[e.date] !== undefined && e.prices && e.prices.length)
      .sort((a, b) => a.date.localeCompare(b.date));

    if (!matched.length) { showNoData(el); return; }

    const dates = matched.map(e => e.date);
    const actualYs = dates.map(d => Math.round(actualsMap[d] * 100) / 100);
    const forecastYs = matched.map(e => {
      const mean = e.prices.reduce((a, b) => a + b, 0) / e.prices.length;
      return Math.round(mean * 100) / 100;
    });

    Plotly.newPlot(el, [
      {
        x: dates, y: actualYs, type: "scatter", mode: "lines+markers",
        name: t("actual"), line: { color: "#6c757d", width: 2 },
        marker: { size: 4, color: "#6c757d" },
      },
      {
        x: dates, y: forecastYs, type: "scatter", mode: "lines+markers",
        name: t("forecast"), line: { color: "#3B82F6", width: 2, dash: "dash" },
        marker: { size: 4, color: "#3B82F6" },
      },
    ], {
      xaxis: { type: "date", title: "" },
      yaxis: { title: "EUR/MWh" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 280,
    }, { responsive: true, displayModeBar: false });
  }

  // ── §8 Daily Error — Price (last 7 days) ──────────────────────────────────

  function renderErrorChart(actuals, history) {
    const el = document.getElementById("error-chart");
    if (!actuals || !history || !history.length) { showNoData(el); return; }

    const actualsMap = {};
    for (const day of (actuals.days || [])) {
      if (day.prices && day.prices.length === 24) actualsMap[day.date] = day.prices;
    }

    const matched = history
      .filter(e => actualsMap[e.date] && e.prices && e.prices.length === 24)
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-7);

    if (!matched.length) { showNoData(el); return; }

    const dates = matched.map(e => e.date);
    const maes = matched.map(e => {
      const act = actualsMap[e.date];
      const sum = act.reduce((s, a, i) => s + Math.abs(a - e.prices[i]), 0);
      return Math.round((sum / 24) * 100) / 100;
    });
    const rmses = matched.map(e => {
      const act = actualsMap[e.date];
      const mse = act.reduce((s, a, i) => s + Math.pow(a - e.prices[i], 2), 0) / 24;
      return Math.round(Math.sqrt(mse) * 100) / 100;
    });

    Plotly.newPlot(el, [
      { x: dates, y: maes, type: "bar", name: t("mae"), marker: { color: "#3B82F6" } },
      { x: dates, y: rmses, type: "bar", name: t("rmse"), marker: { color: "#ef4444" } },
    ], {
      barmode: "group",
      xaxis: { title: "" },
      yaxis: { title: "EUR/MWh" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 240,
    }, { responsive: true, displayModeBar: false });
  }

  // ── Metadata / header ──────────────────────────────────────────────────────

  function renderMetadata(metadata, forecast) {
    const updEl = document.getElementById("last-updated-value");
    if (updEl) {
      const ts = (metadata && metadata.last_retrain) || (forecast && forecast.issued_at);
      if (ts) updEl.textContent = ts.slice(0, 10);
    }
    if (metadata && metadata.needs_reselection) {
      const el = document.getElementById("reselection-warning");
      if (el) el.hidden = false;
    }
  }

  // ── i18n ──────────────────────────────────────────────────────────────────

  function t(key) {
    return (translations[currentLang] && translations[currentLang][key]) || key;
  }

  function applyTranslations() {
    document.querySelectorAll("[data-i18n]").forEach(el => {
      const text = t(el.getAttribute("data-i18n"));
      if (text !== el.getAttribute("data-i18n")) el.textContent = text;
    });
  }

  function setupLanguageToggle() {
    const btn = document.getElementById("lang-toggle");
    if (!btn) return;
    btn.addEventListener("click", () => {
      currentLang = currentLang === "en" ? "de" : "en";
      btn.textContent = currentLang === "en" ? "DE" : "EN";
      applyTranslations();
    });
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  async function init() {
    translations = await fetchJSON("translations.json") || {};

    const [price, historyRaw, actuals, metadata] = await Promise.all([
      fetchJSON(DATA + "price_forecast.json"),
      fetchJSON(DATA + "forecast_history.json"),
      fetchJSON(DATA + "actuals.json"),
      fetchJSON(DATA + "model_metadata.json"),
    ]);
    const history = adaptHistory(historyRaw);

    renderPriceChart(price);
    renderGenLoadSummary();
    setupGenLoadCards();
    renderHistoryChart(actuals, history);
    renderErrorChart(actuals, history);
    renderMetadata(metadata, price);
    setupLanguageToggle();
    applyTranslations();
  }

  document.addEventListener("DOMContentLoaded", init);

})();
