(function () {
  "use strict";

  const DATA = (typeof window !== "undefined" && window.API_DATA_BASE) || "data/";
  let currentLang = "en";
  let translations = {};

  // Tracks which tab panes have been rendered so each renders at most once.
  const _rendered = new Set();
  let _allData = {};

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

  // Per-TSO colors for sub-TSO lines (national always uses cfg.color)
  const TSO_COLORS = {
    "50hz":       "#e67e22",
    "amprion":    "#27ae60",
    "tennet":     "#8e44ad",
    "transnetbw": "#c0392b",
    "creos":      "#16a085",
  };

  const CATEGORY_COLORS = {
    lgbm: "#22c55e",
    xgboost: "#3B82F6",
    catboost: "#f59e0b",
    linear: "#8b5cf6",
  };

  const GL_COLORS = {
    wind_onshore: "#2266CC",
    wind_offshore: "#44AADD",
    solar: "#DDAA00",
    load: "#EE0000",
  };

  // ── Utilities ─────────────────────────────────────────────────────────────

  async function fetchJSON(path) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) return null;
      return await resp.json();
    } catch { return null; }
  }

  // Parse forecast_history.json into an array of daily entries, deduped by delivery date.
  // Where two entries have the same delivery date, production beats backtest; then later
  // issued_at wins. This fixes the case where an afternoon run and the next morning's run
  // both forecast the same delivery day.
  function adaptHistory(history) {
    if (!history || !history.forecasts) return [];
    const all = history.forecasts
      .filter(e => e.forecasts && e.forecasts.length >= 24)
      .map(e => ({
        date: e.forecasts[0].timestamp.slice(0, 10),
        source: e.source || "production",
        issued_at: e.issued_at || "",
        prices: e.forecasts.slice(0, 24).map(f => f.forecast),
        lower: e.forecasts.slice(0, 24).map(f => f.forecast_lower ?? null),
        upper: e.forecasts.slice(0, 24).map(f => f.forecast_upper ?? null),
        timestamps: e.forecasts.slice(0, 24).map(f => f.timestamp),
      }));

    const byDate = new Map();
    for (const e of all) {
      const ex = byDate.get(e.date);
      if (!ex) {
        byDate.set(e.date, e);
      } else {
        const newBetter =
          (e.source === "production" && ex.source !== "production") ||
          (e.source === ex.source && e.issued_at > ex.issued_at);
        if (newBetter) byDate.set(e.date, e);
      }
    }
    return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date));
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

  function noDataInline(el, key) {
    if (el) el.innerHTML = `<p class="no-data">${t(key || "no_data")}</p>`;
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

  // ── Tab switching ─────────────────────────────────────────────────────────

  function activateTab(tabId) {
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.classList.toggle("active", btn.dataset.tab === tabId);
    });
    document.querySelectorAll(".tab-pane").forEach(pane => {
      pane.hidden = pane.id !== "tab-" + tabId;
    });
    if (!_rendered.has(tabId)) {
      _rendered.add(tabId);
      if (tabId === "gen-load") renderGenLoadTab();
      if (tabId === "monitoring") renderMonitoringTab();
    }
  }

  function setupTabs() {
    document.querySelectorAll(".tab-btn").forEach(btn => {
      btn.addEventListener("click", () => activateTab(btn.dataset.tab));
    });
    const hash = location.hash.replace("#", "");
    if (hash === "monitoring") activateTab("monitoring");
  }

  // ── §1 Price Forecast (24h bar + PI as error bars) ────────────────────────

  function renderPriceChart(forecast) {
    const el = document.getElementById("forecast-chart");
    if (!forecast || !forecast.forecasts || !forecast.forecasts.length) {
      showNoData(el); return;
    }

    const hours  = forecast.forecasts.map(f => f.timestamp.slice(11, 16));
    const values = forecast.forecasts.map(f => f.forecast);
    const lower  = forecast.forecasts.map(f => f.forecast_lower ?? null);
    const upper  = forecast.forecasts.map(f => f.forecast_upper ?? null);
    const hasPI  = lower.some(v => v !== null);

    const delivEl = document.getElementById("delivery-date");
    if (delivEl) delivEl.textContent = forecast.forecasts[0].timestamp.slice(0, 10);

    const barTrace = {
      x: hours, y: values, type: "bar",
      name: t("forecast") + " (EUR/MWh)",
      marker: { color: "#3B82F6" },
    };

    if (hasPI) {
      barTrace.error_y = {
        type: "data",
        symmetric: false,
        array:      values.map((v, i) => upper[i] !== null ? Math.max(0, upper[i] - v) : 0),
        arrayminus: values.map((v, i) => lower[i] !== null ? Math.max(0, v - lower[i]) : 0),
        color: "rgba(0,0,0,0.45)",
        thickness: 1.5,
        width: 4,
      };
      barTrace.name = `${t("forecast")} (EUR/MWh) ± ${t("pi_band")}`;
    }

    Plotly.newPlot(el, [barTrace], {
      xaxis: { title: "" },
      yaxis: { title: "EUR/MWh" },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 300,
    }, { responsive: true, displayModeBar: false });
  }

  // ── §2 Forecast vs Actual — Price (7 days, hourly, with 90% CI) ───────────

  function renderHistoryChart(actuals, history) {
    const el = document.getElementById("history-chart");
    if (!actuals || !history || !history.length) { showNoData(el); return; }

    // Hourly actuals map: date → [24 values]
    const actualsMap = {};
    for (const day of (actuals.days || [])) {
      if (day.prices && day.prices.length === 24) actualsMap[day.date] = day.prices;
    }

    // Last 7 days where actuals are available
    const matched = history
      .filter(e => actualsMap[e.date] && e.prices && e.prices.length === 24)
      .sort((a, b) => a.date.localeCompare(b.date))
      .slice(-7);

    if (!matched.length) { showNoData(el); return; }

    // Flatten to hourly arrays
    const xs = [], actualYs = [], forecastYs = [], lowerYs = [], upperYs = [];
    for (const e of matched) {
      const act = actualsMap[e.date];
      for (let h = 0; h < 24; h++) {
        // Normalise timestamp: drop tz suffix, ensure T separator
        const raw = e.timestamps[h] || `${e.date}T${String(h).padStart(2, "0")}:00:00`;
        xs.push(raw.replace(/([+-]\d{2}:\d{2}|Z)$/, "").replace(" ", "T"));
        actualYs.push(act[h]);
        forecastYs.push(e.prices[h]);
        // Collapse null CI bounds to forecast value → zero-width band for backtest entries
        lowerYs.push(e.lower[h] !== null ? e.lower[h] : e.prices[h]);
        upperYs.push(e.upper[h] !== null ? e.upper[h] : e.prices[h]);
      }
    }

    const hasCI = matched.some(e => e.lower.some(v => v !== null));

    const traces = [];
    if (hasCI) {
      // CI band: lower bound trace then fill-to-next upper
      traces.push({
        x: xs, y: lowerYs, type: "scatter", mode: "none",
        showlegend: false, hoverinfo: "skip",
      });
      traces.push({
        x: xs, y: upperYs, type: "scatter", mode: "none",
        fill: "tonexty", fillcolor: "rgba(150,150,150,0.18)",
        name: `${t("pi_band")} (90%)`, hoverinfo: "skip",
      });
    }

    traces.push({
      x: xs, y: actualYs, type: "scatter", mode: "lines",
      name: t("actual"),
      line: { color: "#6c757d", width: 2 },
    });
    traces.push({
      x: xs, y: forecastYs, type: "scatter", mode: "lines",
      name: t("forecast"),
      line: { color: "#3B82F6", width: 1.5, dash: "dash" },
    });

    Plotly.newPlot(el, traces, {
      xaxis: { type: "date", title: "", nticks: 7 },
      yaxis: { title: "EUR/MWh" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 300,
    }, { responsive: true, displayModeBar: false });
  }

  // ── §3 Daily Error — Price (last 7 days) ──────────────────────────────────

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
    const maes  = matched.map(e => {
      const act = actualsMap[e.date];
      return Math.round(act.reduce((s, a, i) => s + Math.abs(a - e.prices[i]), 0) / 24 * 100) / 100;
    });
    const rmses = matched.map(e => {
      const act = actualsMap[e.date];
      const mse = act.reduce((s, a, i) => s + Math.pow(a - e.prices[i], 2), 0) / 24;
      return Math.round(Math.sqrt(mse) * 100) / 100;
    });

    Plotly.newPlot(el, [
      { x: dates, y: maes,  type: "bar", name: t("mae"),  marker: { color: "#3B82F6" } },
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

  // ── Gen & Load tab ────────────────────────────────────────────────────────

  function renderGenLoadTab() {
    renderGenLoadSummary(_allData.genLoad, _allData.glActuals);
    setupGenLoadCards(_allData.glActuals);
  }

  // Build hourly actuals flat arrays for a given target from glActuals
  function buildActualSeries(glActuals, target) {
    if (!glActuals || !glActuals[target]) return { xs: [], ys: [] };
    const xs = [], ys = [];
    for (const day of (glActuals[target].days || [])) {
      for (let h = 0; h < day.values.length; h++) {
        xs.push(`${day.date}T${String(h).padStart(2, "0")}:00:00`);
        ys.push(day.values[h]);
      }
    }
    return { xs, ys };
  }

  function renderGenLoadSummary(genLoad, glActuals) {
    const el = document.getElementById("summary-chart");
    const { wo, woff, sol, load, gld } = genLoad || {};

    if (!wo && !woff && !sol && !load) {
      showNoData(el, "no_gen_load_data"); return;
    }

    const traces = [];

    // ── Actual generation stacked area (solid fill) ──
    // Each target's actual values are stacked on top of the previous.
    const actualGenSeries = [
      { target: "wind_onshore",  name: `${t("wind_onshore")} (${t("actual")})`,  color: "#2266CC" },
      { target: "wind_offshore", name: `${t("wind_offshore")} (${t("actual")})`, color: "#44AADD" },
      { target: "solar",         name: `${t("solar")} (${t("actual")})`,         color: "#DDAA00" },
    ];
    for (const [i, s] of actualGenSeries.entries()) {
      const { xs, ys } = buildActualSeries(glActuals, s.target);
      if (!xs.length) continue;
      traces.push({
        x: xs, y: ys, type: "scatter", mode: "lines",
        name: s.name,
        stackgroup: "gen_actual",
        fill: i === 0 ? "tozeroy" : "tonexty",
        fillcolor: colorWithAlpha(s.color, 0.7),
        line: { color: s.color, width: 1 },
      });
    }

    // ── Forecast generation: cumulative boundary dotted lines ──
    // Manually accumulate so lines trace the stacked tops without fill.
    const fcSeries = [
      { data: wo,   name: `${t("wind_onshore")} (${t("forecast")})`,  color: "#2266CC" },
      { data: woff, name: `${t("wind_offshore")} (${t("forecast")})`, color: "#44AADD" },
      { data: sol,  name: `${t("solar")} (${t("forecast")})`,         color: "#DDAA00" },
    ].filter(s => s.data);

    if (fcSeries.length) {
      // Build timestamp → value maps for each component
      const maps = fcSeries.map(s => new Map(s.data.forecasts.map(f => [f.timestamp, f.forecast])));
      const timestamps = fcSeries[0].data.forecasts.map(f => f.timestamp);

      let cumYs = new Array(timestamps.length).fill(0);
      for (let i = 0; i < fcSeries.length; i++) {
        cumYs = cumYs.map((acc, j) => acc + (maps[i].get(timestamps[j]) ?? 0));
        traces.push({
          x: timestamps, y: [...cumYs],
          type: "scatter", mode: "lines",
          name: fcSeries[i].name,
          line: { color: fcSeries[i].color, width: 1.5, dash: "dot" },
          showlegend: true,
        });
      }
    }

    // ── Load actual (solid red line, y2) ──
    if (glActuals && glActuals.load) {
      const { xs, ys } = buildActualSeries(glActuals, "load");
      if (xs.length) {
        traces.push({
          x: xs, y: ys, type: "scatter", mode: "lines",
          name: `${t("load")} (${t("actual")})`,
          yaxis: "y2",
          line: { color: "#EE0000", width: 2.5 },
        });
      }
    }

    // ── Load forecast (dashed red line, y2) ──
    if (load) {
      traces.push({
        x: load.forecasts.map(f => f.timestamp),
        y: load.forecasts.map(f => f.forecast),
        type: "scatter", mode: "lines",
        name: `${t("load")} (${t("forecast")})`,
        yaxis: "y2",
        line: { color: "#EE0000", width: 1.5, dash: "dash" },
      });
    }

    Plotly.newPlot(el, traces, {
      xaxis: { type: "date", title: "" },
      yaxis: { title: "MW (Generation)", rangemode: "tozero" },
      yaxis2: { title: `MW (${t("load")})`, overlaying: "y", side: "right", rangemode: "tozero" },
      legend: { orientation: "h", y: -0.28 },
      margin: { t: 20, r: 80, b: 100, l: 65 },
      height: 380,
    }, { responsive: true, displayModeBar: false });
  }

  function renderGenLoadCard(container, cfg, dataArr, glActuals) {
    const nationalData = dataArr[0];
    if (!nationalData) {
      showNoData(container, "no_gen_load_data"); return;
    }

    const traces = [];
    const availableTSOs = cfg.tsos
      .map((tso, i) => ({ tso, data: dataArr[i] }))
      .filter(({ data }) => data !== null);

    const controls = document.createElement("div");
    controls.className = "tso-controls";

    const tsoTraceMap = [];

    for (const { tso, data } of availableTSOs) {
      const isNational = tso === "national";
      const tsoColor = isNational ? cfg.color : (TSO_COLORS[tso] || "#888888");
      const forecasts = data.forecasts || [];
      const ts    = forecasts.map(f => f.timestamp);
      const ys    = forecasts.map(f => f.forecast);
      const lower = forecasts.map(f => f.forecast_lower ?? null);
      const upper = forecasts.map(f => f.forecast_upper ?? null);
      const hasPI = isNational && lower.some(v => v !== null);

      const startIdx = traces.length;
      if (hasPI) {
        traces.push({
          x: ts, y: lower, type: "scatter", mode: "none",
          fill: "none", showlegend: false, hoverinfo: "skip", visible: true,
        });
        traces.push({
          x: ts, y: upper, type: "scatter", mode: "none",
          fill: "tonexty", fillcolor: "rgba(130,130,130,0.18)",
          name: t("pi_band"), showlegend: true, hoverinfo: "skip", visible: true,
        });
      }
      traces.push({
        x: ts, y: ys, type: "scatter", mode: "lines",
        name: cfg.tsoLabels[tso] || tso,
        line: { color: tsoColor, width: isNational ? 2 : 1.5, dash: "dash" },
        visible: isNational,
      });

      tsoTraceMap.push({ tso, startIdx, endIdx: traces.length, hasPI });

      const label = document.createElement("label");
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = isNational;
      cb.dataset.tso = tso;
      label.appendChild(cb);
      label.appendChild(document.createTextNode(" " + (cfg.tsoLabels[tso] || tso)));
      controls.appendChild(label);
    }

    // Actuals overlay: solid line (national aggregate)
    const actualsStartIdx = traces.length;
    if (glActuals && glActuals[cfg.target]) {
      const { xs, ys } = buildActualSeries(glActuals, cfg.target);
      if (xs.length) {
        traces.push({
          x: xs, y: ys, type: "scatter", mode: "lines",
          name: `${cfg.tsoLabels["national"] || "National"} (${t("actual")})`,
          line: { color: cfg.color, width: 2.5 },
          opacity: 0.9,
        });
      }
    }
    const hasActualsTrace = traces.length > actualsStartIdx;

    // Determine x-axis window: last 3 days of actuals + forecast extent
    let xRange = null;
    if (glActuals && glActuals[cfg.target] && glActuals[cfg.target].days) {
      const days = glActuals[cfg.target].days;
      if (days.length >= 2) {
        // Start 3 days before the last actuals day
        const lastActualDay = days[days.length - 1].date;
        const startDay = new Date(lastActualDay);
        startDay.setDate(startDay.getDate() - 3);
        xRange = [startDay.toISOString().slice(0, 10), null]; // null = auto end
      }
    }

    const chartDiv = document.createElement("div");
    container.appendChild(controls);
    container.appendChild(chartDiv);

    const layout = {
      xaxis: { type: "date", title: "" },
      yaxis: { title: t("unit_mw"), rangemode: "tozero" },
      legend: { orientation: "h", y: -0.22 },
      margin: { t: 20, r: 20, b: 80, l: 65 },
      height: 300,
    };
    if (xRange) layout.xaxis.range = xRange;

    Plotly.newPlot(chartDiv, traces, layout, { responsive: true, displayModeBar: false });

    controls.querySelectorAll("input[type=checkbox]").forEach(cb => {
      cb.addEventListener("change", () => {
        const checkedTSOs = new Set(
          [...controls.querySelectorAll("input[type=checkbox]")]
            .filter(c => c.checked)
            .map(c => c.dataset.tso)
        );
        const vis = new Array(traces.length).fill(false);
        for (const { tso, startIdx, endIdx } of tsoTraceMap) {
          const show = checkedTSOs.has(tso);
          for (let i = startIdx; i < endIdx; i++) vis[i] = show;
        }
        if (hasActualsTrace) vis[actualsStartIdx] = true;
        Plotly.restyle(chartDiv, { visible: vis });
      });
    });
  }

  function setupGenLoadCards(glActuals) {
    for (const cfg of GEN_LOAD_CARDS) {
      const el = document.getElementById(cfg.id);
      if (!el) continue;
      el.addEventListener("toggle", async function onFirst() {
        if (!el.open) return;
        el.removeEventListener("toggle", onFirst);
        const files = cfg.tsos.map(tso => `${DATA}gen_load/${cfg.target}_${tso}.json`);
        const dataArr = await Promise.all(files.map(fetchJSON));
        renderGenLoadCard(el.querySelector(".chart-container"), cfg, dataArr, glActuals);
      });
    }
  }

  // ── Monitoring tab ────────────────────────────────────────────────────────

  function renderMonitoringTab() {
    renderModelMaeChart(_allData.metadata);
    renderErrorTrend(_allData.summary);
    renderHourlyErrorProfile(_allData.actuals, _allData.history);
    renderCompositionChart(_allData.metadata);
    renderGenLoadErrors(_allData.glErrors);
    renderRetrainLog(_allData.retrainHistory);
  }

  // Per-model CV MAE bar chart (replaces ensemble-only trend as first chart)
  function renderModelMaeChart(metadata) {
    const el = document.getElementById("model-mae-chart");
    if (!el) return;
    if (!metadata || !metadata.models || !metadata.models.length) {
      noDataInline(el); return;
    }

    const models = [...metadata.models]
      .filter(m => m.cv_mae > 0)
      .sort((a, b) => a.cv_mae - b.cv_mae);

    if (!models.length) { noDataInline(el); return; }

    // Short display names
    const shortName = name => name.replace(/__fs_.*$/, "").replace("Regressor", "").replace("Classifier", "");

    Plotly.newPlot(el, [{
      x: models.map(m => m.cv_mae),
      y: models.map(m => shortName(m.name)),
      type: "bar",
      orientation: "h",
      marker: { color: models.map(m => CATEGORY_COLORS[m.category] || "#6c757d") },
      text: models.map(m => `${m.cv_mae.toFixed(1)}`),
      textposition: "outside",
      name: "CV MAE",
    }], {
      xaxis: { title: "CV MAE (EUR/MWh)", zeroline: true },
      yaxis: { automargin: true },
      margin: { t: 10, r: 80, b: 50, l: 200 },
      height: Math.max(180, models.length * 45 + 60),
      shapes: [{
        type: "line",
        x0: metadata.holdout_mae, x1: metadata.holdout_mae,
        y0: -0.5, y1: models.length - 0.5,
        line: { color: "#555", width: 1.5, dash: "dash" },
      }],
      annotations: [{
        x: metadata.holdout_mae, y: models.length - 0.5,
        text: `Ensemble holdout: ${metadata.holdout_mae?.toFixed(2)}`,
        showarrow: false, xanchor: "left", font: { size: 11 },
      }],
    }, { responsive: true, displayModeBar: false });
  }

  function renderErrorTrend(summary) {
    const el = document.getElementById("error-trend-chart");
    if (!summary || !summary.dates || !summary.dates.length) {
      noDataInline(el); return;
    }
    Plotly.newPlot(el, [
      {
        x: summary.dates, y: summary.mae,
        type: "scatter", mode: "lines+markers",
        name: `${t("mae")} (ensemble)`,
        line: { color: "#3B82F6", width: 2 },
        marker: { size: 4 },
      },
      {
        x: summary.dates, y: summary.rmse,
        type: "scatter", mode: "lines+markers",
        name: `${t("rmse")} (ensemble)`,
        line: { color: "#ef4444", width: 2, dash: "dash" },
        marker: { size: 4 },
      },
    ], {
      xaxis: { type: "date", title: "" },
      yaxis: { title: "EUR/MWh" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 260,
    }, { responsive: true, displayModeBar: false });
  }

  // Hourly error profile: average MAE/RMSE per hour-of-day over last 30 days
  function renderHourlyErrorProfile(actuals, history) {
    const el = document.getElementById("hourly-error-chart");
    if (!el) return;
    if (!actuals || !history || !history.length) { noDataInline(el); return; }

    const actualsMap = {};
    for (const day of (actuals.days || [])) {
      if (day.prices && day.prices.length === 24) actualsMap[day.date] = day.prices;
    }

    const maeSums  = new Array(24).fill(0);
    const rmseSums = new Array(24).fill(0);
    const counts   = new Array(24).fill(0);

    for (const e of history) {
      const act = actualsMap[e.date];
      if (!act || e.prices.length < 24) continue;
      for (let h = 0; h < 24; h++) {
        const err = e.prices[h] - act[h];
        maeSums[h]  += Math.abs(err);
        rmseSums[h] += err * err;
        counts[h]   += 1;
      }
    }

    const hours = Array.from({ length: 24 }, (_, h) => h);
    const maeByHour  = hours.map(h => counts[h] ? Math.round(maeSums[h]  / counts[h] * 100) / 100 : null);
    const rmseByHour = hours.map(h => counts[h] ? Math.round(Math.sqrt(rmseSums[h] / counts[h]) * 100) / 100 : null);
    const nDays = Math.max(...counts);

    if (!nDays) { noDataInline(el); return; }

    Plotly.newPlot(el, [
      {
        x: hours, y: maeByHour, type: "bar",
        name: t("mae"),
        marker: { color: "#3B82F6" },
      },
      {
        x: hours, y: rmseByHour, type: "scatter", mode: "lines+markers",
        name: t("rmse"),
        line: { color: "#ef4444", width: 2 },
        marker: { size: 5 },
        yaxis: "y",
      },
    ], {
      xaxis: { title: "Hour of day (UTC)", dtick: 2, tick0: 0 },
      yaxis: { title: "EUR/MWh" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 240,
      annotations: [{
        x: 23, y: 0, xanchor: "right", yanchor: "bottom",
        text: `${nDays} days`,
        showarrow: false, font: { size: 11, color: "#888" },
      }],
    }, { responsive: true, displayModeBar: false });
  }

  function renderCompositionChart(metadata) {
    const el = document.getElementById("composition-chart");
    if (!metadata || !metadata.models || !metadata.models.length) {
      noDataInline(el); return;
    }

    const models    = [...metadata.models].sort((a, b) => b.weight - a.weight);
    const maxWeight = Math.max(...models.map(m => m.weight));

    Plotly.newPlot(el, [{
      x: models.map(m => m.weight),
      y: models.map(m => m.name),
      type: "bar",
      orientation: "h",
      marker: { color: models.map(m => CATEGORY_COLORS[m.category] || "#6c757d") },
      text: models.map(m => `${(m.weight * 100).toFixed(1)}%`),
      textposition: "outside",
    }], {
      xaxis: { title: "Ensemble weight", tickformat: ",.0%", range: [0, maxWeight * 1.25] },
      yaxis: { automargin: true },
      margin: { t: 20, r: 80, b: 60, l: 240 },
      height: Math.max(200, models.length * 50 + 60),
    }, { responsive: true, displayModeBar: false });

    const panel = document.getElementById("info-panel");
    if (panel) {
      const fmt = (v, d = 3) => v !== undefined && v !== null ? v.toFixed(d) : "—";
      panel.innerHTML = [
        { label: t("mae"),               value: `${fmt(metadata.holdout_mae, 2)} EUR/MWh` },
        { label: "RMSE",                  value: `${fmt(metadata.holdout_rmse, 2)} EUR/MWh` },
        { label: t("pi_coverage"),        value: `${(metadata.pi_coverage * 100).toFixed(1)}%` },
        { label: t("conformal_quantile"), value: `±${fmt(metadata.conformal_quantile, 2)} EUR/MWh` },
        { label: t("last_retrain"),       value: metadata.last_retrain ? metadata.last_retrain.slice(0, 10) : "—" },
      ].map(i => `<div class="info-item"><strong>${i.label}</strong>: ${i.value}</div>`).join("");
    }
  }

  function renderGenLoadErrors(glErrors) {
    const el = document.getElementById("gen-load-error-chart");
    if (!el) return;
    if (!glErrors || !Object.keys(glErrors).length) { noDataInline(el); return; }

    const traces = [];
    let hasOOF = false;
    let hasProduction = false;

    for (const [target, data] of Object.entries(glErrors)) {
      if (!data.dates || !data.mae) continue;
      // Tag each point as OOF (pre-production) or production (recent)
      // OOF dates are anything before June 2026; production is July 2026+
      const colors = data.dates.map(d => d >= "2026-07" ? "#e74c3c" : (GL_COLORS[target] || "#888"));
      if (data.dates.some(d => d < "2026-07")) hasOOF = true;
      if (data.dates.some(d => d >= "2026-07")) hasProduction = true;

      traces.push({
        x: data.dates, y: data.mae,
        type: "scatter", mode: "lines+markers",
        name: t(target),
        line: { color: GL_COLORS[target] || "#888", width: 2 },
        marker: { size: 5, color: colors },
      });
    }
    if (!traces.length) { noDataInline(el); return; }

    const annotations = [];
    if (hasOOF && hasProduction) {
      annotations.push({
        x: "2026-04-05", y: 1, xref: "x", yref: "paper",
        text: "← OOF (cross-val) | Production →",
        showarrow: false, font: { size: 11, color: "#888" }, xanchor: "center",
      });
    }

    Plotly.newPlot(el, traces, {
      xaxis: { type: "date", title: "" },
      yaxis: { title: "MAE (MW)" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 80 },
      height: 280,
      annotations,
    }, { responsive: true, displayModeBar: false });
  }

  function renderRetrainLog(history) {
    const el = document.getElementById("retrain-log");
    if (!el) return;
    if (!history || !history.length) {
      el.innerHTML = `<p style="color:#6c757d;font-size:0.875rem">${t("no_retrain_data")}</p>`;
      return;
    }

    const rows = [...history].reverse().slice(0, 5).map(e => {
      const pct    = e.degradation_pct !== undefined ? e.degradation_pct.toFixed(1) + "%" : "—";
      const status = e.needs_reselection ? "⚠ Reselect" : "OK";
      return `<tr>
        <td>${e.date ? e.date.slice(0, 10) : "—"}</td>
        <td>${e.old_holdout_mae !== undefined ? e.old_holdout_mae.toFixed(3) : "—"}</td>
        <td>${e.new_holdout_mae !== undefined ? e.new_holdout_mae.toFixed(3) : "—"}</td>
        <td>${pct}</td>
        <td>${e.n_models !== undefined ? e.n_models : "—"}</td>
        <td>${status}</td>
      </tr>`;
    }).join("");

    el.innerHTML = `<table>
      <thead><tr>
        <th>${t("last_retrain")}</th><th>Old MAE</th><th>New MAE</th>
        <th>${t("mae_change")}</th><th>Models</th><th>Status</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  async function init() {
    translations = await fetchJSON("translations.json") || {};

    const [
      price, historyRaw, actuals, metadata,
      summary, retrainHistory, glErrors, glActuals,
      wo, woff, sol, load, gld,
    ] = await Promise.all([
      fetchJSON(DATA + "price_forecast.json"),
      fetchJSON(DATA + "forecast_history.json"),
      fetchJSON(DATA + "actuals.json"),
      fetchJSON(DATA + "model_metadata.json"),
      fetchJSON(DATA + "errors_summary.json"),
      fetchJSON(DATA + "retrain_history.json"),
      fetchJSON(DATA + "gen_load_errors_summary.json"),
      fetchJSON(DATA + "gen_load_actuals.json"),
      fetchJSON(DATA + "gen_load/wind_onshore_national.json"),
      fetchJSON(DATA + "gen_load/wind_offshore_national.json"),
      fetchJSON(DATA + "gen_load/solar_national.json"),
      fetchJSON(DATA + "gen_load/load_national.json"),
      fetchJSON(DATA + "gen_load/gen_load_diff_national.json"),
    ]);

    const history = adaptHistory(historyRaw);

    _allData = {
      metadata, summary, retrainHistory, glErrors, glActuals,
      actuals, history,
      genLoad: { wo, woff, sol, load, gld },
    };

    // Prices tab renders immediately (default active tab)
    renderPriceChart(price);
    renderHistoryChart(actuals, history);
    renderErrorChart(actuals, history);
    renderMetadata(metadata, price);
    _rendered.add("prices");

    setupTabs();
    setupLanguageToggle();
    applyTranslations();
  }

  document.addEventListener("DOMContentLoaded", init);

})();
