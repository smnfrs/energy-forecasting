(function () {
  "use strict";

  // When a live API is available, set window.API_DATA_BASE before loading this script.
  const DATA = (typeof window !== "undefined" && window.API_DATA_BASE) || "data/";
  let currentLang = "en";
  let translations = {};

  const CATEGORY_COLORS = {
    lgbm: "#22c55e",
    xgboost: "#3B82F6",
    catboost: "#f59e0b",
    linear: "#8b5cf6",
  };

  // ── Utilities ─────────────────────────────────────────────────────────────

  async function fetchJSON(path) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) return null;
      return await resp.json();
    } catch { return null; }
  }

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

  function noData(el, key) {
    el.innerHTML = `<p class="no-data">${t(key || "no_data")}</p>`;
  }

  // ── §1 Price Error Trend ───────────────────────────────────────────────────
  // Stage 7a: uses errors_summary.json (ensemble MAE/RMSE).
  // Stage 7b: extend to per-model lines from model_errors.json.

  function renderErrorTrend(summary) {
    const el = document.getElementById("error-trend-chart");
    if (!summary || !summary.dates || !summary.dates.length) {
      noData(el); return;
    }
    Plotly.newPlot(el, [
      {
        x: summary.dates, y: summary.mae,
        type: "scatter", mode: "lines+markers",
        name: t("mae") + " (ensemble)",
        line: { color: "#3B82F6", width: 2 },
        marker: { size: 4 },
      },
      {
        x: summary.dates, y: summary.rmse,
        type: "scatter", mode: "lines+markers",
        name: t("rmse") + " (ensemble)",
        line: { color: "#ef4444", width: 2, dash: "dash" },
        marker: { size: 4 },
      },
    ], {
      xaxis: { type: "date", title: "" },
      yaxis: { title: "EUR/MWh" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 65 },
      height: 280,
    }, { responsive: true, displayModeBar: false });
  }

  // ── §2 Ensemble Composition ────────────────────────────────────────────────

  function renderCompositionChart(metadata) {
    const el = document.getElementById("composition-chart");
    if (!metadata || !metadata.models || !metadata.models.length) {
      noData(el); return;
    }

    const models = [...metadata.models].sort((a, b) => b.weight - a.weight);
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
      xaxis: {
        title: "Ensemble weight",
        tickformat: ",.0%",
        range: [0, maxWeight * 1.25],
      },
      yaxis: { automargin: true },
      margin: { t: 20, r: 80, b: 60, l: 240 },
      height: Math.max(200, models.length * 50 + 60),
    }, { responsive: true, displayModeBar: false });

    // Info panel
    const panel = document.getElementById("info-panel");
    if (panel) {
      const fmt = (v, digits = 3) => v !== undefined && v !== null ? v.toFixed(digits) : "—";
      const items = [
        { label: t("mae"),               value: `${fmt(metadata.holdout_mae, 2)} EUR/MWh` },
        { label: "RMSE",                  value: `${fmt(metadata.holdout_rmse, 2)} EUR/MWh` },
        { label: t("pi_coverage"),        value: `${(metadata.pi_coverage * 100).toFixed(1)}%` },
        { label: t("conformal_quantile"), value: `±${fmt(metadata.conformal_quantile, 2)} EUR/MWh` },
        { label: t("last_retrain"),       value: metadata.last_retrain ? metadata.last_retrain.slice(0, 10) : "—" },
      ];
      panel.innerHTML = items
        .map(i => `<div class="info-item"><strong>${i.label}</strong>: ${i.value}</div>`)
        .join("");
    }
  }

  // ── §3 Gen/Load Accuracy ──────────────────────────────────────────────────

  const GL_COLORS = {
    wind_onshore: "#2266CC",
    wind_offshore: "#44AADD",
    solar: "#DDAA00",
    load: "#EE0000",
  };

  function renderGenLoadErrors(glErrors) {
    const el = document.getElementById("gen-load-error-chart");
    if (!el) return;
    if (!glErrors || !Object.keys(glErrors).length) {
      noData(el, "no_data"); return;
    }

    const traces = [];
    for (const [target, data] of Object.entries(glErrors)) {
      if (!data.dates || !data.mae) continue;
      traces.push({
        x: data.dates,
        y: data.mae,
        type: "scatter",
        mode: "lines+markers",
        name: t(target),
        line: { color: GL_COLORS[target] || "#888", width: 2 },
        marker: { size: 4 },
      });
    }
    if (!traces.length) { noData(el, "no_data"); return; }

    Plotly.newPlot(el, traces, {
      xaxis: { type: "date", title: "" },
      yaxis: { title: "MAE (MW)" },
      legend: { orientation: "h", y: -0.2 },
      margin: { t: 20, r: 20, b: 60, l: 80 },
      height: 280,
    }, { responsive: true, displayModeBar: false });
  }

  // ── §4 Retrain Log ────────────────────────────────────────────────────────

  function renderRetrainLog(history) {
    const el = document.getElementById("retrain-log");
    if (!el) return;
    if (!history || !history.length) {
      el.innerHTML = `<p style="color:#6c757d;font-size:0.875rem">${t("no_retrain_data")}</p>`;
      return;
    }

    const rows = [...history]
      .reverse()
      .slice(0, 5)
      .map(e => {
        const pct = e.degradation_pct !== undefined ? e.degradation_pct.toFixed(1) + "%" : "—";
        const status = e.needs_reselection ? "⚠ Reselect" : "OK";
        return `<tr>
          <td>${e.date ? e.date.slice(0, 10) : "—"}</td>
          <td>${e.old_holdout_mae !== undefined ? e.old_holdout_mae.toFixed(3) : "—"}</td>
          <td>${e.new_holdout_mae !== undefined ? e.new_holdout_mae.toFixed(3) : "—"}</td>
          <td>${pct}</td>
          <td>${e.n_models !== undefined ? e.n_models : "—"}</td>
          <td>${status}</td>
        </tr>`;
      })
      .join("");

    el.innerHTML = `<table>
      <thead><tr>
        <th>${t("last_retrain")}</th>
        <th>Old MAE</th>
        <th>New MAE</th>
        <th>${t("mae_change")}</th>
        <th>Models</th>
        <th>Status</th>
      </tr></thead>
      <tbody>${rows}</tbody>
    </table>`;
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  async function init() {
    translations = await fetchJSON("translations.json") || {};

    const [summary, metadata, retrainHistory, glErrors] = await Promise.all([
      fetchJSON(DATA + "errors_summary.json"),
      fetchJSON(DATA + "model_metadata.json"),
      fetchJSON(DATA + "retrain_history.json"),
      fetchJSON(DATA + "gen_load_errors_summary.json"),
    ]);

    renderErrorTrend(summary);
    renderCompositionChart(metadata);
    renderGenLoadErrors(glErrors);
    renderRetrainLog(retrainHistory);
    setupLanguageToggle();
    applyTranslations();
  }

  document.addEventListener("DOMContentLoaded", init);

})();
