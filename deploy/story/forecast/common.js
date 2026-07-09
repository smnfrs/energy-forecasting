/* Shared Plotly chart helpers for the Stage 10 forecast story.
   Unlike Stage 9's charts.js, this page's data changes daily/weekly and is
   always served over https in production, so data loads via fetch() rather
   than an inlined <script type="application/json"> data island. */

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

function hexToRgba(hex, alpha) {
  const h = hex.replace("#", "");
  const r = parseInt(h.substring(0, 2), 16);
  const g = parseInt(h.substring(2, 4), 16);
  const b = parseInt(h.substring(4, 6), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

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

/** Fetch and parse a JSON file; returns null (never throws) on any failure so
 * callers can show a graceful "data unavailable" state instead of a dead page. */
async function fetchJSON(url) {
  try {
    const resp = await fetch(url, { cache: "no-store" });
    if (!resp.ok) return null;
    return await resp.json();
  } catch (err) {
    console.warn(`fetchJSON failed for ${url}:`, err);
    return null;
  }
}

/** Fill an element with AI narrative text, or a graceful fallback if unavailable. */
function renderNarrative(elId, text, status) {
  const el = document.getElementById(elId);
  if (!el) return;
  if (status === "ok" && text) {
    el.textContent = text;
    el.classList.remove("narrative-unavailable");
  } else {
    el.textContent = "AI summary unavailable right now — the numbers above are still live.";
    el.classList.add("narrative-unavailable");
  }
}
