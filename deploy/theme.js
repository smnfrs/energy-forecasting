/* Shared light/dark theme toggle for both public surfaces (the forecast story
   deck and the dashboard). The initial theme is set by a tiny inline script in
   each page's <head> BEFORE first paint (reads localStorage, falls back to the
   OS preference) so there is no flash of the wrong theme. This file only wires
   up the toggle button(s) after the DOM is ready.

   On toggle it flips data-theme on <html>, persists the choice, updates every
   [data-theme-toggle] button, and dispatches a "themechange" CustomEvent so the
   page can recolour its Plotly charts (their colours are read from CSS vars at
   build time, so they need an explicit rebuild — see each page's listener). */
(function () {
  const KEY = "energy-theme";

  function current() {
    return document.documentElement.getAttribute("data-theme") || "light";
  }

  function syncButtons(theme) {
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.setAttribute("aria-pressed", String(theme === "dark"));
      btn.setAttribute(
        "aria-label",
        theme === "dark" ? "Switch to light theme" : "Switch to dark theme"
      );
      const icon = btn.querySelector("[data-theme-icon]");
      if (icon) icon.textContent = theme === "dark" ? "☀" : "☾"; // ☀ / ☾
    });
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem(KEY, theme);
    } catch (e) {
      /* private-mode / storage disabled — theme still applies for this page */
    }
    syncButtons(theme);
    document.dispatchEvent(new CustomEvent("themechange", { detail: { theme } }));
  }

  document.addEventListener("DOMContentLoaded", () => {
    syncButtons(current()); // match the button glyph to the pre-painted theme
    document.querySelectorAll("[data-theme-toggle]").forEach((btn) => {
      btn.addEventListener("click", () =>
        setTheme(current() === "dark" ? "light" : "dark")
      );
    });
  });
})();
