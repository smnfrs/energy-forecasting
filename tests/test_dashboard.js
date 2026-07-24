/**
 * Playwright smoke test — run locally with `make test-dashboard`.
 * Requires `make serve-dashboard` running in a separate terminal first.
 */
const { test, expect } = require("@playwright/test");

const BASE = "http://localhost:8080";
// The site root now redirects to the story deck; the dashboard lives at
// dashboard.html (Stage 3 entry-structure change).
const DASH = `${BASE}/dashboard.html`;

test("main page renders price chart", async ({ page }) => {
  await page.goto(DASH);
  await page.waitForSelector("#forecast-chart .svg-container", { timeout: 5000 });
  const svg = await page.$("#forecast-chart svg");
  expect(svg).not.toBeNull();
});

test("gen/load summary chart renders", async ({ page }) => {
  await page.goto(DASH);
  await page.waitForSelector("#summary-chart .svg-container", { timeout: 5000 });
  const svg = await page.$("#summary-chart svg");
  expect(svg).not.toBeNull();
});

test("monitoring page renders composition chart", async ({ page }) => {
  await page.goto(`${BASE}/monitoring.html`);
  await page.waitForSelector("#composition-chart .svg-container", { timeout: 5000 });
  const svg = await page.$("#composition-chart svg");
  expect(svg).not.toBeNull();
});

test("DE/EN toggle switches language on main page", async ({ page }) => {
  await page.goto(DASH);
  const btn = page.locator("#lang-toggle");
  await expect(btn).toHaveText("DE");
  await btn.click();
  await expect(btn).toHaveText("EN");
});

test("monitoring page lang toggle works", async ({ page }) => {
  await page.goto(`${BASE}/monitoring.html`);
  const btn = page.locator("#lang-toggle");
  await expect(btn).toHaveText("DE");
  await btn.click();
  await expect(btn).toHaveText("EN");
});

test("site root redirects to the story deck", async ({ page }) => {
  await page.goto(BASE);
  await page.waitForURL(/story\/forecast\/index\.html$/, { timeout: 5000 });
  await expect(page.locator("nav.story-nav")).toBeVisible();
});

test("story deck theme toggle flips data-theme", async ({ page }) => {
  await page.goto(`${BASE}/story/forecast/index.html`);
  const html = page.locator("html");
  const before = await html.getAttribute("data-theme");
  await page.locator("[data-theme-toggle]").click();
  const after = await html.getAttribute("data-theme");
  expect(after).not.toBe(before);
  expect(["light", "dark"]).toContain(after);
});
