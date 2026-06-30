/**
 * Playwright smoke test — run locally with `make test-dashboard`.
 * Requires `make serve-dashboard` running in a separate terminal first.
 */
const { test, expect } = require("@playwright/test");

const BASE = "http://localhost:8080";

test("main page renders price chart", async ({ page }) => {
  await page.goto(BASE);
  await page.waitForSelector("#forecast-chart .svg-container", { timeout: 5000 });
  const svg = await page.$("#forecast-chart svg");
  expect(svg).not.toBeNull();
});

test("gen/load summary chart renders", async ({ page }) => {
  await page.goto(BASE);
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
  await page.goto(BASE);
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
