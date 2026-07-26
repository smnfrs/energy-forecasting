// Playwright config for the dashboard + story-deck smoke tests.
//
// Without this, `npx playwright test test_dashboard.js` (what `make
// test-dashboard` runs) finds nothing: Playwright's default testMatch only
// picks up *.spec / *.test files, and our file is test_dashboard.js. Point it
// at the file explicitly. Default to headless; the Makefile passes --headed for
// a visual walk, which overrides this.
module.exports = {
  testDir: __dirname,
  testMatch: "test_dashboard.js",
  timeout: 20000,
  use: { headless: true },
};
