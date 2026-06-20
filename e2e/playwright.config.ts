import { defineConfig } from "@playwright/test";

// Drives the running app (API :8000 + web :3000) to produce the end-to-end
// walkthrough: screenshots into ../docs/e2e/shots and a recorded video.
export default defineConfig({
  testDir: ".",
  outputDir: "../docs/e2e/artifacts",
  timeout: 240_000,
  expect: { timeout: 60_000 },
  reporter: "list",
  use: {
    baseURL: "http://localhost:3000",
    viewport: { width: 1440, height: 900 },
    video: { mode: "on", size: { width: 1440, height: 900 } },
    trace: "on",
    screenshot: "only-on-failure",
    actionTimeout: 30_000,
    launchOptions: { slowMo: 250 }, // slower motion → a watchable video
  },
});
