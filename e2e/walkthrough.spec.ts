import { test, expect, type Page, type Locator } from "@playwright/test";
import path from "node:path";
import fs from "node:fs";

const SHOTS = path.resolve(__dirname, "../docs/e2e/shots");
const SAMPLE = path.resolve(__dirname, "sample.png");
fs.mkdirSync(SHOTS, { recursive: true });

// Hold a frame so the recorded video lingers on each screen (smooth pacing).
const dwell = (page: Page, ms = 2600) => page.waitForTimeout(ms);

// Full-screen title / outro cards (amber + white-cream, centred) recorded into the video.
const card = (kicker: string, title: string, sub: string, pill: string) => `<!doctype html><html><body style="margin:0;height:100vh;display:flex;align-items:center;justify-content:center;background:radial-gradient(1200px 600px at 28% 18%,#FCEFD9,transparent),radial-gradient(1000px 520px at 82% 82%,#F7E7D2,transparent),#FAF9F5;font-family:-apple-system,'Inter',system-ui,sans-serif">
  <div style="text-align:center;padding:48px">
    <div style="font-size:14px;letter-spacing:.34em;text-transform:uppercase;color:#C96442;font-weight:800;margin-bottom:20px">${kicker}</div>
    <div style="font-size:52px;font-weight:800;color:#1F1E1C;line-height:1.12;max-width:980px;margin:0 auto;letter-spacing:-.02em">${title}</div>
    <div style="font-size:23px;color:#6E5B47;margin-top:16px">${sub}</div>
    <div style="margin-top:38px;display:inline-block;padding:12px 32px;border-radius:999px;background:linear-gradient(90deg,#C96442,#D9A441);color:#fff;font-weight:800;font-size:19px;box-shadow:0 10px 34px rgba(201,100,66,.28)">${pill}</div>
  </div></body></html>`;

// Wait until a streamed answer has finished PRINTING (text stops growing) before advancing.
async function waitForAnswerComplete(page: Page, bubble: Locator) {
  await bubble.waitFor({ state: "visible", timeout: 90_000 });
  let prev = "";
  let stable = 0;
  for (let i = 0; i < 160; i++) {
    const txt = (await bubble.innerText().catch(() => "")) || "";
    if (txt.length > 0 && txt === prev) {
      if (++stable >= 3) return;
    } else stable = 0;
    prev = txt;
    await page.waitForTimeout(500);
  }
}

test("Deep Search — end-to-end walkthrough", async ({ page }) => {
  test.setTimeout(300_000);

  // ── 0) Intro title card ────────────────────────────────────────────────
  await page.setContent(
    card("Gemma 4 · Qdrant · 100% local", "Deep Search in Media Library",
         "Using Gemma 4 and Qdrant Multivectors", "A Walkthrough")
  );
  await dwell(page, 4500);

  // ── 1) Home: chat-first UI, health banner, sidebar ─────────────────────
  await page.goto("/");
  await expect(page.getByText(/Gemma 4 ready|Gemma 4 unavailable/)).toBeVisible({ timeout: 40_000 });
  await expect(page.getByText("Ask your library")).toBeVisible();
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/01-home.png`, fullPage: true });

  // ── 2) Sidebar navigation ──────────────────────────────────────────────
  await expect(page.locator("aside").getByRole("button", { name: /Chat/ })).toBeVisible();
  await expect(page.locator("aside").getByRole("button", { name: /Search/ })).toBeVisible();
  await expect(page.locator("aside").getByRole("button", { name: /Library/ })).toBeVisible();
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/02-sidebar-nav.png` });

  // ── 3) Attach media (the paperclip button): upload + auto-index with live progress ───────
  await page.setInputFiles('input[type="file"]', SAMPLE);
  await expect(page.getByText(/Uploading|Indexing|Preparing|Transcribing/).first()).toBeVisible({ timeout: 20_000 });
  await dwell(page, 1500);
  await page.screenshot({ path: `${SHOTS}/03-uploading.png` });

  // ── 4) Indexed: wait for the confirmation output ───────────────────────
  await expect(page.getByText(/Added .*to your library/)).toBeVisible({ timeout: 180_000 });
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/04-indexed.png` });

  const composer = page.getByPlaceholder(/Ask about your media/);

  // ── 5) Library question → WAIT for the answer to print ─────────────────
  await composer.fill("How many types of files do I have in my library?");
  await dwell(page, 1000);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await waitForAnswerComplete(page, page.locator(".chat-md").last());
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/05-chat-answer.png` });

  // ── 6) Content question → WAIT for printed answer, expand sources ──────
  await composer.fill("Describe what is in my media.");
  await dwell(page, 1000);
  await page.getByRole("button", { name: "Ask", exact: true }).click();
  await waitForAnswerComplete(page, page.locator(".chat-md").last());
  const sources = page.getByText(/\d+ sources?/).last();
  if (await sources.isVisible().catch(() => false)) {
    await sources.click();
    await dwell(page, 1500);
  }
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/06-chat-sources.png` });

  // ── 7) Search tab ──────────────────────────────────────────────────────
  await page.locator("aside").getByRole("button", { name: /Search/ }).click();
  await expect(page.locator('input[placeholder*="Find the chart"]')).toBeVisible();
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/07-search-tab.png` });

  // ── 8) Run a search → WAIT for results to render ───────────────────────
  await page.locator('input[placeholder*="Find the chart"]').fill("geometric shapes red square");
  await dwell(page, 800);
  await page.locator("main").getByRole("button", { name: "Search", exact: true }).click();
  // wait for actual output: a result card or the "no matches" message
  await page.locator('.ds-card, :text("No matches")').first().waitFor({ timeout: 60_000 }).catch(() => {});
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/08-search-results.png` });

  // ── 9) Library tab ─────────────────────────────────────────────────────
  await page.locator("aside").getByRole("button", { name: /Library/ }).click();
  await expect(page.getByText("Assets", { exact: true })).toBeVisible();
  await dwell(page);
  await page.screenshot({ path: `${SHOTS}/09-library.png`, fullPage: true });

  // ── Outro card ─────────────────────────────────────────────────────────
  await page.setContent(
    card("Thanks for watching", "Deep Search in Media Library",
         "Local multimodal search · Gemma 4 + Qdrant", "Built by Sarvesh Talele")
  );
  await dwell(page, 4000);
});
