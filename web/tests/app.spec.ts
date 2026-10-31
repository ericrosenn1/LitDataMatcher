import { expect, test, type Page } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";
import path from "node:path";
import type { Workspace } from "../src/lib/types";

async function navigate(page: Page, label: string) {
  const menu = page.getByRole("button", {
    name: "Open navigation",
    exact: true,
  });
  if (await menu.isVisible()) await menu.click();
  await page
    .locator(".sidebar")
    .getByRole("button", { name: new RegExp(`^${label}`) })
    .click();
}

test.beforeEach(async ({ page }) => {
  await page.goto("/");
  await expect(
    page.getByRole("heading", { name: "A workspace for what’s next." }),
  ).toBeVisible();
});

test("demo renders responsive evidence workspace and labeled provenance", async ({
  page,
}, testInfo) => {
  await expect(page.locator(".opportunity")).toHaveCount(4);
  await expect(
    page.getByRole("region", { name: "Workspace statistics" }),
  ).toBeVisible();
  await expect(
    page.getByText("Illustrative literature and catalog summaries.", {
      exact: false,
    }),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBeTruthy();
  if (process.env.LDM_CAPTURE_SCREENSHOTS === "1") {
    await page.evaluate(() => document.fonts.ready);
    await mkdir(path.resolve("../docs/assets"), { recursive: true });
    await page.screenshot({
      path: path.resolve(
        `../docs/assets/${testInfo.project.name.startsWith("mobile") ? "mobile" : "workspace"}.png`,
      ),
      fullPage: true,
    });
  }
});

test("review shows evidence and component scores; saved notes persist after reload", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Review match", exact: true })
    .first()
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Connection review",
    exact: true,
  });
  await expect(
    dialog.getByRole("heading", { name: "The source evidence" }),
  ).toBeVisible();
  await expect(dialog.locator("blockquote")).not.toBeEmpty();
  await expect(
    dialog.getByText("Illustrative text", { exact: true }),
  ).toBeVisible();
  expect(await dialog.locator(".component").count()).toBeGreaterThan(3);
  await dialog
    .getByRole("combobox", { name: "Assessment", exact: true })
    .selectOption("promising");
  await dialog
    .getByRole("textbox", { name: "Research notes", exact: true })
    .fill("Verify participant overlap and study-level reuse permissions.");
  await dialog
    .getByRole("button", { name: "Save to library", exact: true })
    .click();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
  await page.reload();
  await navigate(page, "Saved library");
  await expect(page.locator(".opportunity")).toHaveCount(1);
  await page.getByRole("button", { name: "Review match", exact: true }).click();
  const restored = page.getByRole("dialog", {
    name: "Connection review",
    exact: true,
  });
  await expect(
    restored.getByRole("combobox", { name: "Assessment", exact: true }),
  ).toHaveValue("promising");
  await expect(
    restored.getByRole("textbox", { name: "Research notes", exact: true }),
  ).toHaveValue(
    "Verify participant overlap and study-level reuse permissions.",
  );
  await expect(
    restored.getByRole("button", { name: "Saved to library", exact: true }),
  ).toBeVisible();
});

test("JSON export and import restore saved reviews", async ({ page }) => {
  await page
    .getByRole("button", { name: "Save to library", exact: true })
    .first()
    .click();
  await page
    .getByRole("button", { name: "Export workspace", exact: true })
    .click();
  const downloaded = page.waitForEvent("download");
  await page.getByRole("button", { name: /Complete workspace/ }).click();
  const download = await downloaded;
  const filePath = await download.path();
  expect(filePath).not.toBeNull();
  const payload = JSON.parse(await readFile(filePath!, "utf8"));
  expect(payload.workspace.opportunities.length).toBeGreaterThan(0);
  expect(
    Object.values(payload.reviews).some(
      (review) => (review as { saved: boolean }).saved,
    ),
  ).toBeTruthy();
  await page.evaluate(() => localStorage.clear());
  await page.reload();
  await page
    .getByLabel("Import workspace or pipeline results", { exact: true })
    .setInputFiles(filePath!);
  await expect(page.getByText(/^Imported \d+ opportunities$/)).toBeVisible();
  await navigate(page, "Saved library");
  await expect(page.locator(".opportunity")).toHaveCount(1);
});

test("dataset records expose unknown sample counts and source links", async ({
  page,
}) => {
  await navigate(page, "Dataset catalog");
  await expect(page.locator(".dataset-card")).toHaveCount(5);
  await page
    .getByRole("button", { name: "View details", exact: true })
    .first()
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Dataset details",
    exact: true,
  });
  await expect(
    dialog.locator("dl").getByText("Not reported", { exact: true }),
  ).toBeVisible();
  const source = dialog.getByRole("link", { name: "Open source record" });
  await expect(source).toHaveAttribute("href", /^https:\/\//);
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
});

test("keyword filters have a useful empty state and can be cleared", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Filter matches", exact: true })
    .click();
  await page
    .getByLabel("Filter by keyword", { exact: true })
    .fill("zzzz-nonexistent-dataset");
  await expect(page.locator(".opportunity")).toHaveCount(0);
  await expect(
    page.getByRole("heading", { name: "No connections here yet." }),
  ).toBeVisible();
  await page.getByLabel("Filter by keyword", { exact: true }).fill("");
  await expect(page.locator(".opportunity")).toHaveCount(4);
});

test("pasted text is analyzed locally with its exact evidence", async ({
  page,
}) => {
  const savedQuestion = await page
    .locator(".opportunity .question-link")
    .first()
    .innerText();
  await page
    .getByRole("button", { name: "Save to library", exact: true })
    .first()
    .click();
  const requests: string[] = [];
  page.on("request", (request) => {
    if (/europepmc|datacite/.test(request.url())) requests.push(request.url());
  });
  await navigate(page, "Discover");
  await page
    .getByRole("button", { name: "Paste research text", exact: true })
    .click();
  const dialog = page.getByRole("dialog", {
    name: "Follow a question in your own text",
    exact: true,
  });
  const evidence =
    "Future studies should examine whether longitudinal microbiome measurements predict recovery after antibiotic exposure.";
  await dialog
    .getByLabel("Paper title", { exact: true })
    .fill("A locally analyzed study");
  await dialog.getByLabel("Research text", { exact: true }).fill(evidence);
  await dialog
    .getByRole("button", { name: "Find connections", exact: true })
    .click();
  await expect(dialog).toHaveCount(0);
  await expect(
    page.getByText("Text analyzed against the current dataset catalog", {
      exact: true,
    }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Review match", exact: true })
    .first()
    .click();
  await expect(page.getByRole("dialog").locator("blockquote")).toHaveText(
    evidence,
  );
  await page.keyboard.press("Escape");
  await navigate(page, "Saved library");
  await expect(page.locator(".opportunity")).toHaveCount(1);
  await expect(page.locator(".opportunity .question-link")).toHaveText(
    savedQuestion,
  );
  expect(requests).toHaveLength(0);
});

test("provider failure is visible and never replaces the workspace with fabricated results", async ({
  page,
}) => {
  const priorId = await page.evaluate(
    () => JSON.parse(localStorage.getItem("ldm.workspace.v1")!).id,
  );
  await page.route("https://www.ebi.ac.uk/europepmc/**", (route) =>
    route.fulfill({ status: 503, body: "Unavailable" }),
  );
  await page.route("https://api.datacite.org/**", (route) =>
    route.fulfill({ status: 503, body: "Unavailable" }),
  );
  await navigate(page, "Discover");
  await page
    .getByLabel("What are you curious about?", { exact: true })
    .fill("sleep cognition");
  await page
    .getByRole("button", { name: "Discover matches", exact: true })
    .click();
  await expect(page.getByRole("alert")).toContainText(
    "Both metadata sources are currently unavailable",
  );
  expect(
    await page.evaluate(
      () => JSON.parse(localStorage.getItem("ldm.workspace.v1")!).id,
    ),
  ).toBe(priorId);
});

test("a partial source failure preserves real metadata without invented matches", async ({
  page,
}) => {
  await page
    .getByRole("button", { name: "Save to library", exact: true })
    .first()
    .click();
  await page.route("https://www.ebi.ac.uk/europepmc/**", (route) =>
    route.fulfill({
      json: {
        resultList: {
          result: [
            {
              id: "12345678",
              source: "MED",
              title: "Sleep and cognition research",
              abstractText:
                "Future studies should examine sleep and cognition in adults.",
              pubYear: "2025",
            },
          ],
        },
      },
    }),
  );
  await page.route("https://api.datacite.org/**", (route) =>
    route.fulfill({ status: 503, body: "Unavailable" }),
  );
  await navigate(page, "Discover");
  await page
    .getByRole("button", { name: "Discover matches", exact: true })
    .click();
  await expect(
    page.getByText("Search complete. Source metadata is ready to explore.", {
      exact: true,
    }),
  ).toBeVisible();
  const workspace = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("ldm.workspace.v1")!),
  );
  expect(workspace.mode).toBe("live");
  expect(workspace.papers).toHaveLength(1);
  expect(workspace.datasets).toHaveLength(0);
  expect(workspace.opportunities).toHaveLength(0);
  expect(workspace.warnings.join(" ")).toContain("DataCite unavailable");
  await expect(page.locator(".opportunity")).toHaveCount(0);
  await navigate(page, "Saved library");
  await expect(page.locator(".opportunity")).toHaveCount(1);
});

test("keyboard search opens a native dialog and Escape closes it", async ({
  page,
}) => {
  await page.keyboard.press("Control+k");
  const dialog = page.getByRole("dialog", {
    name: "Explore your next question",
    exact: true,
  });
  await expect(dialog).toBeVisible();
  await expect(
    dialog.getByLabel("Search scientific literature and public datasets"),
  ).toBeFocused();
  await page.keyboard.press("Escape");
  await expect(dialog).toHaveCount(0);
});

test("importing during a pending search keeps the imported workspace and suppresses stale errors", async ({
  page,
}) => {
  const fixture: Workspace = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("ldm.workspace.v1")!),
  );
  fixture.id = "imported-while-searching";
  fixture.name = "Imported while searching";
  let release!: () => void;
  const held = new Promise<void>((resolve) => {
    release = resolve;
  });
  let requested = 0;
  let settled = 0;
  await page.route(
    /https:\/\/(www\.ebi\.ac\.uk\/europepmc|api\.datacite\.org)\//,
    async (route) => {
      requested += 1;
      await held;
      try {
        await route.fulfill({ status: 503, body: "Delayed source failure" });
      } catch {
        // The imported workspace intentionally cancels these old requests.
      } finally {
        settled += 1;
      }
    },
  );
  try {
    await navigate(page, "Discover");
    await page
      .getByRole("button", { name: "Discover matches", exact: true })
      .click();
    await expect.poll(() => requested).toBe(2);
    await page
      .getByLabel("Import workspace or pipeline results", { exact: true })
      .setInputFiles({
        name: "pending-import.json",
        mimeType: "application/json",
        buffer: Buffer.from(JSON.stringify(fixture)),
      });
    await expect(
      page.getByText("Exploring: Imported while searching", { exact: true }),
    ).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
    release();
    await expect.poll(() => settled).toBe(2);
    await expect(
      page.getByRole("button", { name: "Discover matches", exact: true }),
    ).toBeEnabled();
    expect(
      await page.evaluate(
        () => JSON.parse(localStorage.getItem("ldm.workspace.v1")!).id,
      ),
    ).toBe(fixture.id);
    await expect(
      page.getByText("Exploring: Imported while searching", { exact: true }),
    ).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
  } finally {
    release();
  }
});

test("saved source snapshots survive same-ID metadata from a later import and second save", async ({
  page,
}) => {
  const fixture: Workspace = await page.evaluate(() =>
    JSON.parse(localStorage.getItem("ldm.workspace.v1")!),
  );
  const original = fixture.opportunities[0];
  const originalPaper = fixture.papers.find(
    (paper) => paper.id === original.paperId,
  )!;
  const originalDataset = fixture.datasets.find(
    (dataset) => dataset.id === original.datasetId,
  )!;
  const originalPaperTitle = originalPaper.title;
  const originalDatasetTitle = originalDataset.title;
  await page
    .getByRole("button", { name: "Save to library", exact: true })
    .first()
    .click();

  fixture.id = "different-workspace-reused-identifiers";
  fixture.name = "Changed metadata import";
  originalPaper.title = "Changed publication title using the same source ID";
  originalDataset.title = "Changed dataset title using the same dataset ID";
  fixture.opportunities = [
    {
      ...original,
      id: "second-distinct-opportunity",
      question: "Can the changed dataset support a second research question?",
    },
  ];
  await page
    .getByLabel("Import workspace or pipeline results", { exact: true })
    .setInputFiles({
      name: "changed-metadata.json",
      mimeType: "application/json",
      buffer: Buffer.from(JSON.stringify(fixture)),
    });
  await expect(
    page.getByText("Exploring: Changed metadata import", { exact: true }),
  ).toBeVisible();

  const inspectOriginal = async () => {
    await navigate(page, "Saved library");
    await page
      .getByRole("button", { name: original.question, exact: true })
      .click();
    const dialog = page.getByRole("dialog", {
      name: "Connection review",
      exact: true,
    });
    await expect(dialog.locator(".source-citation")).toContainText(
      originalPaperTitle,
    );
    await expect(
      dialog.getByRole("heading", { name: originalDatasetTitle, exact: true }),
    ).toBeVisible();
    await expect(
      dialog.getByText(originalPaper.title, { exact: true }),
    ).toHaveCount(0);
    await expect(
      dialog.getByRole("heading", { name: originalDataset.title, exact: true }),
    ).toHaveCount(0);
    await page.keyboard.press("Escape");
  };
  await inspectOriginal();
  await navigate(page, "Discover");
  await page
    .getByRole("button", { name: "Save to library", exact: true })
    .click();
  await inspectOriginal();
  await expect(page.locator(".opportunity")).toHaveCount(2);
});
