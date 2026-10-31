import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";
import { readFile, writeFile } from "node:fs/promises";

const contract = {
  compilation_status: "COMPLETE",
  question_purpose: "comparative",
  matching_requirements: [
    { field: "species", expected: "human", essential: true },
    { field: "paired", expected: false, essential: false },
  ],
  ambiguities: [],
};

const result = {
  question: {
    question: "Does insulin increase glucose uptake in humans?",
    question_id: "fixture-selected",
    source_locator: "user question",
    novelty_status: "NOT_ESTABLISHED",
  },
  compilation: contract,
  status: "PARTIAL",
  scientific_status: "UNVALIDATED_DEVELOPMENT_FIXTURE",
  questions: {
    questions: [
      {
        question_id: "fixture-q",
        question: "Further studies should examine glucose uptake in humans.",
        category: "SOURCE_STATED_UNRESOLVED",
        unanswered_status: "SOURCE_STATES_UNRESOLVED",
        source_locator: "https://example.org/source#chars=0:64",
      },
    ],
  },
  candidates: [
    {
      dataset_id: "FIXTURE-A",
      dataset: {
        title: "Synthetic compatible metadata",
        source: "test fixture",
        url: "https://example.org/a",
      },
      assessment: {
        eligibility: "DIRECT_FIT",
        compatibility_status: "METADATA_COMPATIBLE",
        requirements: [
          {
            field: "species",
            expected: "human",
            essential: true,
            status: "MATCH",
            observation: {
              value: "Homo sapiens",
              source_locator: "https://example.org/a#species",
            },
          },
        ],
      },
    },
    {
      dataset_id: "FIXTURE-B",
      dataset: {
        title: "Synthetic incompatible metadata",
        source: "test fixture",
        url: "javascript:alert(1)",
      },
      assessment: {
        eligibility: "NOT_QUALIFIED",
        compatibility_status: "INCOMPATIBLE",
        requirements: [
          {
            field: "species",
            expected: "human",
            essential: true,
            status: "MISMATCH",
            observation: {
              value: "mouse",
              source_locator: "https://example.org/b#species",
            },
          },
        ],
      },
    },
  ],
};

async function navigate(page: Page) {
  await page.goto("/");
  const menu = page.getByRole("button", {
    name: "Open navigation",
    exact: true,
  });
  if (await menu.isVisible()) await menu.click();
  await page
    .locator(".sidebar")
    .getByRole("button", { name: "Scientific workflow", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Scientific discovery", exact: true }),
  ).toBeVisible();
}

async function mockedEngine(page: Page) {
  const requests: {
    path: string;
    body: Record<string, unknown> | undefined;
  }[] = [];
  let reads = 0;
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const body =
      request.method() === "POST" ? request.postDataJSON() : undefined;
    requests.push({ path, body });
    let payload: unknown;
    if (path === "/api/health")
      payload = {
        mode: "local",
        worker_active: true,
        model_available: false,
        semantic_available: false,
        sources: {
          literature: ["pubmed", "europepmc"],
          datasets: ["geo", "pride"],
        },
      };
    else if (path === "/api/requirements") payload = contract;
    else if (path === "/api/jobs" && request.method() === "GET")
      payload = { jobs: [{ id: "saved-fixture", status: "COMPLETED" }] };
    else if (path === "/api/jobs")
      payload = { id: "new-fixture", status: "QUEUED" };
    else if (path === "/api/jobs/new-fixture")
      payload =
        ++reads === 1
          ? {
              id: "new-fixture",
              status: "RUNNING",
              progress: { stage: "inspection" },
            }
          : {
              id: "new-fixture",
              status: "COMPLETED",
              progress: { stage: "complete" },
              result,
            };
    else if (path === "/api/jobs/saved-fixture")
      payload = { id: "saved-fixture", status: "COMPLETED", result };
    else if (path === "/api/literature")
      payload = {
        documents: [
          {
            document_id: "fixture-paper",
            title: "Synthetic source for workflow testing",
            text: "Further studies should examine glucose uptake in humans.",
            url: "https://example.org/source",
          },
        ],
        coverage: {},
      };
    else if (path === "/api/pause" || path === "/api/resume")
      payload = { status: "SUCCESS" };
    else payload = { error: `Unexpected fixture request: ${path}` };
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(payload),
    });
  });
  return requests;
}

test("explicit requirements reach a queued job and source evidence remains inspectable", async ({
  page,
}) => {
  const requests = await mockedEngine(page);
  await navigate(page);
  await expect(
    page.getByText("Engine connected", { exact: false }),
  ).toBeVisible();
  await expect(
    page.getByLabel("Question extraction").locator('option[value="model"]'),
  ).toHaveAttribute("disabled", "");
  await expect(
    page.getByLabel("Candidate retrieval").locator('option[value="hybrid"]'),
  ).toHaveAttribute("disabled", "");
  await page
    .getByLabel("Research question", { exact: true })
    .fill(result.question.question);
  await page
    .getByRole("button", { name: "Add requirement", exact: true })
    .click();
  await page.getByLabel("Requirement field 1").selectOption("paired");
  await page.getByLabel("Expected value 1").fill("false");
  await page.getByLabel("Required", { exact: true }).uncheck();
  await page.getByRole("button", { name: "Preview requirements" }).click();
  await expect(page.locator(".scientific-contract")).toContainText("Preferred");
  expect(
    requests.find((row) => row.path === "/api/requirements")?.body
      ?.requirements,
  ).toEqual([{ field: "paired", expected: false, essential: false }]);
  await page.getByRole("button", { name: "Launch research workflow" }).click();
  await expect(page.locator(".scientific-job")).toContainText("inspection");
  await expect(
    page.getByRole("heading", {
      name: "Evidence and compatibility",
      exact: true,
    }),
  ).toBeVisible();
  expect(
    requests.find((row) => row.path === "/api/jobs" && row.body)?.body
      ?.requirements,
  ).toEqual([{ field: "paired", expected: false, essential: false }]);
  await expect(page.locator(".scientific-candidates > article")).toHaveCount(2);
  await page.getByLabel("Compatibility filter").selectOption("DIRECT_FIT");
  await expect(page.locator(".scientific-candidates > article")).toHaveCount(1);
  await page.getByText("Inspect requirement evidence", { exact: true }).click();
  await expect(page.locator(".scientific-table-wrap")).toContainText(
    "Homo sapiens",
  );
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await expect(page.locator(".scientific-table-wrap a")).toHaveAttribute(
    "href",
    "https://example.org/a#species",
  );
  await expect(
    page.getByRole("link", { name: "Dossier", exact: true }),
  ).toHaveAttribute("href", "/api/jobs/new-fixture/dossier.md");
  await page.getByText("Source-linked questions (1)", { exact: true }).click();
  await page
    .getByRole("button", { name: "Use this question", exact: true })
    .click();
  await expect(
    page.getByLabel("Research question", { exact: true }),
  ).toHaveValue(result.questions.questions[0].question);
  await expect(page.locator(".scientific-contract")).toHaveCount(0);
});

test("saved jobs, local source selection and queue controls survive navigation", async ({
  page,
}) => {
  const requests = await mockedEngine(page);
  await navigate(page);
  await page
    .getByLabel("Literature search", { exact: true })
    .fill("glucose uptake");
  await page
    .getByRole("button", { name: "Search local corpus", exact: true })
    .click();
  await page.getByRole("button", { name: "Use source", exact: true }).click();
  await expect(page.locator(".scientific-attached")).toContainText(
    "Synthetic source for workflow testing",
  );
  await page.getByLabel("Saved research job").selectOption("saved-fixture");
  await expect(
    page.getByRole("heading", {
      name: "Evidence and compatibility",
      exact: true,
    }),
  ).toBeVisible();
  await expect(page.locator('a[href^="javascript:"]')).toHaveCount(0);
  await page
    .getByRole("button", { name: "Pause new jobs", exact: true })
    .click();
  await expect(
    page.getByRole("button", { name: "Resume jobs", exact: true }),
  ).toBeVisible();
  await page.getByRole("button", { name: "Resume jobs", exact: true }).click();
  expect(requests.some((row) => row.path === "/api/pause")).toBe(true);
  expect(requests.some((row) => row.path === "/api/resume")).toBe(true);
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  const accessibility = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(
    accessibility.violations.map((row) => ({
      id: row.id,
      nodes: row.nodes.map((node) => node.target),
    })),
  ).toEqual([]);
});

test("an absent engine is a visible connection failure and cannot launch a job", async ({
  page,
}) => {
  await page.route("**/api/**", (route) =>
    route.fulfill({
      status: 200,
      contentType: "text/html",
      body: "<html>Static application only</html>",
    }),
  );
  await navigate(page);
  await expect(page.getByRole("alert")).toContainText(
    "Start the local Python server",
  );
  await page
    .getByLabel("Research question", { exact: true })
    .fill(result.question.question);
  await expect(
    page.getByRole("button", { name: "Launch research workflow" }),
  ).toBeDisabled();
  await expect(
    page.getByRole("heading", { name: "Evidence and compatibility" }),
  ).toHaveCount(0);
});

test("live local catalog completes a browser research workflow and exports its dossier", async ({
  page,
}, testInfo) => {
  test.skip(
    process.env.LDM_LIVE_PLATFORM !== "1",
    "Requires the explicitly configured local Python research server.",
  );
  test.skip(
    testInfo.project.name !== "desktop-chromium",
    "Run the real computation once; responsive behavior uses the mocked suite.",
  );
  test.setTimeout(180_000);
  await navigate(page);
  await expect(
    page.getByText("Engine connected", { exact: false }),
  ).toBeVisible();
  const question = "Is cholesterol associated with heart disease in humans?";
  let job: { id: string };
  if (process.env.LDM_LIVE_JOB_ID) {
    job = { id: process.env.LDM_LIVE_JOB_ID };
    await page.getByLabel("Saved research job").selectOption(job.id);
  } else {
    await page
      .getByLabel("Literature search", { exact: true })
      .fill("heart disease cholesterol");
    await page.getByLabel("Research question", { exact: true }).fill(question);
    await page.getByRole("button", { name: "Preview requirements" }).click();
    await expect(page.locator(".scientific-contract")).toContainText(
      "cholesterol",
    );
    await page.getByText("Public sources", { exact: true }).click();
    while (
      await page.locator(".scientific-source-groups input:checked").count()
    ) {
      await page
        .locator(".scientific-source-groups input:checked")
        .first()
        .uncheck();
    }
    await page.getByLabel("Datasets to inspect").fill("0");
    const submitted = page.waitForResponse(
      (response) =>
        new URL(response.url()).pathname === "/api/jobs" &&
        response.request().method() === "POST",
    );
    await page
      .getByRole("button", { name: "Launch research workflow" })
      .click();
    const submission = await submitted;
    expect(submission.status()).toBe(202);
    job = await submission.json();
  }
  await expect(
    page.getByRole("heading", {
      name: "Evidence and compatibility",
      exact: true,
    }),
  ).toBeVisible({ timeout: 150_000 });
  const response = await page.request.get(
    `/api/jobs/${encodeURIComponent(job.id)}`,
  );
  expect(response.ok()).toBe(true);
  const completed = await response.json();
  expect(completed.status).toBe("SUCCEEDED");
  expect(completed.result.question.question || completed.result.question).toBe(
    question,
  );
  expect(
    completed.result.compilation.matching_requirements.length,
  ).toBeGreaterThan(0);
  expect(Array.isArray(completed.result.candidates)).toBe(true);
  const downloadEvent = page.waitForEvent("download");
  await page.getByRole("link", { name: "Dossier", exact: true }).click();
  const download = await downloadEvent;
  const path = await download.path();
  expect(path).not.toBeNull();
  expect(await readFile(path!, "utf-8")).toContain(question);
  const receipt = {
    job_id: job.id,
    status: completed.status,
    question,
    candidates: completed.result.candidates.length,
    eligibility: completed.result.candidates.map(
      (row: { dataset_id: string; assessment: { eligibility: string } }) => ({
        dataset_id: row.dataset_id,
        eligibility: row.assessment.eligibility,
      }),
    ),
    scientific_status: completed.result.scientific_status,
    dossier_downloaded: true,
    scope:
      "Actual browser/API/local-catalog execution; not scientific accuracy or a qualified-fit claim.",
  };
  await writeFile(
    testInfo.outputPath("live-workflow.json"),
    JSON.stringify(receipt, null, 2),
  );
});
