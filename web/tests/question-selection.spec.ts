import { expect, test } from "@playwright/test";

test("extract first, select verified reference, and keep edited wording explicit", async ({
  page,
}) => {
  const requests: { path: string; body: Record<string, unknown> }[] = [];
  const question = {
    question_id: "rq_" + "a".repeat(24),
    question: "Further research is needed on glucose uptake in humans.",
    source_document_id: "PMID:123",
    source_locator: "https://example.org/paper#chars=0:55",
    category: "SOURCE_STATED_UNRESOLVED",
  };
  const extractedId = "a".repeat(32);
  let submissions = 0;
  let modified = false;
  await page.route("**/api/**", async (route) => {
    const request = route.request();
    const path = new URL(request.url()).pathname;
    const body = request.method() === "POST" ? request.postDataJSON() : {};
    requests.push({ path, body });
    let response: unknown;
    if (path === "/api/health")
      response = {
        mode: "RUNNING",
        worker_active: true,
        model_available: false,
        semantic_available: false,
        sources: { literature: ["pubmed"], datasets: ["geo"] },
      };
    else if (path === "/api/jobs" && request.method() === "GET")
      response = { jobs: [] };
    else if (path === "/api/jobs") {
      submissions += 1;
      modified = body.question !== question.question;
      response = {
        id: submissions === 1 ? extractedId : "b".repeat(32),
        status: "QUEUED",
      };
    } else if (path === "/api/requirements")
      response = {
        compilation_status: "COMPLETE",
        matching_requirements: body.requirements,
        question_selection: { ...question, question: body.question },
      };
    else if (path.startsWith("/api/jobs/")) {
      const extraction = path.endsWith(extractedId);
      response = {
        id: extraction ? extractedId : "b".repeat(32),
        status: "SUCCEEDED",
        result: {
          workflow_mode: extraction ? "questions" : "full",
          processing_status: "COMPLETE",
          question: extraction
            ? null
            : {
                ...question,
                selection_reference: {
                  job_id: extractedId,
                  question_id: question.question_id,
                },
                ...(modified
                  ? {
                      question: "Does glucose uptake change with age?",
                      derived_from: question,
                    }
                  : {}),
              },
          questions: {
            questions: [question],
            coverage: {
              processed_documents: extraction ? 1 : 0,
              reused_documents: extraction ? 0 : 1,
            },
          },
          candidates: [],
          addressed_evidence: extraction
            ? undefined
            : {
                status: "POSSIBLY_ADDRESSED_NEEDS_REVIEW",
                searched_document_ids: ["PMID:123"],
                candidates: [
                  {
                    evidence_span: {
                      text: "We observed glucose uptake in humans.",
                    },
                    source_locator: "https://example.org/result",
                  },
                ],
              },
        },
      };
    } else response = {};
    await route.fulfill({
      status: 200,
      contentType: "application/json",
      body: JSON.stringify(response),
    });
  });
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
  await page
    .getByLabel("Literature search", { exact: true })
    .fill("glucose uptake");
  await expect(
    page.getByRole("button", { name: "Find datasets", exact: true }),
  ).toBeDisabled();
  await page
    .getByRole("button", { name: "Find questions", exact: true })
    .click();
  await expect(
    page.getByRole("heading", { name: "Source questions", exact: true }),
  ).toBeVisible();
  const extraction = requests.find(
    (row) => row.path === "/api/jobs" && row.body.workflow_mode === "questions",
  )!.body;
  expect(extraction.dataset_sources).toEqual([]);
  expect(extraction.inspection_limit).toBe(0);
  expect(extraction.question).toBe("");
  expect(extraction.question_reference).toBeUndefined();
  await expect(page.getByLabel("Compatibility filter")).toHaveCount(0);
  await page
    .getByRole("button", { name: "Use this question", exact: true })
    .click();
  await expect(
    page.getByText("Selected source passage:", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Add requirement", exact: true })
    .click();
  await page.getByLabel("Requirement field 1").selectOption("paired");
  await page.getByLabel("Expected value 1").fill("false");
  await page.getByLabel("Required", { exact: true }).uncheck();
  await page
    .getByRole("button", { name: "Preview requirements", exact: true })
    .click();
  const reference = { job_id: extractedId, question_id: question.question_id };
  expect(
    requests.filter((row) => row.path === "/api/requirements").at(-1)!.body,
  ).toEqual({
    question: question.question,
    question_reference: reference,
    requirements: [{ field: "paired", expected: false, essential: false }],
  });
  await page
    .getByLabel("Research question", { exact: true })
    .fill("Does glucose uptake change with age?");
  await expect(
    page.getByText("User-modified question:", { exact: false }),
  ).toBeVisible();
  await page
    .getByRole("button", { name: "Find datasets", exact: true })
    .click();
  await expect(
    page.getByRole("heading", {
      name: "Evidence and compatibility",
      exact: true,
    }),
  ).toBeVisible();
  const launch = requests
    .filter(
      (row) => row.path === "/api/jobs" && row.body.workflow_mode === "full",
    )
    .at(-1)!.body;
  expect(launch.question_reference).toEqual(reference);
  expect(launch.question).toBe("Does glucose uptake change with age?");
  expect(launch.source_question_binding).toBeUndefined();
  expect(launch.citation).toBeUndefined();
  await expect(
    page.getByText("User-modified wording;", { exact: false }),
  ).toBeVisible();
  await page.getByText("Bounded addressing evidence", { exact: true }).click();
  await expect(
    page.getByText("We observed glucose uptake in humans.", { exact: true }),
  ).toBeVisible();
  await expect(
    page.getByText(
      "this check establishes neither an answer nor global novelty",
      { exact: false },
    ),
  ).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= window.innerWidth,
    ),
  ).toBe(true);
  await page
    .getByRole("button", { name: "Use as a new question", exact: true })
    .click();
  await page
    .getByRole("button", { name: "Preview requirements", exact: true })
    .click();
  expect(
    requests.filter((row) => row.path === "/api/requirements").at(-1)!.body
      .question_reference,
  ).toBeUndefined();
});
