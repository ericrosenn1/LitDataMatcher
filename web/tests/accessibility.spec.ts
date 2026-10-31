import { expect, test, type Page } from "@playwright/test";
import AxeBuilder from "@axe-core/playwright";

async function scan(page: Page) {
  const result = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa"])
    .analyze();
  expect(
    result.violations.map(({ id, nodes }) => ({
      id,
      elements: nodes.map((node) => node.target),
    })),
  ).toEqual([]);
}

for (const label of [
  "Overview",
  "Discover",
  "Dataset catalog",
  "Saved library",
  "How it works",
]) {
  test(`${label} has no detected WCAG A/AA violations`, async ({ page }) => {
    await page.goto("/");
    await expect(
      page.getByRole("heading", { name: "A workspace for what’s next." }),
    ).toBeVisible();
    if (label !== "Overview") {
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
    await scan(page);
  });
}

test("review dialog has no detected WCAG A/AA violations", async ({ page }) => {
  await page.goto("/");
  await page
    .getByRole("button", { name: "Review match", exact: true })
    .first()
    .click();
  await expect(
    page.getByRole("dialog", { name: "Connection review" }),
  ).toBeVisible();
  await scan(page);
});
