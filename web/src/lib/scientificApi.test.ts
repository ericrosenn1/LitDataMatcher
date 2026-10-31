import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseRequirementValue,
  safeScientificUrl,
  scientificExportUrl,
  scientificRequest,
} from "./scientificApi";

afterEach(() => vi.unstubAllGlobals());

describe("scientific research API", () => {
  it("preserves explicit booleans and finite numeric values without treating strings as truthy flags", () => {
    expect(parseRequirementValue("false")).toBe(false);
    expect(parseRequirementValue("true")).toBe(true);
    expect(parseRequirementValue("20")).toBe(20);
    expect(parseRequirementValue("human")).toBe("human");
    expect(parseRequirementValue("001")).toBe("001");
    expect(() => parseRequirementValue(" ")).toThrow();
  });
  it("rejects unsafe evidence links and encodes job identities", () => {
    expect(safeScientificUrl("javascript:alert(1)")).toBeUndefined();
    expect(safeScientificUrl("https://example.org/evidence")).toBe(
      "https://example.org/evidence",
    );
    expect(scientificExportUrl("job/one", "dossier.md")).toBe(
      "/api/jobs/job%2Fone/dossier.md",
    );
  });
  it("surfaces missing Python API behind an HTML development server", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response("<html></html>", {
            headers: { "Content-Type": "text/html" },
          }),
        ),
    );
    await expect(scientificRequest("health")).rejects.toThrow(
      "Start the local Python server",
    );
  });
  it("retains API failures instead of returning a success-shaped empty result", async () => {
    vi.stubGlobal(
      "fetch",
      vi
        .fn()
        .mockResolvedValue(
          new Response(JSON.stringify({ error: "Missing assay evidence" }), {
            status: 422,
            headers: { "Content-Type": "application/json" },
          }),
        ),
    );
    await expect(
      scientificRequest("requirements", { question: "test" }),
    ).rejects.toThrow("Missing assay evidence");
  });
});
