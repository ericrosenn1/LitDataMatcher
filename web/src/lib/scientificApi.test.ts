import { afterEach, describe, expect, it, vi } from "vitest";
import {
  parseRequirementValue,
  safeScientificUrl,
  scientificExportUrl,
  scientificRequest,
  scientificSourcePayload,
} from "./scientificApi";

afterEach(() => vi.unstubAllGlobals());

describe("scientific research API", () => {
  it("submits selected catalog identities once and keeps pasted text unverified", () => {
    const payload = scientificSourcePayload([
      {
        document_id: "pmid:123",
        title: "Trusted server record",
        text: "Do not resubmit cached text",
      },
      { document_id: "pmid:123", title: "Same source again" },
      { document_id: "pmid:456", text: "Another catalog record" },
      {
        id: "unverified-client-id",
        title: "Pasted source",
        text: "User supplied source text.",
        url: "https://example.org/paper",
        source_locator: "ignored extra field",
      },
    ]);
    expect(payload.document_ids).toEqual(["pmid:123", "pmid:456"]);
    expect(payload.documents).toEqual([
      {
        title: "Pasted source",
        text: "User supplied source text.",
        url: "https://example.org/paper",
      },
    ]);
  });
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
      vi.fn().mockResolvedValue(
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
      vi.fn().mockResolvedValue(
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
