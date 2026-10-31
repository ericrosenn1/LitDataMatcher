import { afterEach, describe, expect, it, vi } from "vitest";
import {
  analyzeText,
  demoWorkspace,
  exportWorkspace,
  importWorkspace,
  safeUrl,
  searchWorkspace,
} from "./engine";

afterEach(() => vi.restoreAllMocks());

describe("demonstration provenance", () => {
  it("contains actual pipeline output with valid references and illustrative labels", () => {
    expect(demoWorkspace.opportunities.length).toBeGreaterThan(0);
    expect(demoWorkspace.papers.every((paper) => paper.isIllustrative)).toBe(
      true,
    );
    expect(
      demoWorkspace.papers.every((paper) => !paper.doi && !paper.url),
    ).toBe(true);
    expect(
      demoWorkspace.datasets.every((dataset) => dataset.sampleSize === null),
    ).toBe(true);
    for (const match of demoWorkspace.opportunities) {
      expect(
        demoWorkspace.papers.some(
          (paper) =>
            paper.id === match.paperId &&
            paper.abstract.includes(match.evidence),
        ),
      ).toBe(true);
      expect(
        demoWorkspace.datasets.some(
          (dataset) => dataset.id === match.datasetId,
        ),
      ).toBe(true);
      expect(match.score).toBeGreaterThanOrEqual(0);
      expect(match.score).toBeLessThanOrEqual(100);
    }
  });

  it("preserves raw Python scores in its provenance", () => {
    const raw = demoWorkspace.provenance?.originalMatches as {
      match_id: string;
      score: { combined: number };
    }[];
    for (const match of demoWorkspace.opportunities) {
      expect(match.score).toBeCloseTo(
        raw.find((item) => item.match_id === match.id)!.score.combined * 100,
        1,
      );
    }
  });
});

describe("safe workspace imports and exports", () => {
  it("round-trips full workspaces and review notes without loss", () => {
    const first = demoWorkspace.opportunities[0].id;
    const reviews = {
      [first]: {
        saved: true,
        status: "relevant",
        note: "Needs independent validation.",
      },
    };
    const decoded = importWorkspace(
      exportWorkspace(demoWorkspace, reviews, "json"),
      "workspace.json",
    );
    expect(decoded.opportunities).toEqual(demoWorkspace.opportunities);
    expect(decoded.reviews).toEqual(reviews);
    expect(decoded.provenance).toEqual(demoWorkspace.provenance);
    expect(decoded.mode).toBe("import");
  });

  it("imports unmodified Python matches and retains source scores", () => {
    const raw = demoWorkspace.provenance?.originalMatches as unknown[];
    const imported = importWorkspace(
      raw.map((value) => JSON.stringify(value)).join("\n"),
      "matches.jsonl",
    );
    expect(imported.opportunities).toHaveLength(
      demoWorkspace.opportunities.length,
    );
    expect(imported.opportunities[0].score).toBe(
      demoWorkspace.opportunities[0].score,
    );
    expect(
      imported.papers.every(
        (paper) => !paper.doi || !paper.doi.startsWith("10.0000/"),
      ),
    ).toBe(true);
  });

  it("accepts BOM and a single Python match", () => {
    const raw = demoWorkspace.provenance?.originalMatches as unknown[];
    expect(
      importWorkspace("\uFEFF" + JSON.stringify(raw[0]), "match.json")
        .opportunities,
    ).toHaveLength(1);
  });

  it("hides assumed counts for all five curated collections without mislabeling real papers", () => {
    const originals = demoWorkspace.provenance?.originalMatches as Record<
      string,
      unknown
    >[];
    const copies = structuredClone(originals);
    for (const copy of copies) {
      const question = copy.question as { evidence: { doi: string }[] };
      question.evidence[0].doi = "10.1234/real-paper";
    }
    const imported = importWorkspace(JSON.stringify(copies), "matches.json");
    expect(imported.datasets).toHaveLength(5);
    expect(
      imported.datasets.every(
        (dataset) => dataset.isIllustrative && dataset.sampleSize === null,
      ),
    ).toBe(true);
    expect(
      imported.papers.every(
        (paper) => !paper.isIllustrative && paper.doi === "10.1234/real-paper",
      ),
    ).toBe(true);
    expect(imported.provenance?.rawMatches).toEqual(copies);
  });

  it("preserves reported counts for external datasets and suppresses synthetic DOI links", () => {
    const originals = demoWorkspace.provenance?.originalMatches as Record<
      string,
      unknown
    >[];
    const copy = structuredClone(originals[0]);
    const dataset = copy.dataset as { dataset_id: string; sample_size: number };
    dataset.dataset_id = "external-accession-123";
    dataset.sample_size = 53;
    const question = copy.question as { evidence: { doi: string }[] };
    question.evidence[0].doi = "10.0000/demo-ibd-microbiome";
    const imported = importWorkspace(JSON.stringify(copy), "match.json");
    expect(imported.datasets[0].sampleSize).toBe(53);
    expect(imported.datasets[0].isIllustrative).toBe(false);
    expect(imported.papers[0].isIllustrative).toBe(true);
    expect(imported.papers[0].doi).toBeUndefined();
    expect(imported.papers[0].url).toBeUndefined();
  });

  it("imports the Python formula exactly, including its separately subtracted penalty", () => {
    const originals = demoWorkspace.provenance?.originalMatches as Record<
      string,
      unknown
    >[];
    const imported = importWorkspace(JSON.stringify(originals), "matches.json");
    for (const match of imported.opportunities) {
      expect(match.components.map((component) => component.weight)).toEqual([
        0.32, 0.28, 0.16, 0.1, 0.08, 0.06, 0.18,
      ]);
      expect(match.components[6].label).toContain("subtracted");
      const positive = match.components
        .slice(0, 6)
        .reduce(
          (sum, component) => sum + component.weight * component.score,
          0,
        );
      const penalty = match.components[6].weight * match.components[6].score;
      expect(match.score).toBeCloseTo(positive - penalty, 0);
    }
  });

  it("keeps multiple evidence excerpts from the same imported source document", () => {
    const originals = demoWorkspace.provenance?.originalMatches as Record<
      string,
      unknown
    >[];
    const first = structuredClone(originals[0]);
    const second = structuredClone(originals[0]);
    second.match_id = "a_second_match";
    const question = second.question as { evidence: { text: string }[] };
    question.evidence[0].text =
      "A second independently extracted source sentence.";
    const imported = importWorkspace(
      JSON.stringify([first, second]),
      "matches.json",
    );
    expect(imported.papers).toHaveLength(1);
    expect(imported.papers[0].abstract).toContain(
      imported.opportunities[0].evidence,
    );
    expect(imported.papers[0].abstract).toContain(
      imported.opportunities[1].evidence,
    );
  });

  it("rejects malformed files and dangling references", () => {
    expect(() => importWorkspace("not json", "bad.txt")).toThrow(
      "not valid JSON",
    );
    const broken = structuredClone(demoWorkspace);
    broken.opportunities[0].paperId = "missing";
    expect(() => importWorkspace(JSON.stringify(broken), "bad.json")).toThrow(
      "missing paper",
    );
  });

  it("rejects duplicate identifiers, non-finite and out-of-range scores", () => {
    const duplicate = structuredClone(demoWorkspace);
    duplicate.papers.push(duplicate.papers[0]);
    expect(() =>
      importWorkspace(JSON.stringify(duplicate), "bad.json"),
    ).toThrow("Duplicate");
    const bad = structuredClone(demoWorkspace);
    bad.opportunities[0].score = 101;
    expect(() => importWorkspace(JSON.stringify(bad), "bad.json")).toThrow(
      "between",
    );
    bad.opportunities[0].score = NaN;
    expect(() => importWorkspace(JSON.stringify(bad), "bad.json")).toThrow(
      "finite",
    );
  });

  it("removes navigable non-http URLs from imported content", () => {
    const unsafe = structuredClone(demoWorkspace);
    unsafe.datasets[0].url = "javascript:alert(1)";
    const imported = importWorkspace(JSON.stringify(unsafe), "workspace.json");
    expect(imported.datasets[0].url).toBe("");
    expect(safeUrl("file:///etc/passwd")).toBe("");
    expect(safeUrl("https://example.org/data")).toBe(
      "https://example.org/data",
    );
  });

  it("quotes fields and neutralizes spreadsheet formulas in CSV exports", () => {
    const copy = structuredClone(demoWorkspace);
    copy.opportunities = [
      { ...copy.opportunities[0], question: '=HYPERLINK("evil")' },
    ];
    const csv = exportWorkspace(
      copy,
      {
        [copy.opportunities[0].id]: {
          saved: true,
          status: "review",
          note: "+SUM(1,2)\nsecond line",
        },
      },
      "csv",
    );
    expect(csv).toContain('"\'=HYPERLINK(""evil"")"');
    expect(csv).toContain('"\'+SUM(1,2)\nsecond line"');
    expect(csv.startsWith("\uFEFF")).toBe(true);
  });
});

describe("local source-text extraction", () => {
  it("retains exact supporting text and performs no network access", () => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");
    const source =
      "Human microbiome research benefits from longitudinal sampling. Can microbiome composition predict disease activity after antibiotics?";
    const workspace = analyzeText(
      "Microbiome disease activity",
      source,
      demoWorkspace,
    );
    expect(workspace.opportunities.length).toBeGreaterThan(0);
    expect(workspace.papers[0].abstract).toBe(source);
    expect(workspace.opportunities[0].evidence).toBe(
      "Can microbiome composition predict disease activity after antibiotics?",
    );
    expect(fetchSpy).not.toHaveBeenCalled();
  });

  it("labels a topic match when no explicit question is present", () => {
    const workspace = analyzeText(
      "Microbiome",
      "The human microbiome contains microbial communities that vary with diet and disease activity.",
      demoWorkspace,
    );
    expect(workspace.opportunities[0].extractionType).toBe("topic_match");
    expect(
      workspace.opportunities[0].caveats.some((value) =>
        value.includes("No explicit"),
      ),
    ).toBe(true);
  });

  it("validates text limits and permits zero-match outcomes", () => {
    expect(() => analyzeText("Test", "Too short", demoWorkspace)).toThrow("40");
    expect(() => analyzeText("", "x".repeat(50), demoWorkspace)).toThrow(
      "title",
    );
    expect(
      analyzeText(
        "Astronomy",
        "Do distant quasars show a correlation between redshift and stellar metallicity?",
        demoWorkspace,
      ).opportunities,
    ).toHaveLength(0);
  });
});

const literatureResponse = {
  resultList: {
    result: [
      {
        id: "123",
        source: "MED",
        title: "Gut microbiome recovery",
        abstractText:
          "<p>Can human microbiome measurements predict recovery?</p>",
        pubYear: "2024",
        doi: "10.1234/example",
      },
    ],
  },
};
const dataResponse = {
  data: [
    {
      id: "10.5678/example",
      attributes: {
        doi: "10.5678/example",
        titles: [{ title: "Human microbiome measurements and recovery" }],
        descriptions: [
          { description: "<p>Longitudinal gut microbiome recovery data.</p>" },
        ],
        publicationYear: 2023,
        rightsList: [{ rights: "CC-BY-4.0" }],
      },
    },
  ],
};

describe("live metadata retrieval", () => {
  it("normalizes real API shapes without inventing samples or measured variables", async () => {
    const fetchMock = vi
      .spyOn(globalThis, "fetch")
      .mockImplementation(async (input) => {
        const url = new URL(String(input));
        return new Response(
          JSON.stringify(
            url.hostname === "api.datacite.org"
              ? dataResponse
              : literatureResponse,
          ),
          { status: 200 },
        );
      });
    const result = await searchWorkspace("gut microbiome");
    expect(fetchMock).toHaveBeenCalledTimes(2);
    expect(result.papers[0].abstract).toBe(
      "Can human microbiome measurements predict recovery?",
    );
    expect(result.datasets[0].sampleSize).toBeNull();
    expect(result.datasets[0].variables).toEqual([]);
    expect(result.opportunities.length).toBeGreaterThan(0);
    for (const match of result.opportunities)
      expect(match.score).toBeCloseTo(
        match.components.reduce(
          (sum, component) => sum + component.score * component.weight,
          0,
        ),
        1,
      );
  });

  it("preserves successful results and exposes a failed source", async () => {
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) =>
      new URL(String(input)).hostname === "api.datacite.org"
        ? new Response("rate limited", { status: 429 })
        : new Response(JSON.stringify(literatureResponse)),
    );
    const result = await searchWorkspace("gut microbiome");
    expect(result.papers).toHaveLength(1);
    expect(result.datasets).toHaveLength(0);
    expect(result.opportunities).toHaveLength(0);
    expect(result.warnings.some((value) => value.includes("429"))).toBe(true);
  });

  it("does not silently substitute demo records when both sources fail", async () => {
    vi.spyOn(globalThis, "fetch").mockRejectedValue(
      new Error("Network unavailable"),
    );
    await expect(searchWorkspace("gut microbiome")).rejects.toThrow(
      "Both metadata sources",
    );
  });

  it("keeps empty retrievals as empty results and encodes user queries", async () => {
    const urls: URL[] = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (input) => {
      urls.push(new URL(String(input)));
      return new Response(
        JSON.stringify({ data: [], resultList: { result: [] } }),
      );
    });
    const result = await searchWorkspace("diet & health");
    expect(result.papers).toHaveLength(0);
    expect(result.datasets).toHaveLength(0);
    expect(
      urls.every((url) => url.searchParams.get("query") === "diet & health"),
    ).toBe(true);
    expect(
      urls
        .find((url) => url.hostname === "api.datacite.org")
        ?.searchParams.get("resource-type-id"),
    ).toBe("dataset");
  });

  it("rejects invalid queries before contacting external services", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch");
    await expect(searchWorkspace("a")).rejects.toThrow("between");
    expect(fetchMock).not.toHaveBeenCalled();
  });

  it("honors a caller cancellation", async () => {
    const controller = new AbortController();
    controller.abort();
    vi.spyOn(globalThis, "fetch").mockRejectedValue(new Error("Aborted"));
    await expect(
      searchWorkspace("microbiome", controller.signal),
    ).rejects.toThrow("cancelled");
  });
});
