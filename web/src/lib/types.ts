export interface Paper {
  id: string;
  title: string;
  abstract: string;
  doi?: string;
  url?: string;
  year?: number;
  source: string;
  isIllustrative?: boolean;
}

export interface Dataset {
  id: string;
  title: string;
  description: string;
  url: string;
  source: string;
  year?: number;
  sampleSize: number | null;
  variables: string[];
  population: string[];
  access: string;
  license: string;
  isIllustrative?: boolean;
}

export interface Opportunity {
  id: string;
  question: string;
  evidence: string;
  extractionType: string;
  paperId: string;
  datasetId: string;
  score: number;
  components: { label: string; score: number; weight: number }[];
  rationale: string[];
  caveats: string[];
  terms: string[];
}

export interface Review {
  saved: boolean;
  status: string;
  note: string;
}

export interface Workspace {
  id: string;
  name: string;
  mode: "demo" | "live" | "import";
  createdAt: string;
  papers: Paper[];
  datasets: Dataset[];
  opportunities: Opportunity[];
  warnings: string[];
  provenance?: Record<string, unknown>;
  reviews?: Record<string, Review>;
}
