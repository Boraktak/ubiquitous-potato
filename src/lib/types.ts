export type CheckType =
  | "http_ok"
  | "dom_exists"
  | "dom_contains"
  | "form_positive_test"
  | "form_negative_test"
  | "constraint_absence"
  | "visual_smoke";

export type CapabilityAdapter =
  | "page"
  | "form"
  | "list"
  | "gallery"
  | "proof"
  | "custom";

export interface CapabilityAssignment {
  id: string;
  name: string;
  adapter: CapabilityAdapter;
  goal: string;
  params: Record<string, string | number | boolean>;
  dod_ids: string[];
}

export interface DodItem {
  id: string;
  description: string;
  check_type: CheckType;
  selector?: string;
  contains?: string;
  min_count?: number;
}

export type VerificationStatus = "pass" | "fail" | "manual" | "error";

export interface VerificationResult {
  dodId: string;
  checkType: CheckType;
  status: VerificationStatus;
  detail: string;
  createdAt: string;
}

export interface VerificationSummary {
  total: number;
  pass: number;
  fail: number;
  manual: number;
  lastRunAt: string | null;
}

export interface Batch {
  id: string;
  name: string;
  goal: string;
  depends_on: string[];
  dod_ids: string[];
}

export interface RawContract {
  intent?: string;
  user_summary?: string;
  clarification_questions?: string[];
  included?: string[];
  excluded?: string[];
  capabilities?: Partial<CapabilityAssignment>[];
  definition_of_done?: Partial<DodItem>[];
  batches?: Partial<Batch>[];
}

export interface ClarificationAnswer {
  question: string;
  answer: string;
}

export interface Contract {
  intent: string;
  user_summary: string;
  clarification_questions: string[];
  clarification_answers: ClarificationAnswer[];
  included: string[];
  excluded: string[];
  capabilities: CapabilityAssignment[];
  definition_of_done: DodItem[];
  batches: Batch[];
  source: "mock" | "llm";
  confirmation_status: "pending" | "confirmed";
  confirmed_at: string | null;
}

export type GeneratedContract = Omit<Contract, "confirmation_status" | "confirmed_at">;

export type ProjectStatus =
  | "DRAFT"
  | "AWAITING_CLARIFICATION"
  | "CONTRACT_READY"
  | "CONFIRMED"
  | "RUNNING"
  | "COMPLETED"
  | "FAILED";

export type ExecutionMode = "manual" | "agent";

export interface LogEntry {
  id: string;
  message: string;
  createdAt: string;
}

export interface Delivery {
  previewUrl: string;
  summary: string;
  deliveredAt: string;
}

export interface Lead {
  id: string;
  payload: Record<string, string>;
  isTest: boolean;
  createdAt: string;
}

export interface Project {
  id: string;
  prompt: string;
  status: ProjectStatus;
  createdAt: string;
  updatedAt: string;
  contract: Contract | null;
  delivery: Delivery | null;
  logs: LogEntry[];
  verifications: Record<string, VerificationResult>;
  verificationSummary: VerificationSummary | null;
  executionMode: ExecutionMode | null;
  agentBuilt: boolean;
  artifactSource: "template" | "llm" | null;
  leadsCount: number;
}

export const CHECK_TYPES: CheckType[] = [
  "http_ok",
  "dom_exists",
  "dom_contains",
  "form_positive_test",
  "form_negative_test",
  "constraint_absence",
  "visual_smoke",
];
