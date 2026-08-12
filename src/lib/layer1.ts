import { fetchWithTimeout } from "./fetch";
import type {
  Batch,
  CapabilityAdapter,
  CapabilityAssignment,
  CheckType,
  DodItem,
  GeneratedContract,
  RawContract,
} from "./types";
import { CHECK_TYPES } from "./types";
import {
  containsKeyword,
  discoverCapabilities,
  dodFromCapabilities,
  normalizeCapabilityId,
} from "./capabilities";

export const SYSTEM_PROMPT = [
  "Kamu adalah Layer 1 — Capability Planner untuk produk agent builder non-teknis.",
  "",
  "Tugas utama:",
  "- Menerjemahkan permintaan bahasa awam menjadi Execution Contract berbasis capability.",
  "- Jangan mengasumsikan landing page kecuali intent benar-benar mengarah ke sana.",
  "- Fokus pada keputusan bisnis dan pola kemampuan yang dibutuhkan.",
  "- Tidak bertanya tentang framework, database, bahasa pemrograman, atau arsitektur.",
  "- Tidak menambahkan fitur yang tidak diminta user.",
  "",
  "Pilih adapter capability hanya dari daftar berikut:",
  "- page",
  "- form",
  "- list",
  "- gallery",
  "- proof",
  "- custom",
  "",
  "Jika intent benar-benar baru, gunakan adapter yang paling mendekati,",
  "lalu jelaskan goal dan params secara bisnis.",
  "",
  "Output harus JSON valid tanpa markdown, tanpa komentar, tanpa teks tambahan.",
  "",
  "Gunakan skema JSON berikut:",
  `{
  "intent": "string",
  "user_summary": "string",
  "clarification_questions": ["string"],
  "included": ["string"],
  "excluded": ["string"],
  "capabilities": [
    {
      "id": "CAP-XXX",
      "name": "string",
      "adapter": "page|form|list|gallery|proof|custom",
      "goal": "string",
      "params": {},
      "dod_ids": ["DOD-1"]
    }
  ],
  "definition_of_done": [
    {
      "id": "DOD-1",
      "description": "string",
      "check_type": "http_ok|dom_exists|dom_contains|form_positive_test|form_negative_test|constraint_absence|visual_smoke",
      "selector": "selector CSS opsional",
      "contains": "teks opsional",
      "min_count": 0
    }
  ],
  "batches": [
    {
      "id": "BATCH-1",
      "name": "string",
      "goal": "string",
      "depends_on": [],
      "dod_ids": ["DOD-1"]
    }
  ]
}`,
  "",
  "Aturan penting:",
  "- user_summary harus memakai bahasa awam.",
  "- excluded harus eksplisit, misalnya: tanpa payment, tanpa login, tanpa dashboard admin.",
  "- definition_of_done harus bisa diverifikasi otomatis atau minimal bisa diperiksa sistem.",
  "- batches maksimal 3 batch besar.",
  "- clarification_questions hanya diisi jika ada keputusan bisnis yang benar-benar ambigu.",
].join("\n");

const VALID_ADAPTERS: CapabilityAdapter[] = [
  "page",
  "form",
  "list",
  "gallery",
  "proof",
  "custom",
];

const MAX_CAPABILITIES = 8;
const MAX_DOD = 30;
const MAX_CLARIFICATION_QUESTIONS = 3;
const MAX_TEXT_LENGTH = 300;
const MAX_SUMMARY_LENGTH = 900;
const MAX_SELECTOR_LENGTH = 220;
const MAX_DOD_IDS = 20;

function clampText(value: unknown, max: number): string {
  return String(value ?? "").trim().slice(0, max);
}

function normalizeCapabilities(raw: unknown, prompt: string): CapabilityAssignment[] {
  if (!Array.isArray(raw) || raw.length === 0) {
    return discoverCapabilities(prompt).slice(0, MAX_CAPABILITIES);
  }

  return raw.slice(0, MAX_CAPABILITIES).map((item, index) => {
    const candidate = item as Partial<CapabilityAssignment>;

    const adapter = VALID_ADAPTERS.includes(candidate.adapter as CapabilityAdapter)
      ? (candidate.adapter as CapabilityAdapter)
      : "custom";

    const id = normalizeCapabilityId(candidate.id || `CAP-${index + 1}`);

    const params: Record<string, string | number | boolean> = {};

    if (
      candidate.params &&
      typeof candidate.params === "object" &&
      !Array.isArray(candidate.params)
    ) {
      for (const [key, value] of Object.entries(candidate.params).slice(0, 10)) {
        if (typeof value === "string") {
          params[key] = value.slice(0, MAX_TEXT_LENGTH);
        } else if (typeof value === "number" || typeof value === "boolean") {
          params[key] = value;
        }
      }
    }

    return {
      id,
      name: clampText(candidate.name, 120) || `Capability ${index + 1}`,
      adapter,
      goal: clampText(candidate.goal, MAX_TEXT_LENGTH),
      params,
      dod_ids: Array.isArray(candidate.dod_ids)
        ? candidate.dod_ids.slice(0, MAX_DOD_IDS).map(String)
        : [],
    };
  });
}

function normalizeDod(raw: unknown, capabilities: CapabilityAssignment[]): DodItem[] {
  const baseDod = dodFromCapabilities(capabilities);

  if (!Array.isArray(raw) || raw.length === 0) {
    return baseDod.slice(0, MAX_DOD);
  }

  const normalized = raw.slice(0, MAX_DOD).map((item, index) => {
    const d = item as Partial<DodItem>;

    const checkType = CHECK_TYPES.includes(d.check_type as CheckType)
      ? (d.check_type as CheckType)
      : "visual_smoke";

    const minCount =
      typeof d.min_count === "number" && Number.isFinite(d.min_count)
        ? Math.min(Math.max(Math.floor(d.min_count), 0), 100)
        : undefined;

    return {
      id: clampText(d.id, 80) || `DOD-${index + 1}`,
      description: clampText(d.description, 220) || "Kondisi selesai belum lengkap.",
      check_type: checkType,
      selector:
        typeof d.selector === "string"
          ? d.selector.slice(0, MAX_SELECTOR_LENGTH)
          : undefined,
      contains:
        typeof d.contains === "string"
          ? d.contains.slice(0, MAX_SELECTOR_LENGTH)
          : undefined,
      min_count: minCount,
    };
  });

  const byId = new Map<string, DodItem>();

  for (const d of normalized) {
    byId.set(d.id, d);
  }

  for (const d of baseDod) {
    if (!byId.has(d.id)) {
      byId.set(d.id, d);
    }
  }

  return Array.from(byId.values()).slice(0, MAX_DOD);
}

function makeBatches(capabilities: CapabilityAssignment[], dod: DodItem[]): Batch[] {
  const coreIds = capabilities.filter((c) => c.adapter === "page").flatMap((c) => c.dod_ids);
  const domainIds = capabilities.filter((c) => c.adapter !== "page").flatMap((c) => c.dod_ids);
  const constraint = dod.find((d) => d.check_type === "constraint_absence");

  const batches: Batch[] = [
    {
      id: "BATCH-1",
      name: "Permukaan & struktur inti",
      goal: "Membuat output dasar yang dapat dibuka dan dibaca.",
      depends_on: [],
      dod_ids: coreIds.length > 0 ? coreIds : dod.slice(0, 2).map((d) => d.id),
    },
    {
      id: "BATCH-2",
      name: "Kemampuan domain",
      goal: "Membuat kemampuan spesifik yang diminta oleh intent.",
      depends_on: ["BATCH-1"],
      dod_ids: domainIds,
    },
    {
      id: "BATCH-3",
      name: "Verifikasi akhir",
      goal: "Memastikan tidak ada kemampuan berbahaya atau tidak diminta.",
      depends_on: ["BATCH-1", "BATCH-2"],
      dod_ids: constraint ? [constraint.id] : [],
    },
  ];

  return batches.filter((b) => b.dod_ids.length > 0);
}

function composeContract(
  prompt: string,
  capabilities: CapabilityAssignment[],
  source: "mock" | "llm",
  raw?: RawContract,
): GeneratedContract {
  const safeCapabilities = capabilities.map((cap) => ({
    ...cap,
    dod_ids: cap.dod_ids.length > 0 ? cap.dod_ids : dodFromCapabilities([cap]).map((d) => d.id),
  }));

  const baseDod = dodFromCapabilities(safeCapabilities);

  const constraintDod: DodItem = {
    id: "DOD-CONSTRAINT",
    description: "Tidak ada elemen login/payment/fitur berbahaya yang tidak diminta.",
    check_type: "constraint_absence",
  };

  let dod = [...baseDod, constraintDod];

  if (raw?.definition_of_done) {
    dod = normalizeDod(raw.definition_of_done, safeCapabilities);

    if (!dod.some((d) => d.id === constraintDod.id)) {
      dod.push(constraintDod);
    }
  }

  const included = safeCapabilities.map((c) => `${c.name}: ${c.goal}`);

  const excluded = [
    "Payment gateway / pembayaran online",
    "Login atau register",
    "Dashboard admin",
    "Database kompleks",
    "Integrasi pihak ketiga yang tidak diminta",
    "Eksekusi kode berbahaya",
  ];

  const shortPrompt = prompt.slice(0, 220).trim();
  const capNames = safeCapabilities.map((c) => c.name).join(", ");

  const intent = raw?.intent || `Menyelesaikan intent: ${shortPrompt}`;

  const userSummary = raw?.user_summary
    ? clampText(raw.user_summary, MAX_SUMMARY_LENGTH)
    : [
        `Permintaan awal: "${shortPrompt}".`,
        `Mesin mengidentifikasi kemampuan: ${capNames}.`,
        `Setiap kemampuan diturunkan menjadi DoD yang dapat diverifikasi.`,
        `Fitur berbahaya/over-scope sengaja tidak dibuat.`,
      ].join(" ");

  return {
    intent,
    user_summary: userSummary,
    clarification_questions: Array.isArray(raw?.clarification_questions)
      ? raw
          .clarification_questions!.slice(0, MAX_CLARIFICATION_QUESTIONS)
          .map((q) => clampText(q, MAX_TEXT_LENGTH))
          .filter(Boolean)
      : [],
    clarification_answers: [],
    included,
    excluded,
    capabilities: safeCapabilities,
    definition_of_done: dod,
    batches: makeBatches(safeCapabilities, dod),
    source,
  };
}

export function createMockContract(prompt: string): GeneratedContract {
  const capabilities = discoverCapabilities(prompt);
  return composeContract(prompt, capabilities, "mock");
}

export function normalizeContract(
  raw: RawContract,
  prompt: string,
  source: "mock" | "llm",
): GeneratedContract {
  const capabilities = normalizeCapabilities(raw.capabilities, prompt);
  return composeContract(prompt, capabilities, source, raw);
}

export function extractJson(text: string): string {
  const fenced = text.match(/```(?:json)?\s*([\s\S]*?)```/i);
  const candidate = fenced?.[1] || text || "";

  const start = candidate.indexOf("{");
  const end = candidate.lastIndexOf("}");

  if (start >= 0 && end > start) {
    return candidate.slice(start, end + 1);
  }

  return candidate;
}

export async function generateContractWithLLM(prompt: string): Promise<GeneratedContract> {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  const model = process.env.OPENAI_MODEL || "gpt-4o-mini";

  if (!apiKey) {
    throw new Error("OPENAI_API_KEY belum diisi.");
  }

  const response = await fetchWithTimeout(
    `${baseUrl}/chat/completions`,
    {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify({
        model,
        temperature: 0.2,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          {
            role: "user",
            content: `Permintaan user:\n${prompt}\nBuat Execution Contract berbasis capability sekarang.`,
          },
        ],
      }),
    },
    30_000,
  );

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`LLM error ${response.status}: ${errText.slice(0, 200)}`);
  }

  const data = await response.json();
  const content: string = data.choices?.[0]?.message?.content || "";

  const raw = JSON.parse(extractJson(content)) as RawContract;

  return normalizeContract(raw, prompt, "llm");
}

export const MOCK_SUPPORTED_SCENARIOS = [
  "jasa fotografi / galeri / event",
  "katering / makanan / kue / toko sederhana",
  "kursus / les / tutor dengan form kontak atau jadwal",
  "SOP / checklist / kebijakan internal",
  "form pengajuan / permintaan internal",
  "papan pengumuman / katalog sederhana",
] as const;

export type MockScenario =
  | "photography"
  | "food-business"
  | "education"
  | "sop-checklist"
  | "internal-form"
  | "announcement"
  | "generic";

export function classifyMockPrompt(prompt: string): MockScenario | null {
  const text = prompt.toLowerCase();

  const has = (keywords: string[]) => keywords.some((keyword) => containsKeyword(text, keyword));

  if (has(["foto", "fotografi", "photography", "wedding", "pernikahan", "galeri", "portofolio fotografer"])) {
    return "photography";
  }

  if (has(["katering", "catering", "makanan", "kuliner", "kue", "bakery", "toko", "menu", "dapur", "restoran"])) {
    return "food-business";
  }

  if (has(["kursus", "kelas", "les", "pelatihan", "course", "privat", "tutor", "bimbel", "belajar"])) {
    return "education";
  }

  if (has(["sop", "checklist", "kebijakan", "onboarding", "prosedur"])) {
    return "sop-checklist";
  }

  if (has(["pengajuan", "form", "permintaan", "aset", "internal", "gudang"])) {
    return "internal-form";
  }

  if (has(["pengumuman", "papan", "notice", "katalog", "daftar"])) {
    return "announcement";
  }

  if (prompt.trim().length > 0) return "generic";

  return null;
}
