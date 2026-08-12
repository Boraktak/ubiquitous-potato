import type {
  CapabilityAdapter,
  CapabilityAssignment,
  ClarificationAnswer,
  DodItem,
} from "./types";

export interface CapabilityPattern {
  adapter: CapabilityAdapter;
  id: string;
  name: string;
  goal: string;
  keywords: string[];
  promptRules: string[];
  makeDod: (capId: string) => DodItem[];
}

export function normalizeCapabilityId(input: string): string {
  return (
    input
      .toUpperCase()
      .replace(/[^A-Z0-9]+/g, "-")
      .replace(/^-+|-+$/g, "") || "CAP-CUSTOM"
  );
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

export function containsKeyword(text: string, keyword: string): boolean {
  const pattern = escapeRegex(keyword.toLowerCase());
  const regex = new RegExp(`\\b${pattern}\\b`, "i");
  return regex.test(text);
}

export const CAPABILITY_PATTERNS: CapabilityPattern[] = [
  {
    adapter: "page",
    id: "cap-page",
    name: "Permukaan web",
    goal: "Menyajikan halaman yang dapat dibuka dan dibaca.",
    keywords: [
      "halaman",
      "page",
      "web",
      "website",
      "situs",
      "landing",
      "profile",
      "portofolio",
      "portfolio",
      "beranda",
      "dashboard",
      "papan",
      "tampilan",
    ],
    promptRules: [
      "Harus menghasilkan dokumen HTML lengkap.",
      "Harus ada tepat satu elemen <h1> sebagai judul utama.",
      "Konten tidak boleh kosong.",
    ],
    makeDod: (capId) => [
      {
        id: `${capId}-HTTP`,
        description: "Halaman dapat dibuka tanpa error (HTTP 200).",
        check_type: "http_ok",
      },
      {
        id: `${capId}-H1`,
        description: "Halaman memiliki judul utama.",
        check_type: "dom_exists",
        selector: "h1",
      },
      {
        id: `${capId}-SMOKE`,
        description: "Halaman tidak kosong dan ter-render.",
        check_type: "visual_smoke",
      },
    ],
  },
  {
    adapter: "form",
    id: "cap-form",
    name: "Penangkapan input",
    goal: "Mengumpulkan data, permintaan, atau pendaftaran dari pengguna.",
    keywords: [
      "form",
      "kontak",
      "hubungi",
      "daftar",
      "registrasi",
      "lead",
      "permintaan",
      "penawaran",
      "booking",
      "jadwal",
      "cek ketersediaan",
      "tanya",
      "konsultasi",
      "pengajuan",
      "input",
      "isi",
    ],
    promptRules: [
      "Harus ada SATU <form> dengan action=/api/projects/{projectId}/preview dan method=post.",
      "Form harus memiliki minimal 2 field required.",
      "Setelah submit sukses, tampilkan pesan terima kasih/sukses.",
    ],
    makeDod: (capId) => [
      {
        id: `${capId}-FORM`,
        description: "Terdapat form yang bisa diisi.",
        check_type: "dom_exists",
        selector: "form",
      },
      {
        id: `${capId}-NEG`,
        description: "Form menolak pengiriman jika field wajib kosong.",
        check_type: "form_negative_test",
      },
      {
        id: `${capId}-POS`,
        description: "Form valid berhasil menyimpan data uji.",
        check_type: "form_positive_test",
      },
    ],
  },
  {
    adapter: "list",
    id: "cap-list",
    name: "Daftar item",
    goal: "Menampilkan daftar item, langkah, paket, kebijakan, atau checklist.",
    keywords: [
      "daftar",
      "list",
      "menu",
      "harga",
      "paket",
      "layanan",
      "katalog",
      "produk",
      "pricelist",
      "price list",
      "langkah",
      "checklist",
      "sop",
      "kebijakan",
      "aturan",
    ],
    promptRules: [
      "Harus menampilkan daftar dalam <ul>, <ol>, atau <table>.",
      "Item daftar harus relevan dengan intent pengguna.",
    ],
    makeDod: (capId) => [
      {
        id: `${capId}-LIST`,
        description: "Daftar item tampil.",
        check_type: "dom_exists",
        selector: "ul, ol, table",
        min_count: 1,
      },
    ],
  },
  {
    adapter: "gallery",
    id: "cap-gallery",
    name: "Galeri media",
    goal: "Menampilkan koleksi gambar atau media.",
    keywords: [
      "galeri",
      "gallery",
      "foto",
      "gambar",
      "portofolio",
      "portfolio",
      "karya",
      "dokumentasi",
      "visual",
    ],
    promptRules: [
      "Harus menampilkan minimal 3 elemen <img>.",
      "Gambar boleh memakai placeholder SVG/data URI.",
    ],
    makeDod: (capId) => [
      {
        id: `${capId}-GALLERY`,
        description: "Galeri tampil dengan minimal 3 gambar.",
        check_type: "dom_exists",
        selector: "img",
        min_count: 3,
      },
    ],
  },
  {
    adapter: "proof",
    id: "cap-proof",
    name: "Bukti sosial",
    goal: "Menampilkan testimoni, ulasan, atau bukti sosial lainnya.",
    keywords: [
      "testimoni",
      "testimony",
      "ulasan",
      "review",
      "kata mereka",
      "apresiasi",
      "feedback",
    ],
    promptRules: [
      "Tampilkan minimal satu blok testimoni/ulasan.",
      "Gunakan blockquote, figure, atau card testimoni.",
    ],
    makeDod: (capId) => [
      {
        id: `${capId}-PROOF`,
        description: "Terdapat bukti sosial/testimoni.",
        check_type: "dom_exists",
        selector:
          "blockquote, figure, [class*='testimoni'], [class*='testimonial'], [class*='review']",
      },
    ],
  },
  {
    adapter: "custom",
    id: "cap-custom",
    name: "Kemampuan khusus",
    goal: "Memenuhi kebutuhan unik yang tidak cocok dengan pola umum.",
    keywords: [],
    promptRules: [
      "Render kemampuan khusus secara eksplisit.",
      "Jelaskan tujuan kemampuan ini dalam konten halaman.",
      "Jangan menambahkan fitur berbahaya yang tidak diminta.",
    ],
    makeDod: (capId) => [
      {
        id: `${capId}-CUSTOM`,
        description: "Kemampuan khusus dirender dan tidak kosong.",
        check_type: "visual_smoke",
      },
    ],
  },
];

function toAssignment(pattern: CapabilityPattern, prompt: string): CapabilityAssignment {
  const id = normalizeCapabilityId(pattern.id);
  const dod = pattern.makeDod(id);

  return {
    id,
    name: pattern.name,
    adapter: pattern.adapter,
    goal: pattern.goal,
    params: {
      source_prompt: prompt.slice(0, 220),
    },
    dod_ids: dod.map((d) => d.id),
  };
}

export function discoverCapabilities(
  prompt: string,
  answers: ClarificationAnswer[] = [],
): CapabilityAssignment[] {
  const text = [prompt, ...answers.map((a) => a.answer)].join("\n").toLowerCase();

  const excludedAdapters = new Set<CapabilityAdapter>();

  const negationPairs: Array<[RegExp, CapabilityAdapter]> = [
    [/\b(tanpa|without|no)\s+form\b/, "form"],
    [/\b(tanpa|without|no)\s+(galeri|gallery)\b/, "gallery"],
    [/\b(tanpa|without|no)\s+(testimoni|review|ulasan)\b/, "proof"],
    [/\b(tanpa|without|no)\s+(daftar|list|menu|checklist)\b/, "list"],
  ];

  for (const [re, adapter] of negationPairs) {
    if (re.test(text)) excludedAdapters.add(adapter);
  }

  const selected = new Map<string, CapabilityPattern>();

  const page = CAPABILITY_PATTERNS.find((p) => p.id === "cap-page");
  if (page) selected.set(page.id, page);

  for (const pattern of CAPABILITY_PATTERNS) {
    if (pattern.id === "cap-page" || pattern.id === "cap-custom") continue;
    if (excludedAdapters.has(pattern.adapter)) continue;

    if (pattern.keywords.some((k) => containsKeyword(text, k))) {
      selected.set(pattern.id, pattern);
    }
  }

  if (selected.size === 1) {
    const custom = CAPABILITY_PATTERNS.find((p) => p.id === "cap-custom");
    if (custom) selected.set(custom.id, custom);
  }

  return Array.from(selected.values()).map((p) => toAssignment(p, prompt));
}

function makeDodForCapability(cap: CapabilityAssignment): DodItem[] {
  const pattern = CAPABILITY_PATTERNS.find((p) => normalizeCapabilityId(p.id) === cap.id);

  if (pattern) {
    return pattern.makeDod(cap.id);
  }

  switch (cap.adapter) {
    case "page":
      return [
        {
          id: `${cap.id}-HTTP`,
          description: "Halaman dapat dibuka tanpa error (HTTP 200).",
          check_type: "http_ok",
        },
        {
          id: `${cap.id}-H1`,
          description: "Halaman memiliki judul utama.",
          check_type: "dom_exists",
          selector: "h1",
        },
        {
          id: `${cap.id}-SMOKE`,
          description: "Halaman tidak kosong dan ter-render.",
          check_type: "visual_smoke",
        },
      ];
    case "form":
      return [
        {
          id: `${cap.id}-FORM`,
          description: "Terdapat form yang bisa diisi.",
          check_type: "dom_exists",
          selector: "form",
        },
        {
          id: `${cap.id}-NEG`,
          description: "Form menolak pengiriman jika field wajib kosong.",
          check_type: "form_negative_test",
        },
        {
          id: `${cap.id}-POS`,
          description: "Form valid berhasil menyimpan data uji.",
          check_type: "form_positive_test",
        },
      ];
    case "list":
      return [
        {
          id: `${cap.id}-LIST`,
          description: "Daftar item tampil.",
          check_type: "dom_exists",
          selector: "ul, ol, table",
          min_count: 1,
        },
      ];
    case "gallery":
      return [
        {
          id: `${cap.id}-GALLERY`,
          description: "Galeri tampil dengan minimal 3 gambar.",
          check_type: "dom_exists",
          selector: "img",
          min_count: 3,
        },
      ];
    case "proof":
      return [
        {
          id: `${cap.id}-PROOF`,
          description: "Terdapat bukti sosial/testimoni.",
          check_type: "dom_exists",
          selector:
            "blockquote, figure, [class*='testimoni'], [class*='testimonial'], [class*='review']",
        },
      ];
    default:
      return [
        {
          id: `${cap.id}-CUSTOM`,
          description: "Kemampuan khusus dirender dan tidak kosong.",
          check_type: "visual_smoke",
        },
      ];
  }
}

export function dodFromCapabilities(capabilities: CapabilityAssignment[]): DodItem[] {
  const dod: DodItem[] = [];

  for (const cap of capabilities) {
    const items = makeDodForCapability(cap);

    for (const item of items) {
      if (!dod.some((d) => d.id === item.id)) {
        dod.push(item);
      }
    }
  }

  return dod;
}

export function capabilityPromptRules(cap: CapabilityAssignment, projectId: string): string[] {
  const pattern = CAPABILITY_PATTERNS.find((p) => normalizeCapabilityId(p.id) === cap.id);
  const baseRules = pattern?.promptRules ?? [];
  const adapterRules: string[] = [];

  if (cap.adapter === "page") {
    adapterRules.push(
      "Harus menghasilkan dokumen HTML lengkap.",
      "Harus ada tepat satu elemen <h1> sebagai judul utama.",
      "Konten tidak boleh kosong.",
    );
  }

  if (cap.adapter === "form") {
    adapterRules.push(
      `Harus ada SATU <form> dengan action=/api/projects/${projectId}/preview dan method=post.`,
      "Form harus memiliki minimal 2 field required.",
      "Setelah submit sukses, tampilkan pesan terima kasih/sukses.",
    );
  }

  if (cap.adapter === "list") {
    adapterRules.push("Harus menampilkan daftar dalam <ul>, <ol>, atau <table>.");
  }

  if (cap.adapter === "gallery") {
    adapterRules.push("Harus menampilkan minimal 3 elemen <img>.");
  }

  if (cap.adapter === "proof") {
    adapterRules.push("Tampilkan minimal satu blok testimoni/ulasan.");
  }

  if (cap.adapter === "custom") {
    adapterRules.push(
      "Render kemampuan khusus secara eksplisit.",
      "Jangan menambahkan fitur berbahaya yang tidak diminta.",
    );
  }

  const rules = baseRules.length > 0 ? baseRules : adapterRules;

  return Array.from(new Set(rules.map((r) => r.replaceAll("{projectId}", projectId))));
}
