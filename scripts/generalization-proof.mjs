const CAPABILITY_PATTERNS = [
  {
    adapter: "page",
    id: "cap-page",
    keywords: ["halaman", "page", "web", "website", "situs", "landing", "papan", "tampilan"],
  },
  {
    adapter: "form",
    id: "cap-form",
    keywords: [
      "form",
      "kontak",
      "hubungi",
      "daftar",
      "lead",
      "permintaan",
      "penawaran",
      "booking",
      "jadwal",
      "pengajuan",
    ],
  },
  {
    adapter: "list",
    id: "cap-list",
    keywords: ["daftar", "list", "menu", "harga", "paket", "checklist", "sop", "kebijakan"],
  },
  {
    adapter: "gallery",
    id: "cap-gallery",
    keywords: ["galeri", "gallery", "foto", "gambar", "portofolio"],
  },
  {
    adapter: "proof",
    id: "cap-proof",
    keywords: ["testimoni", "ulasan", "review"],
  },
];

function escapeRegex(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function containsKeyword(text, keyword) {
  const pattern = escapeRegex(keyword.toLowerCase());
  const regex = new RegExp(`\\b${pattern}\\b`, "i");
  return regex.test(text);
}

function discover(prompt) {
  const text = prompt.toLowerCase();
  const excluded = new Set();

  if (/\b(tanpa|without|no)\s+form\b/.test(text)) excluded.add("form");
  if (/\b(tanpa|without|no)\s+(galeri|gallery)\b/.test(text)) excluded.add("gallery");
  if (/\b(tanpa|without|no)\s+(testimoni|review|ulasan)\b/.test(text)) excluded.add("proof");
  if (/\b(tanpa|without|no)\s+(daftar|list|menu|checklist)\b/.test(text)) excluded.add("list");

  const selected = new Set(["page"]);

  for (const p of CAPABILITY_PATTERNS) {
    if (p.adapter === "page") continue;
    if (excluded.has(p.adapter)) continue;
    if (p.keywords.some((k) => containsKeyword(text, k))) selected.add(p.adapter);
  }

  if (selected.size === 1) selected.add("custom");

  return [...selected];
}

const CASES = [
  {
    name: "SOP checklist tanpa form",
    prompt: "Saya butuh halaman checklist SOP onboarding karyawan baru tanpa form.",
    expect: { has: ["page", "list"], not: ["form"] },
  },
  {
    name: "Form pengajuan internal",
    prompt: "Buat form pengajuan perbaikan aset internal untuk tim gudang.",
    expect: { has: ["page", "form"], not: [] },
  },
  {
    name: "Papan pengumuman",
    prompt: "Buat papan pengumuman internal untuk menampilkan kebijakan baru perusahaan.",
    expect: { has: ["page"], not: ["form", "gallery"] },
  },
  {
    name: "Fotografi klasik",
    prompt: "Landing page jasa fotografi pernikahan dengan galeri dan testimoni serta form cek tanggal.",
    expect: { has: ["page", "gallery", "proof", "form"], not: [] },
  },
  {
    name: "Katalog + form",
    prompt: "Katalog layanan konsultasi dengan daftar paket dan form permintaan.",
    expect: { has: ["page", "list", "form"], not: [] },
  },
  {
    name: "Intent benar-benar asing",
    prompt: "Saya mau sesuatu yang belum pernah ada namanya: ringkasan keputusan rapat.",
    expect: { has: ["page", "custom"], not: ["form", "gallery"] },
  },
];

let failed = 0;

for (const c of CASES) {
  const got = discover(c.prompt);
  const missing = c.expect.has.filter((a) => !got.includes(a));
  const unexpected = c.expect.not.filter((a) => got.includes(a));
  const ok = missing.length === 0 && unexpected.length === 0;

  if (!ok) failed++;

  console.log(
    `${ok ? "PASS" : "FAIL"}  ${c.name}\n  prompt: ${c.prompt}\n  got: ${got.join(",")}\n  missing: ${missing.join(",") || "-"}  unexpected: ${unexpected.join(",") || "-"}`,
  );
}

console.log(`\n${CASES.length - failed}/${CASES.length} passed`);
process.exit(failed > 0 ? 1 : 0);
