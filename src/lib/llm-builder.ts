import { fetchWithTimeout } from "./fetch";
import type { Contract } from "./types";
import { capabilityPromptRules } from "./capabilities";

function extractHtml(text: string): string {
  let t = text.trim();

  const fence = t.match(/```(?:html)?\s*([\s\S]*?)```/i);
  if (fence) t = fence[1].trim();

  const idx = t.search(/<!doctype html|<html[\s>]/i);
  if (idx > 0) t = t.slice(idx);

  return t.trim();
}

function buildPrompt(contract: Contract, prompt: string, projectId: string): string {
  const capabilities = contract.capabilities ?? [];

  const capabilityBlocks = capabilities.map((cap) => {
    const rules = capabilityPromptRules(cap, projectId);

    return [
      `- Capability: ${cap.name} (${cap.adapter})`,
      `  Goal: ${cap.goal}`,
      `  Params: ${JSON.stringify(cap.params)}`,
      ...rules.map((r) => `  Rule: ${r}`),
    ].join("\n");
  });

  const dodBlocks = contract.definition_of_done.map((d) => `- ${d.id}: ${d.description}`);
  const needsForm = capabilities.some((c) => c.adapter === "form");

  return [
    "Kamu adalah Agent Runner generic untuk produk agent builder non-teknis.",
    "",
    "Tugas: hasilkan SATU file HTML lengkap dan mandiri berdasarkan Execution Contract.",
    "",
    "Aturan KETAT:",
    "- Output HANYA dokumen HTML yang diawali <!doctype html>.",
    "- Tanpa markdown, tanpa penjelasan, tanpa teks di luar HTML.",
    "- Semua CSS harus inline di dalam <style> pada <head>.",
    "- DILARANG <link> CSS eksternal atau <script>.",
    "- Jangan mengasumsikan landing page jika intent tidak meminta landing page.",
    "- Bangun output sesuai capability yang diberikan.",
    "- Tepat satu elemen <h1>.",
    `- Jika ada capability form, WAJIB ada SATU <form> dengan action=/api/projects/${projectId}/preview dan method=post data-harness-test-safe="true".`,
    "- Jika tidak ada capability form, JANGAN membuat form.",
    "- DILARANG input password, field kartu kredit, atau elemen payment/login.",
    "- Untuk gambar, gunakan placeholder SVG/data URI bila perlu.",
    "",
    "Capabilities:",
    ...capabilityBlocks,
    "",
    "Definition of Done:",
    ...dodBlocks,
    "",
    `Permintaan user: ${prompt}`,
    `Yang dibuat: ${contract.included.join("; ")}`,
    `Yang TIDAK dibuat: ${contract.excluded.join("; ")}`,
    "",
    needsForm ? "Hasilkan HTML dengan form yang valid sekarang." : "Hasilkan HTML tanpa form sekarang.",
  ]
    .filter(Boolean)
    .join("\n");
}

export async function generateArtifactWithLLM(
  contract: Contract,
  prompt: string,
  projectId: string,
): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  const baseUrl = process.env.OPENAI_BASE_URL || "https://api.openai.com/v1";
  const model = process.env.OPENAI_MODEL || "gpt-4o-mini";

  if (!apiKey) {
    throw new Error("OPENAI_API_KEY belum diisi.");
  }

  const userPrompt = buildPrompt(contract, prompt, projectId);

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
        temperature: 0.3,
        max_tokens: 3000,
        messages: [{ role: "user", content: userPrompt }],
      }),
    },
    45_000,
  );

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(`LLM error ${response.status}: ${errText.slice(0, 200)}`);
  }

  const data = await response.json();
  const content: string = data.choices?.[0]?.message?.content || "";
  const html = extractHtml(content);

  const needsForm = (contract.capabilities ?? []).some((c) => c.adapter === "form");

  if (!html) {
    throw new Error("Output LLM kosong.");
  }

  if (html.length > 200_000) {
    throw new Error("Output LLM terlalu besar.");
  }

  if (!/<h1[\s>]/i.test(html)) {
    throw new Error("Output LLM tidak mengandung <h1>.");
  }

  if (needsForm && !/<form[\s>]/i.test(html)) {
    throw new Error("Output LLM tidak mengandung <form> yang valid.");
  }

  return html;
}
