import { load } from "cheerio";
import type { CapabilityAssignment, Contract, DodItem } from "./types";

function escapeHtml(value: string): string {
  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  return value.replace(/[&<>"']/g, (char) => map[char] ?? char);
}

export function brandFromPrompt(prompt: string): string {
  const STOP = new Set([
    "saya",
    "mau",
    "ingin",
    "butuh",
    "punya",
    "ada",
    "bikin",
    "buat",
    "buatkan",
    "jadi",
    "jasa",
    "toko",
    "untuk",
    "yang",
    "di",
    "dan",
    "dengan",
    "atau",
    "ke",
    "dari",
    "bisa",
    "halaman",
    "landing",
    "page",
    "web",
    "website",
    "situs",
    "online",
    "sederhana",
    "menampilkan",
    "form",
    "isi",
    "mengisi",
    "permintaan",
    "penawaran",
    "cek",
    "ketersediaan",
    "tanggal",
    "jadwal",
    "orang",
    "tua",
    "sangat",
  ]);

  const words = prompt
    .toLowerCase()
    .replace(/[^\w\s]/g, " ")
    .split(/\s+/)
    .filter((w) => w.length > 2 && !STOP.has(w));

  const picked = words.slice(0, 2);

  if (picked.length === 0) return "Output Anda";

  return picked.map((w) => w[0].toUpperCase() + w.slice(1)).join(" ");
}

function inlineSvg(label: string, color: string): string {
  const safeLabel = escapeHtml(label);
  const svg = `<svg xmlns='http://www.w3.org/2000/svg' width='600' height='400'><rect width='600' height='400' fill='${color}'/><text x='300' y='210' font-family='sans-serif' font-size='30' fill='white' text-anchor='middle'>${safeLabel}</text></svg>`;
  return `data:image/svg+xml,${encodeURIComponent(svg)}`;
}

function renderForm(cap: CapabilityAssignment, projectId: string): string {
  const title = escapeHtml(String(cap.params.title || cap.name || "Form"));
  const showDate = String(cap.params.date ?? "").toLowerCase() === "true";
  const safeProjectId = encodeURIComponent(projectId);

  return `
<section class="section" id="${escapeHtml(cap.id.toLowerCase())}">
<h2>${title}</h2>
<form action="/api/projects/${safeProjectId}/preview" method="post" data-harness-test-safe="true">
<label>Nama<input type="text" name="nama" required /></label>
<label>Email<input type="email" name="email" required /></label>
${showDate ? `<label>Tanggal<input type="date" name="tanggal" required /></label>` : ""}
<label>Pesan<textarea name="pesan" required></textarea></label>
<button type="submit">Kirim</button>
</form>
</section>`;
}

function renderList(cap: CapabilityAssignment): string {
  const title = escapeHtml(String(cap.params.title || cap.name || "Daftar"));

  return `
<section class="section" id="${escapeHtml(cap.id.toLowerCase())}">
<h2>${title}</h2>
<ul class="list">
<li>Item utama 1</li>
<li>Item utama 2</li>
<li>Item utama 3</li>
</ul>
</section>`;
}

function renderGallery(cap: CapabilityAssignment): string {
  const title = escapeHtml(String(cap.params.title || cap.name || "Galeri"));

  return `
<section class="section" id="${escapeHtml(cap.id.toLowerCase())}">
<h2>${title}</h2>
<div class="gallery">
<img alt="Media 1" src="${inlineSvg("Media 1", "#6366f1")}" />
<img alt="Media 2" src="${inlineSvg("Media 2", "#0ea5e9")}" />
<img alt="Media 3" src="${inlineSvg("Media 3", "#10b981")}" />
</div>
</section>`;
}

function renderProof(cap: CapabilityAssignment): string {
  const title = escapeHtml(String(cap.params.title || cap.name || "Bukti Sosial"));

  return `
<section class="section" id="${escapeHtml(cap.id.toLowerCase())}">
<h2>${title}</h2>
<div class="cards">
<div class="card"><p>"Hasilnya sesuai kebutuhan dan mudah dipahami."</p><b>— Pengguna A</b></div>
<div class="card"><p>"Prosesnya jelas dan outputnya bisa diverifikasi."</p><b>— Pengguna B</b></div>
</div>
</section>`;
}

function renderCustom(cap: CapabilityAssignment): string {
  return `
<section class="section" id="${escapeHtml(cap.id.toLowerCase())}">
<h2>${escapeHtml(cap.name)}</h2>
<p>${escapeHtml(cap.goal)}</p>
<p>Kemampuan ini dirender sebagai pola custom berdasarkan intent pengguna.</p>
</section>`;
}

function renderCapability(cap: CapabilityAssignment, projectId: string): string {
  switch (cap.adapter) {
    case "form":
      return renderForm(cap, projectId);
    case "list":
      return renderList(cap);
    case "gallery":
      return renderGallery(cap);
    case "proof":
      return renderProof(cap);
    case "custom":
      return renderCustom(cap);
    default:
      return "";
  }
}

export function buildArtifact(contract: Contract, prompt: string, projectId: string): string {
  const capabilities = contract.capabilities ?? [];
  const brand = brandFromPrompt(prompt);
  const pageCap = capabilities.find((c) => c.adapter === "page");

  const headline = pageCap?.goal || contract.intent || brand;
  const subtitle = prompt.length > 160 ? `${prompt.slice(0, 160)}…` : prompt;

  const safeBrand = escapeHtml(brand);
  const safeHeadline = escapeHtml(headline);
  const safeSubtitle = escapeHtml(subtitle);

  const sections = capabilities
    .filter((c) => c.adapter !== "page")
    .map((c) => renderCapability(c, projectId))
    .join("\n");

  return `<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>${safeBrand}</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;color:#0f172a;background:#f8fafc;line-height:1.6}
.container{max-width:1000px;margin:0 auto;padding:0 20px}
.hero{background:linear-gradient(135deg,#4f46e5,#0ea5e9);color:#fff;padding:64px 0;text-align:center}
.hero h1{font-size:clamp(28px,5vw,44px);font-weight:800;line-height:1.15}
.hero p{max-width:680px;margin:16px auto 0;opacity:.95}
.section{padding:48px 0}
.section h2{font-size:26px;font-weight:800;margin-bottom:18px}
.gallery{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}
.gallery img{width:100%;height:200px;object-fit:cover;border-radius:12px}
.list{padding-left:22px}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:16px}
.card{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:20px}
form{max-width:560px;background:#fff;border:1px solid #e2e8f0;border-radius:14px;padding:24px}
label{display:block;font-weight:600;font-size:14px;margin-bottom:14px;color:#334155}
input,textarea{display:block;width:100%;margin-top:6px;padding:11px;border:1px solid #cbd5e1;border-radius:8px;font-size:15px;font-family:inherit}
textarea{min-height:90px}
button{width:100%;margin-top:6px;background:#4f46e5;color:#fff;border:0;padding:14px;border-radius:9px;font-size:16px;font-weight:700;cursor:pointer}
footer{padding:28px 0;text-align:center;color:#64748b;font-size:14px}
@media(max-width:640px){.gallery,.cards{grid-template-columns:1fr}}
</style>
</head>
<body>
<header class="hero">
<div class="container">
<h1>${safeHeadline}</h1>
<p>${safeSubtitle}</p>
</div>
</header>
<main class="container">
${sections}
</main>
<footer>
© ${new Date().getFullYear()} ${safeBrand}. Dibuat oleh HARNESS Capability Runner.
</footer>
</body>
</html>`;
}

function ensureForm($: ReturnType<typeof load>, projectId: string) {
  let form = $("form").first();

  if (form.length === 0) {
    $("body").append(`
<section id="form-fallback">
<h2>Form</h2>
<form action="/api/projects/${encodeURIComponent(projectId)}/preview" method="post" data-harness-test-safe="true">
<label>Nama<input type="text" name="nama" /></label>
<label>Email<input type="email" name="email" /></label>
<label>Pesan<textarea name="pesan"></textarea></label>
<button type="submit">Kirim</button>
</form>
</section>`);

    form = $("form").first();
  }

  form.attr("action", `/api/projects/${encodeURIComponent(projectId)}/preview`);
  form.attr("method", "post");
  form.attr("data-harness-test-safe", "true");

  const required = form.find("input[required], textarea[required], select[required]");

  if (required.length < 2) {
    form
      .find("input[type='text'], input[type='email'], input:not([type]), textarea")
      .slice(0, 2)
      .each((_, el) => {
        $(el).attr("required", "required");
      });
  }
}

export function ensureInvariants(
  html: string,
  projectId: string,
  capabilities: CapabilityAssignment[],
): string {
  const $ = load(html);

  if ($("h1").length === 0) {
    $("body").first().prepend("<h1>Output</h1>");
  } else {
    $("h1")
      .slice(1)
      .each((_, el) => {
        const $el = $(el);
        $el.replaceWith(`<h2>${$el.text()}</h2>`);
      });
  }

  const needsForm = capabilities.some((c) => c.adapter === "form");
  const needsGallery = capabilities.some((c) => c.adapter === "gallery");
  const needsList = capabilities.some((c) => c.adapter === "list");
  const needsProof = capabilities.some((c) => c.adapter === "proof");

  if (needsForm) {
    ensureForm($, projectId);
  }

  if (needsGallery) {
    const have = $("img").length;

    for (let i = have; i < 3; i++) {
      $("body").append(
        `<img alt="Media ${i + 1}" src="${inlineSvg(`Media ${i + 1}`, "#8b5cf6")}" />`,
      );
    }
  }

  if (needsList && $("ul, ol, table").length === 0) {
    $("body").append(`
<section id="list-fallback">
<h2>Daftar</h2>
<ul>
<li>Item 1</li>
<li>Item 2</li>
<li>Item 3</li>
</ul>
</section>`);
  }

  if (
    needsProof &&
    $("blockquote, figure, [class*='testimoni'], [class*='testimonial'], [class*='review']")
      .length === 0
  ) {
    $("body").append(`
<section id="proof-fallback">
<h2>Bukti Sosial</h2>
<blockquote>"Output sesuai kebutuhan."</blockquote>
</section>`);
  }

  return $.html();
}

export function thankYouHtml(brand: string, projectId: string): string {
  const safeBrand = escapeHtml(brand);
  const safeProjectId = encodeURIComponent(projectId);

  return `<!doctype html>
<html lang="id">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Terima Kasih</title>
<style>
body{font-family:system-ui,sans-serif;background:#f0fdf4;color:#064e3b;display:flex;min-height:100vh;align-items:center;justify-content:center;text-align:center;padding:20px}
.box{max-width:460px;background:#fff;border:1px solid #bbf7d0;border-radius:16px;padding:40px}
h1{font-size:28px;margin-bottom:10px}
a{display:inline-block;margin-top:20px;color:#059669;font-weight:700}
</style>
</head>
<body>
<div class="box">
<h1>Terima kasih!</h1>
<p>Data Anda sudah kami terima. Tim ${safeBrand} akan segera menindaklanjuti.</p>
<a href="/api/projects/${safeProjectId}/preview">← Kembali</a>
</div>
</body>
</html>`;
}

export interface RepairArtifactResult {
  html: string;
  applied: string[];
  unrepairable: string[];
}

export function repairArtifact(
  html: string,
  tagsOrItems: string[] | DodItem[],
  capabilities: CapabilityAssignment[],
  projectId: string,
): RepairArtifactResult {
  const $ = load(html);
  const applied = new Set<string>();
  const unrepairable = new Set<string>();

  const tags: string[] =
    Array.isArray(tagsOrItems) && tagsOrItems.length > 0 && typeof tagsOrItems[0] === "string"
      ? (tagsOrItems as string[])
      : (tagsOrItems as DodItem[]).flatMap((item) => {
          const t = [item.check_type, item.id];
          if (item.selector) t.push(item.selector);
          return t;
        });

  const hasTag = (keyword: string) => tags.some((t) => t.toLowerCase().includes(keyword.toLowerCase()));

  if (hasTag("h1") || hasTag("page") || hasTag("smoke") || hasTag("visual")) {
    if ($("h1").length === 0) {
      $("body").first().prepend("<h1>Output</h1>");
      applied.add("h1");
    }

    if (($("body").text() || "").trim().length < 20) {
      $("body").append("<main><p>Halaman sedang disiapkan untuk kebutuhan Anda.</p></main>");
      applied.add("visual-smoke");
    }
  }

  const needsForm =
    hasTag("form") ||
    capabilities.some(
      (c) => c.adapter === "form" && tags.some((t) => t.toLowerCase().includes(c.id.toLowerCase())),
    );

  if (needsForm) {
    ensureForm($, projectId);
    applied.add("form");
  }

  if (hasTag("gallery") || capabilities.some((c) => c.adapter === "gallery" && hasTag(c.id))) {
    const have = $("img").length;

    for (let i = have; i < 3; i++) {
      $("body").append(
        `<img alt="Media ${i + 1}" src="${inlineSvg(`Media ${i + 1}`, "#8b5cf6")}" />`,
      );
    }

    applied.add("gallery");
  }

  if (hasTag("list") || capabilities.some((c) => c.adapter === "list" && hasTag(c.id))) {
    if ($("ul, ol, table").length === 0) {
      $("body").append(`
<section id="list-repair">
<h2>Daftar</h2>
<ul>
<li>Item 1</li>
<li>Item 2</li>
<li>Item 3</li>
</ul>
</section>`);

      applied.add("list");
    }
  }

  if (hasTag("proof") || capabilities.some((c) => c.adapter === "proof" && hasTag(c.id))) {
    if (
      $("blockquote, figure, [class*='testimoni'], [class*='testimonial'], [class*='review']")
        .length === 0
    ) {
      $("body").append(`
<section id="proof-repair">
<h2>Bukti Sosial</h2>
<blockquote>"Output sesuai kebutuhan."</blockquote>
</section>`);

      applied.add("proof");
    }
  }

  if (hasTag("custom") || hasTag("constraint")) {
    if (hasTag("custom") && $("[id*='custom']").length === 0) {
      $("body").append(`
<section id="custom-repair">
<h2>Kemampuan Khusus</h2>
<p>Bagian ini ditambahkan untuk memenuhi kemampuan custom dari intent pengguna.</p>
</section>`);

      applied.add("custom");
    }

    $("input[type='password']").remove();

    $("input[name]").each((_, el) => {
      const name = ($(el).attr("name") || "").toLowerCase();

      if (["card", "cvc", "cvv", "ccnumber", "cc-number", "cc-exp", "expiry"].some((k) => name.includes(k))) {
        $(el).remove();
      }
    });

    if (hasTag("constraint")) applied.add("constraint");
  }

  if (hasTag("http")) {
    unrepairable.add("http_ok: reachability harus diperbaiki pada deployment");
  }

  return { html: $.html(), applied: [...applied], unrepairable: [...unrepairable] };
}
