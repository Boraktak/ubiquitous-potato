import { load } from "cheerio";
import { assertSafePublicUrl } from "./security";
import type { CheckType, DodItem, VerificationResult, VerificationStatus } from "./types";

interface FetchInfo {
  ok: boolean;
  status: number;
  contentType: string;
  html: string | null;
  finalUrl: string;
  error?: string;
}

const FETCH_TIMEOUT_MS = 10_000;
const UA = "HARNESS-Verifier/1.0 (+layer1)";

const SUCCESS_KEYWORDS = [
  "terima kasih",
  "thank you",
  "thanks",
  "success",
  "berhasil",
  "terkirim",
  "submitted",
  "request received",
  "we'll be in touch",
  "we will contact",
];

const STATIC_SUBMIT_STATUSES = new Set([404, 405, 501]);

function makeResult(
  item: DodItem,
  status: VerificationStatus,
  detail: string,
  createdAt: string,
): VerificationResult {
  return { dodId: item.id, checkType: item.check_type, status, detail, createdAt };
}

async function fetchUrl(url: string): Promise<FetchInfo> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    await assertSafePublicUrl(url);

    const res = await fetch(url, {
      redirect: "error",
      signal: controller.signal,
      headers: { "User-Agent": UA },
    });

    const contentType = res.headers.get("content-type") ?? "";
    const isHtml =
      contentType.includes("text/html") || contentType.includes("application/xhtml");

    const html = isHtml ? await res.text() : null;

    return {
      ok: res.ok,
      status: res.status,
      contentType,
      html,
      finalUrl: res.url || url,
    };
  } catch (err) {
    return {
      ok: false,
      status: 0,
      contentType: "",
      html: null,
      finalUrl: url,
      error: err instanceof Error ? err.message : "fetch gagal",
    };
  } finally {
    clearTimeout(timer);
  }
}

function resolveUrl(base: string, action?: string | null): string {
  if (!action || action.trim() === "") return base;

  try {
    return new URL(action, base).toString();
  } catch {
    return base;
  }
}

function defaultSelectorFor(item: DodItem): string | null {
  if (item.selector) return item.selector;

  const id = item.id.toUpperCase();
  const desc = item.description.toLowerCase();

  if (item.check_type === "dom_exists") {
    if (id.includes("GALLERY") || desc.includes("galeri") || desc.includes("gambar")) {
      return "img";
    }

    if (id.includes("FORM") || desc.includes("form")) {
      return "form";
    }

    if (id.includes("LIST") || desc.includes("daftar") || desc.includes("list")) {
      return "ul, ol, table";
    }

    if (id.includes("PROOF") || desc.includes("testimoni") || desc.includes("ulasan")) {
      return "blockquote, figure, [class*='testimoni'], [class*='testimonial'], [class*='review']";
    }

    if (id.includes("H1") || desc.includes("judul")) {
      return "h1";
    }
  }

  return null;
}

function sampleValue(name: string, type: string): string {
  const n = name.toLowerCase();

  if (type === "email" || n.includes("email") || n.includes("surel")) return "harness-test@example.com";

  if (
    type === "tel" ||
    n.includes("phone") ||
    n.includes("telp") ||
    n.includes("hp") ||
    n.includes("wa") ||
    n.includes("whatsapp")
  ) {
    return "081200001111";
  }

  if (type === "number") return "1";
  if (n.includes("date") || type === "date") return "2099-01-01";
  if (type === "checkbox" || type === "radio") return "yes";

  return "Tes Harness";
}

export async function verifyContract(url: string, dodItems: DodItem[]): Promise<VerificationResult[]> {
  const createdAt = new Date().toISOString();
  const info = await fetchUrl(url);
  const $ = info.html ? load(info.html) : null;

  const bodyText = $
    ? ($("body").text() || $("html").text() || "").replace(/\s+/g, " ").toLowerCase()
    : "";

  const results: VerificationResult[] = [];

  for (const item of dodItems) {
    try {
      results.push(await runCheck(item, url, info, $, bodyText, createdAt));
    } catch (err) {
      results.push(
        makeResult(
          item,
          "error",
          `Verifier error: ${err instanceof Error ? err.message : "error tidak diketahui"}.`,
          createdAt,
        ),
      );
    }
  }

  return results;
}

export async function verifyContractHtml(html: string, dodItems: DodItem[]): Promise<VerificationResult[]> {
  const createdAt = new Date().toISOString();
  const $ = load(html);

  const bodyText = ($("body").text() || $("html").text() || "").replace(/\s+/g, " ").toLowerCase();

  const info: FetchInfo = {
    ok: true,
    status: 200,
    contentType: "text/html",
    html,
    finalUrl: "artifact",
  };

  const results: VerificationResult[] = [];

  for (const item of dodItems) {
    try {
      results.push(await runCheck(item, "artifact", info, $, bodyText, createdAt));
    } catch (err) {
      results.push(
        makeResult(
          item,
          "error",
          `Verifier error: ${err instanceof Error ? err.message : "error tidak diketahui"}.`,
          createdAt,
        ),
      );
    }
  }

  return results;
}

async function runCheck(
  item: DodItem,
  url: string,
  info: FetchInfo,
  $: ReturnType<typeof load> | null,
  bodyText: string,
  createdAt: string,
): Promise<VerificationResult> {
  switch (item.check_type as CheckType) {
    case "http_ok": {
      if (info.ok && info.status >= 200 && info.status < 400) {
        return makeResult(item, "pass", `HTTP ${info.status} · ${info.contentType || "ok"}`, createdAt);
      }

      return makeResult(
        item,
        "fail",
        info.error ? `Gagal mengambil: ${info.error}` : `HTTP ${info.status}`,
        createdAt,
      );
    }

    case "visual_smoke": {
      if (!$) return makeResult(item, "fail", "HTML tidak terbaca / bukan halaman HTML.", createdAt);

      const len = info.html?.trim().length ?? 0;
      const hasBody = $("body").length > 0 && $("body").text().trim().length > 0;

      if (len > 400 && hasBody) {
        return makeResult(item, "pass", `Halaman ter-render (${len} karakter HTML).`, createdAt);
      }

      return makeResult(item, "fail", "Halaman tampak kosong atau error.", createdAt);
    }

    case "dom_exists": {
      if (!$) return makeResult(item, "fail", "HTML tidak terbaca.", createdAt);

      const selector = defaultSelectorFor(item);

      if (!selector) {
        return makeResult(item, "manual", "Tidak ada selector yang bisa diuji otomatis.", createdAt);
      }

      const count = $(selector).length;
      const min = item.min_count ?? (item.id.toUpperCase().includes("GALLERY") ? 3 : 1);

      if (count >= min) {
        return makeResult(item, "pass", `Ditemukan ${count} × "${selector}" (min ${min}).`, createdAt);
      }

      if (item.contains && bodyText.includes(item.contains.toLowerCase())) {
        return makeResult(
          item,
          "manual",
          `"${selector}" tidak ditemukan, tapi teks "${item.contains}" ada.`,
          createdAt,
        );
      }

      return makeResult(item, "fail", `Hanya ditemukan ${count} × "${selector}" (min ${min}).`, createdAt);
    }

    case "dom_contains": {
      if (!$) return makeResult(item, "fail", "HTML tidak terbaca.", createdAt);

      const target = item.contains;

      if (!target) {
        return makeResult(item, "manual", "Tidak ada teks target (contains) untuk dicek.", createdAt);
      }

      const hay = item.selector ? $(item.selector).text() : bodyText;

      if (hay.toLowerCase().includes(target.toLowerCase())) {
        return makeResult(item, "pass", `Teks "${target}" ditemukan.`, createdAt);
      }

      return makeResult(item, "fail", `Teks "${target}" tidak ditemukan.`, createdAt);
    }

    case "form_negative_test": {
      if (!$) return makeResult(item, "fail", "HTML tidak terbaca.", createdAt);

      const form = $("form").first();

      if (form.length === 0) return makeResult(item, "fail", "Tidak ada <form> di halaman.", createdAt);

      const requiredCount = form.find("input[required], select[required], textarea[required]").length;

      if (requiredCount > 0) {
        return makeResult(item, "pass", `Form punya ${requiredCount} field wajib (validasi HTML5).`, createdAt);
      }

      return makeResult(
        item,
        "manual",
        "Tidak ada field 'required' terdeteksi — verifikasi validasi manual.",
        createdAt,
      );
    }

    case "form_positive_test": {
      return submitForm(item, url, $, createdAt);
    }

    case "constraint_absence": {
      if (!$) return makeResult(item, "fail", "HTML tidak terbaca.", createdAt);

      const found: string[] = [];

      if ($("input[type='password']").length > 0) found.push("input password (login)");

      const payFields = $("input[name]").filter((_, el) => {
        const nm = ($(el).attr("name") ?? "").toLowerCase();
        return ["card", "cvc", "cvv", "ccnumber", "cc-number", "cc-exp", "expiry"].some((k) => nm.includes(k));
      });

      if (payFields.length > 0) found.push("field kartu/checkout");

      const scripts = $("script[src]")
        .map((_, el) => $(el).attr("src") ?? "")
        .get()
        .join(" ")
        .toLowerCase();

      if (["stripe", "paypal", "midtrans", "xendit", "razorpay"].some((g) => scripts.includes(g))) {
        found.push("script payment gateway");
      }

      if (found.length === 0) {
        return makeResult(item, "pass", "Tidak ditemukan elemen login/payment.", createdAt);
      }

      return makeResult(item, "fail", `Terdeteksi: ${found.join("; ")}.`, createdAt);
    }

    default:
      return makeResult(item, "manual", "Tipe cek tidak dikenal.", createdAt);
  }
}

async function submitForm(
  item: DodItem,
  url: string,
  $: ReturnType<typeof load> | null,
  createdAt: string,
): Promise<VerificationResult> {
  if (!$) {
    return makeResult(item, "fail", "HTML tidak terbaca.", createdAt);
  }

  if (!/^https?:\/\//i.test(url)) {
    return makeResult(
      item,
      "manual",
      "Form positive test memerlukan preview URL publik yang valid.",
      createdAt,
    );
  }

  const form = $("form").first();

  if (form.length === 0) {
    return makeResult(item, "fail", "Tidak ada <form> di halaman.", createdAt);
  }

  const action = form.attr("action");
  const method = (form.attr("method") || "get").toLowerCase();
  const target = resolveUrl(url, action);

  try {
    await assertSafePublicUrl(target);
  } catch {
    return makeResult(
      item,
      "manual",
      "Form action tidak lolos SSRF guard.",
      createdAt,
    );
  }

  const isHarnessEndpoint = /\/api\/projects\/[0-9a-f-]+\/preview(?:[/?#]|$)/i.test(target);
  const isExplicitlySafe = form.attr("data-harness-test-safe") === "true";

  if (!isHarnessEndpoint && !isExplicitlySafe) {
    return makeResult(
      item,
      "manual",
      "Form tidak di-submit agar data lead produksi tidak tercemar.",
      createdAt,
    );
  }

  const params = new URLSearchParams();
  params.set("_harness_test", "1");

  form.find("input, select, textarea").each((_, el) => {
    const name = $(el).attr("name");

    if (!name) return;

    const type = ($(el).attr("type") ?? "").toLowerCase();

    if (["submit", "button", "reset", "image", "file"].includes(type)) return;

    params.append(name, sampleValue(name, type));
  });

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), FETCH_TIMEOUT_MS);

  try {
    const resp =
      method === "post"
        ? await fetch(target, {
            method: "POST",
            body: params,
            redirect: "error",
            signal: controller.signal,
            headers: {
              "User-Agent": UA,
              "X-Harness-Verification": "1",
            },
          })
        : await fetch(
            `${target}${target.includes("?") ? "&" : "?"}${params.toString()}`,
            {
              redirect: "error",
              signal: controller.signal,
              headers: {
                "User-Agent": UA,
                "X-Harness-Verification": "1",
              },
            },
          );

    const text = await resp.text().catch(() => "");
    const lower = text.toLowerCase();

    const okKeyword = SUCCESS_KEYWORDS.some((k) => lower.includes(k));
    const okStatus = resp.status >= 200 && resp.status < 400;

    if (okStatus && okKeyword) {
      return makeResult(
        item,
        "pass",
        `Form terkirim → HTTP ${resp.status} · pesan sukses terdeteksi.`,
        createdAt,
      );
    }

    if (okStatus) {
      return makeResult(
        item,
        "pass",
        `Form terkirim → HTTP ${resp.status} (tanpa pesan sukses eksplisit).`,
        createdAt,
      );
    }

    if (STATIC_SUBMIT_STATUSES.has(resp.status)) {
      return makeResult(
        item,
        "manual",
        `Endpoint form menolak (HTTP ${resp.status}) — kemungkinan preview statis tanpa backend.`,
        createdAt,
      );
    }

    return makeResult(
      item,
      "fail",
      `Pengiriman ke ${target} → HTTP ${resp.status}.`,
      createdAt,
    );
  } catch (err) {
    return makeResult(
      item,
      "error",
      `Gagal submit form ke ${target}: ${err instanceof Error ? err.message : "error"}.`,
      createdAt,
    );
  } finally {
    clearTimeout(timer);
  }
}
