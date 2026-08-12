import { load } from "cheerio";
import { promises as dns } from "node:dns";
import net from "node:net";

export function escapeHtml(value: string): string {
  const map: Record<string, string> = {
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  };

  return value.replace(/[&<>"']/g, (char) => map[char] ?? char);
}

export const PREVIEW_HTML_HEADERS: Record<string, string> = {
  "Content-Type": "text/html; charset=utf-8",
  "X-Content-Type-Options": "nosniff",
  "Referrer-Policy": "no-referrer",
  "X-Frame-Options": "DENY",
  "Content-Security-Policy": [
    "default-src 'none'",
    "style-src 'unsafe-inline'",
    "img-src data:",
    "form-action 'self'",
    "base-uri 'none'",
    "frame-ancestors 'none'",
    "object-src 'none'",
    "script-src 'none'",
  ].join("; "),
};

const ALLOWED_TAGS = new Set([
  "html",
  "head",
  "body",
  "title",
  "style",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "p",
  "br",
  "hr",
  "section",
  "article",
  "header",
  "footer",
  "main",
  "nav",
  "aside",
  "div",
  "span",
  "ul",
  "ol",
  "li",
  "table",
  "thead",
  "tbody",
  "tfoot",
  "tr",
  "td",
  "th",
  "form",
  "label",
  "input",
  "textarea",
  "select",
  "option",
  "button",
  "fieldset",
  "legend",
  "blockquote",
  "figure",
  "figcaption",
  "img",
  "strong",
  "b",
  "em",
  "i",
  "u",
  "small",
  "mark",
  "dl",
  "dt",
  "dd",
  "abbr",
  "time",
]);

const GLOBAL_ATTRS = new Set(["class", "id", "lang", "dir"]);

const TAG_ATTRS: Record<string, Set<string>> = {
  form: new Set(["action", "method", "novalidate", "data-harness-test-safe"]),
  input: new Set([
    "type",
    "name",
    "required",
    "placeholder",
    "value",
    "disabled",
    "readonly",
    "min",
    "max",
    "step",
    "maxlength",
  ]),
  textarea: new Set(["name", "required", "placeholder", "rows", "cols", "maxlength"]),
  select: new Set(["name", "required"]),
  option: new Set(["value", "selected"]),
  button: new Set(["type", "disabled"]),
  img: new Set(["src", "alt", "width", "height", "loading"]),
  label: new Set(["for"]),
  time: new Set(["datetime"]),
};

const PAYMENT_FIELD_KEYWORDS = [
  "card",
  "cvc",
  "cvv",
  "ccnumber",
  "cc-number",
  "cc-exp",
  "expiry",
];

function cleanCss(css: string): string {
  return css
    .replace(/@import[^;]+;?/gi, "")
    .replace(/url\([^)]*\)/gi, "transparent")
    .replace(/javascript:[^;}"]*/gi, "")
    .replace(/expression\([^)]*\)/gi, "");
}

function isSafeUrl(value: string, kind: "action" | "href" | "src"): boolean {
  const v = value.trim();

  if (!v) return false;

  if (kind === "action") {
    return v.startsWith("/") && !v.startsWith("//");
  }

  if (v.startsWith("#")) return true;
  if (v.startsWith("/") && !v.startsWith("//")) return true;

  if (/^data:image\/(png|jpe?g|gif|webp|svg\+xml);/i.test(v)) {
    return !/script/i.test(v);
  }

  if (/^(javascript|vbscript|file|data):/i.test(v)) return false;

  try {
    const url = new URL(v);

    if (!["http:", "https:", "mailto:", "tel:"].includes(url.protocol)) {
      return false;
    }

    return true;
  } catch {
    return false;
  }
}

export function sanitizeHtml(
  input: string,
  opts: { allowForm?: boolean } = {},
): string {
  const $ = load(input);

  $(
    "script, noscript, template, link, meta, base, iframe, frame, object, embed, applet, svg, canvas, video, audio, source, track",
  ).remove();

  if (opts.allowForm === false) {
    $("form").remove();
  }

  $("input[type='password']").remove();

  $("input[name], select[name], textarea[name]").each((_, el) => {
    const name = String($(el).attr("name") || "").toLowerCase();

    if (PAYMENT_FIELD_KEYWORDS.some((k) => name.includes(k))) {
      $(el).remove();
    }
  });

  $("style").each((_, el) => {
    const css = String($(el).html() || "");
    $(el).text(cleanCss(css));
  });

  $("*").each((_, el) => {
    const $el = $(el);
    const tag = String((el as { tagName?: string }).tagName || "").toLowerCase();

    if (!ALLOWED_TAGS.has(tag)) {
      $el.replaceWith($el.contents());
      return;
    }

    const attribs = (el as { attribs?: Record<string, string> }).attribs || {};

    for (const rawName of Object.keys(attribs)) {
      const name = rawName.toLowerCase();
      const value = attribs[rawName] ?? "";

      if (name.startsWith("on")) {
        $el.removeAttr(rawName);
        continue;
      }

      const allowedForTag = TAG_ATTRS[tag]?.has(name) ?? false;
      const isAllowed = GLOBAL_ATTRS.has(name) || allowedForTag;

      if (!isAllowed) {
        $el.removeAttr(rawName);
        continue;
      }

      if (name === "action") {
        if (!isSafeUrl(value, "action")) {
          $el.attr("action", "#");
        }
        continue;
      }

      if (name === "href" || name === "src") {
        if (!isSafeUrl(value, name)) {
          $el.removeAttr(rawName);
        }
      }
    }
  });

  $("*")
    .contents()
    .each((_, child) => {
      if ((child as { type?: string }).type === "comment") {
        $(child).remove();
      }
    });

  const output = $.html();

  return /<!doctype html>/i.test(output)
    ? output
    : `<!doctype html>\n${output}`;
}

const BLOCKED_HOSTNAMES = new Set([
  "localhost",
  "0.0.0.0",
  "127.0.0.1",
  "::1",
  "metadata.google.internal",
  "169.254.169.254",
]);

function isPrivateIpv4(ip: string): boolean {
  const parts = ip.split(".").map(Number);

  if (parts.length !== 4 || parts.some((n) => Number.isNaN(n))) return true;

  const [a, b] = parts;

  return (
    a === 0 ||
    a === 10 ||
    a === 127 ||
    (a === 169 && b === 254) ||
    (a === 172 && b >= 16 && b <= 31) ||
    (a === 192 && b === 168) ||
    (a === 198 && (b === 18 || b === 19)) ||
    (a === 100 && b >= 64 && b <= 127)
  );
}

function isPrivateIpv6(ip: string): boolean {
  const v6 = ip.toLowerCase();

  if (v6 === "::" || v6 === "::1") return true;
  if (v6.startsWith("fc") || v6.startsWith("fd")) return true;
  if (v6.startsWith("fe80")) return true;

  if (v6.startsWith("::ffff:")) {
    const maybeIpv4 = v6.slice(7);
    if (net.isIPv4(maybeIpv4)) return isPrivateIpv4(maybeIpv4);
  }

  return false;
}

function isPrivateIp(ip: string): boolean {
  if (net.isIPv4(ip)) return isPrivateIpv4(ip);
  if (net.isIPv6(ip)) return isPrivateIpv6(ip);
  return true;
}

function isLocalHostname(hostname: string): boolean {
  const h = hostname.toLowerCase();

  return (
    h === "localhost" ||
    h.endsWith(".localhost") ||
    h === "0.0.0.0" ||
    h === "127.0.0.1" ||
    h === "::1"
  );
}

export async function assertSafePublicUrl(rawUrl: string): Promise<URL> {
  let url: URL;

  try {
    url = new URL(rawUrl);
  } catch {
    throw new Error("URL tidak valid.");
  }

  if (!["http:", "https:"].includes(url.protocol)) {
    throw new Error("Protocol URL harus http atau https.");
  }

  if (url.username || url.password) {
    throw new Error("URL tidak boleh mengandung kredensial.");
  }

  const hostname = url.hostname.replace(/^\[|\]$/g, "").toLowerCase();

  const allowLocal =
    process.env.ALLOW_LOCAL_SSRF === "1" || process.env.NODE_ENV !== "production";

  if (isLocalHostname(hostname)) {
    if (!allowLocal) {
      throw new Error("URL lokal tidak diperbolehkan.");
    }

    return url;
  }

  if (BLOCKED_HOSTNAMES.has(hostname)) {
    throw new Error("URL target tidak diperbolehkan.");
  }

  if (net.isIP(hostname)) {
    if (isPrivateIp(hostname) && !allowLocal) {
      throw new Error("URL menuju alamat privat/lokal.");
    }

    return url;
  }

  let ips: string[] = [];

  try {
    ips = await dns.resolve4(hostname);
  } catch {
    // ignore
  }

  try {
    const ipv6 = await dns.resolve6(hostname);
    ips = ips.concat(ipv6);
  } catch {
    // ignore
  }

  if (ips.length === 0) {
    throw new Error("DNS tidak dapat di-resolve.");
  }

  for (const ip of ips) {
    if (isPrivateIp(ip) && !allowLocal) {
      throw new Error("URL menuju alamat privat/lokal.");
    }
  }

  return url;
}
