import { db } from "@/db";
import { projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { brandFromPrompt, thankYouHtml } from "@/lib/agent";
import { addLead, getLatestArtifact } from "@/lib/project-store";
import { PREVIEW_HTML_HEADERS, sanitizeHtml } from "@/lib/security";

export const dynamic = "force-dynamic";

const MAX_LEAD_FIELDS = 20;
const MAX_LEAD_KEY_LENGTH = 100;
const MAX_LEAD_VALUE_LENGTH = 2_000;

export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const artifact = await getLatestArtifact(id);

  if (!artifact) {
    return new Response("Preview belum tersedia. Jalankan Agent Runner terlebih dahulu.", {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }

  const safeHtml = sanitizeHtml(artifact.html, { allowForm: true });

  return new Response(safeHtml, { headers: PREVIEW_HTML_HEADERS });
}

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const [proj] = await db.select({ prompt: projects.prompt }).from(projects).where(eq(projects.id, id));

  if (!proj) {
    return new Response("Project tidak ditemukan.", {
      status: 404,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }

  const form = await req.formData();
  const entries = Array.from(form.entries());

  if (entries.length > MAX_LEAD_FIELDS) {
    return new Response("Terlalu banyak field form.", {
      status: 400,
      headers: { "Content-Type": "text/plain; charset=utf-8" },
    });
  }

  const isTest =
    req.headers.get("x-harness-verification") === "1" || String(form.get("_harness_test") ?? "") === "1";

  const payload: Record<string, string> = {};

  for (const [key, value] of entries) {
    if (key === "_harness_test") continue;

    const safeKey = String(key).slice(0, MAX_LEAD_KEY_LENGTH).trim();
    const safeValue = String(value).slice(0, MAX_LEAD_VALUE_LENGTH);

    if (!safeKey) continue;

    payload[safeKey] = safeValue;
  }

  await addLead(id, payload, isTest);

  const brand = brandFromPrompt(proj.prompt);
  const html = thankYouHtml(brand, id);

  return new Response(html, { headers: PREVIEW_HTML_HEADERS });
}
