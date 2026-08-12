import { db } from "@/db";
import { deliveries, projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { addLog, serializeProject } from "@/lib/project-store";
import { assertSafePublicUrl } from "@/lib/security";

export const dynamic = "force-dynamic";

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const body = (await req.json().catch(() => ({}))) as {
    preview_url?: unknown;
    summary?: unknown;
  };

  const previewUrl = String(body?.preview_url ?? "").trim();
  const summary = String(body?.summary ?? "").trim();

  if (!previewUrl) {
    return Response.json({ error: "preview_url wajib diisi." }, { status: 400 });
  }

  try {
    const parsed = await assertSafePublicUrl(previewUrl);

    if (process.env.NODE_ENV === "production" && parsed.protocol !== "https:") {
      throw new Error("production harus memakai https");
    }
  } catch (err) {
    return Response.json(
      {
        error: `preview_url tidak aman atau tidak valid: ${err instanceof Error ? err.message : "unknown"}`,
      },
      { status: 400 },
    );
  }

  const [project] = await db.select().from(projects).where(eq(projects.id, id));

  if (!project) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  if (project.executionMode === "agent") {
    return Response.json(
      {
        error: "Project ini sudah memilih Agent Runner. Buat project baru agar observasi manual tidak tercampur.",
      },
      { status: 409 },
    );
  }

  const isInitialManualDelivery = project.status === "CONFIRMED" && project.executionMode === null;

  const isManualRetry =
    (project.status === "COMPLETED" || project.status === "FAILED") && project.executionMode === "manual";

  if (!isInitialManualDelivery && !isManualRetry) {
    return Response.json(
      { error: "Kontrak harus dikonfirmasi dan jalur manual harus konsisten sebelum hasil dikirim." },
      { status: 409 },
    );
  }

  await db.transaction(async (tx) => {
    await tx
      .update(projects)
      .set({ status: "COMPLETED", executionMode: "manual", updatedAt: new Date() })
      .where(eq(projects.id, id));

    await tx.delete(deliveries).where(eq(deliveries.projectId, id));

    await tx.insert(deliveries).values({
      projectId: id,
      previewUrl,
      summary:
        summary || "Hasil manual sudah siap; jalankan Verifier sebelum sesi dianggap tervalidasi.",
    });
  });

  await addLog(id, "MODE DIPILIH: Wizard-of-Oz manual. Jalur Agent Runner dikunci untuk project ini.");
  await addLog(id, "Hasil manual dikirim oleh founder. Jalankan Verifier untuk memeriksa DoD.");

  return Response.json(await serializeProject(id));
}
