import { db } from "@/db";
import { projects, verifications } from "@/db/schema";
import { eq } from "drizzle-orm";
import { addLog, serializeProject } from "@/lib/project-store";
import { verifyContract } from "@/lib/verifier";

export const dynamic = "force-dynamic";

export async function POST(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const project = await serializeProject(id);

  if (!project) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  if (!project.delivery) {
    return Response.json(
      { error: "Deliver hasil (preview URL) dulu sebelum verifikasi." },
      { status: 400 },
    );
  }

  if (!project.contract) {
    return Response.json({ error: "Kontrak belum dibuat." }, { status: 400 });
  }

  await addLog(id, `Verifier dijalankan terhadap ${project.delivery.previewUrl}…`);

  let results;

  try {
    results = await verifyContract(project.delivery.previewUrl, project.contract.definition_of_done);
  } catch (err) {
    console.error("Verifier error:", err);

    await db.update(projects).set({ status: "FAILED", updatedAt: new Date() }).where(eq(projects.id, id));

    await addLog(id, "ESKALASI: Verifier gagal dijalankan; project ditandai GAGAL.");

    return Response.json(await serializeProject(id));
  }

  const pass = results.filter((r) => r.status === "pass").length;
  const fail = results.filter((r) => r.status === "fail").length;
  const errors = results.filter((r) => r.status === "error").length;
  const manual = results.filter((r) => r.status === "manual").length;

  const finalStatus = fail > 0 || errors > 0 ? "FAILED" : "COMPLETED";

  await db.transaction(async (tx) => {
    await tx.delete(verifications).where(eq(verifications.projectId, id));

    if (results.length > 0) {
      await tx.insert(verifications).values(
        results.map((r) => ({
          projectId: id,
          dodId: r.dodId,
          checkType: r.checkType,
          status: r.status,
          detail: r.detail,
        })),
      );
    }

    await tx.update(projects).set({ status: finalStatus, updatedAt: new Date() }).where(eq(projects.id, id));
  });

  await addLog(
    id,
    finalStatus === "FAILED"
      ? `ESKALASI: Verifier menemukan ${fail} gagal dan ${errors} error. Project ditandai GAGAL, bukan selesai.`
      : `Verifier selesai: ${pass}/${results.length} lolos, ${manual} perlu pengecekan manual.`,
  );

  return Response.json(await serializeProject(id));
}
