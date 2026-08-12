import { db } from "@/db";
import { contracts, projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { addLog, serializeProject } from "@/lib/project-store";

export const dynamic = "force-dynamic";

export async function POST(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const [project] = await db.select().from(projects).where(eq(projects.id, id));

  if (!project) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  const [contract] = await db.select().from(contracts).where(eq(contracts.projectId, id));

  if (!contract) {
    return Response.json({ error: "Kontrak belum dibuat." }, { status: 400 });
  }

  if (project.status === "AWAITING_CLARIFICATION") {
    return Response.json(
      {
        error: "Masih ada klarifikasi bisnis yang perlu dijawab sebelum konfirmasi.",
      },
      { status: 400 },
    );
  }

  if (project.status === "CONFIRMED" && contract.confirmationStatus === "confirmed") {
    return Response.json(await serializeProject(id));
  }

  if (project.status !== "CONTRACT_READY") {
    return Response.json({ error: "Project tidak berada pada status kontrak siap." }, { status: 409 });
  }

  await db.transaction(async (tx) => {
    await tx.update(projects).set({ status: "CONFIRMED", updatedAt: new Date() }).where(eq(projects.id, id));
    await tx
      .update(contracts)
      .set({ confirmationStatus: "confirmed", confirmedAt: new Date() })
      .where(eq(contracts.projectId, id));
  });

  await addLog(
    id,
    "Kontrak dikonfirmasi. Pilih tepat satu jalur untuk sesi ini: Wizard-of-Oz manual ATAU Agent Runner.",
  );

  const updated = await serializeProject(id);
  return Response.json(updated);
}
