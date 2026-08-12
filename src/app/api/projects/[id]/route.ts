import { db } from "@/db";
import { projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { serializeProject } from "@/lib/project-store";

export const dynamic = "force-dynamic";

export async function GET(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const project = await serializeProject(id);

  if (!project) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  return Response.json(project);
}

export async function DELETE(_req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const deleted = await db.delete(projects).where(eq(projects.id, id)).returning({ id: projects.id });

  if (deleted.length === 0) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  return Response.json({ ok: true, id });
}
