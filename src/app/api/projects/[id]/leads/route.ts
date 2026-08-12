import { listLeads, serializeProject } from "@/lib/project-store";

export const dynamic = "force-dynamic";

export async function GET(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const project = await serializeProject(id);

  if (!project) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  const includeTest = new URL(req.url).searchParams.get("include_test") === "1";

  return Response.json(await listLeads(id, includeTest));
}
