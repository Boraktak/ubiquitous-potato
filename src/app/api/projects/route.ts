import { listProjects } from "@/lib/project-store";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    const list = await listProjects();
    return Response.json(list);
  } catch (err) {
    console.error("Gagal mengambil daftar project:", err);
    return Response.json({ error: "Gagal mengambil daftar project." }, { status: 500 });
  }
}
