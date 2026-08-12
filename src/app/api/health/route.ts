import { db } from "@/db";
import { sql } from "drizzle-orm";

export const dynamic = "force-dynamic";

export async function GET() {
  try {
    await db.execute(sql`select 1`);

    const warnings: string[] = [];

    if (!process.env.APP_BASE_URL) warnings.push("APP_BASE_URL belum diisi; Agent Runner production akan ditolak.");
    if (!process.env.PILOT_ACCESS_KEY) warnings.push("PILOT_ACCESS_KEY belum diisi; metadata project belum dilindungi.");
    if (!process.env.OPENAI_API_KEY) warnings.push("OPENAI_API_KEY belum diisi; sistem memakai mock capability.");

    return Response.json({
      ok: true,
      database: "ok",
      config: {
        publicBaseUrl: Boolean(process.env.APP_BASE_URL),
        pilotGate: Boolean(process.env.PILOT_ACCESS_KEY),
        llm: Boolean(process.env.OPENAI_API_KEY),
        sharedRateLimit: Boolean(process.env.UPSTASH_REDIS_REST_URL && process.env.UPSTASH_REDIS_REST_TOKEN),
      },
      warnings,
    });
  } catch {
    return Response.json({ ok: false, database: "error" }, { status: 500 });
  }
}
