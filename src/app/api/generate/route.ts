import { db } from "@/db";
import { contracts, projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import {
  MOCK_SUPPORTED_SCENARIOS,
  classifyMockPrompt,
  createMockContract,
  generateContractWithLLM,
} from "@/lib/layer1";
import { addLog, serializeProject } from "@/lib/project-store";
import type { GeneratedContract } from "@/lib/types";

export const dynamic = "force-dynamic";

const MOCK_LIMIT_MESSAGE = `Mode mock memakai capability discovery generik. Isi OPENAI_API_KEY untuk hasil LLM yang lebih adaptif. Contoh: ${MOCK_SUPPORTED_SCENARIOS.slice(0, 3).join("; ")}.`;

export async function POST(req: Request) {
  const body = (await req.json().catch(() => ({}))) as { prompt?: unknown };
  const prompt = String(body?.prompt ?? "").trim();

  if (!prompt) {
    return Response.json({ error: "Prompt tidak boleh kosong." }, { status: 400 });
  }

  if (prompt.length > 4_000) {
    return Response.json({ error: "Prompt terlalu panjang (maksimal 4.000 karakter)." }, { status: 400 });
  }

  const mockScenario = classifyMockPrompt(prompt);

  if (!process.env.OPENAI_API_KEY && !mockScenario) {
    return Response.json({ error: MOCK_LIMIT_MESSAGE }, { status: 422 });
  }

  const [projectRow] = await db
    .insert(projects)
    .values({ prompt, status: "DRAFT" })
    .returning({ id: projects.id });

  const projectId = projectRow.id;

  await addLog(projectId, "Prompt diterima. Layer 1 sedang menerjemahkan kontrak berbasis capability.");

  let contract: GeneratedContract;

  try {
    if (process.env.OPENAI_API_KEY) {
      contract = await generateContractWithLLM(prompt);
      await addLog(projectId, "Kontrak dihasilkan oleh LLM (capability planner).");
    } else {
      contract = createMockContract(prompt);
      await addLog(
        projectId,
        `Mode mock capability dipakai (skenario: ${mockScenario}). Caps: ${contract.capabilities.map((c) => c.adapter).join(",")}.`,
      );
    }
  } catch (err) {
    console.error("Gagal generate kontrak dengan LLM:", err);

    if (!mockScenario) {
      await db.update(projects).set({ status: "FAILED", updatedAt: new Date() }).where(eq(projects.id, projectId));
      await addLog(projectId, "ESKALASI: LLM gagal dan prompt tidak aman untuk fallback mock. Kontrak tidak dikarang.");

      return Response.json(
        { error: `LLM gagal dan fallback mock ditolak. ${MOCK_LIMIT_MESSAGE}` },
        { status: 502 },
      );
    }

    contract = createMockContract(prompt);
    await addLog(projectId, "LLM gagal. Fallback mock capability dipakai.");
  }

  await db.insert(contracts).values({
    projectId,
    intent: contract.intent,
    userSummary: contract.user_summary,
    clarificationQuestions: contract.clarification_questions,
    included: contract.included,
    excluded: contract.excluded,
    capabilities: contract.capabilities,
    definitionOfDone: contract.definition_of_done,
    batches: contract.batches,
    source: contract.source,
    confirmationStatus: "pending",
  });

  const hasClarification = contract.clarification_questions.length > 0;
  const nextStatus = hasClarification ? "AWAITING_CLARIFICATION" : "CONTRACT_READY";

  await db
    .update(projects)
    .set({ status: nextStatus, updatedAt: new Date() })
    .where(eq(projects.id, projectId));

  await addLog(
    projectId,
    hasClarification
      ? "Ada keputusan bisnis yang perlu diklarifikasi."
      : "Execution Contract (capability-based) siap dikonfirmasi.",
  );

  return Response.json(await serializeProject(projectId));
}
