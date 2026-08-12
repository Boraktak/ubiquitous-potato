import { db } from "@/db";
import { contracts, projects } from "@/db/schema";
import { eq } from "drizzle-orm";
import { classifyMockPrompt, createMockContract, generateContractWithLLM } from "@/lib/layer1";
import { addLog, serializeProject } from "@/lib/project-store";
import type { ClarificationAnswer, GeneratedContract } from "@/lib/types";

export const dynamic = "force-dynamic";

const MAX_ANSWER_LENGTH = 1_000;

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const body = (await req.json().catch(() => ({}))) as { answers?: unknown };
  const project = await serializeProject(id);

  if (!project) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  const contract = project.contract;

  if (!contract) {
    return Response.json({ error: "Kontrak belum dibuat." }, { status: 400 });
  }

  const questions = contract.clarification_questions;

  if (questions.length === 0) {
    return Response.json(
      { error: "Tidak ada pertanyaan klarifikasi untuk kontrak ini." },
      { status: 400 },
    );
  }

  const rawAnswers = Array.isArray(body?.answers) ? (body.answers as unknown[]) : [];

  if (rawAnswers.length !== questions.length) {
    return Response.json(
      { error: "Jumlah jawaban tidak sesuai dengan jumlah pertanyaan." },
      { status: 400 },
    );
  }

  const answers: ClarificationAnswer[] = questions.map((q, i) => ({
    question: q,
    answer: String(rawAnswers[i] ?? "").trim().slice(0, MAX_ANSWER_LENGTH),
  }));

  if (answers.some((a) => a.answer.length === 0)) {
    return Response.json({ error: "Semua pertanyaan wajib dijawab." }, { status: 400 });
  }

  const qaBlock = answers
    .map((a, i) => `Q${i + 1}: ${a.question}\nA${i + 1}: ${a.answer}`)
    .join("\n");

  const refinedPrompt = `${project.prompt}\nKeputusan bisnis tambahan dari user (jawaban klarifikasi):\n${qaBlock}`;

  let generated: GeneratedContract;

  try {
    if (process.env.OPENAI_API_KEY) {
      generated = await generateContractWithLLM(refinedPrompt);
      await addLog(id, "Kontrak diperbarui oleh LLM setelah klarifikasi.");
    } else {
      generated = createMockContract(refinedPrompt);
      await addLog(id, "Kontrak diperbarui (mode mock capability) setelah klarifikasi.");
    }
  } catch (err) {
    console.error("Gagal regenerate kontrak saat klarifikasi:", err);

    if (!classifyMockPrompt(project.prompt)) {
      await db.update(projects).set({ status: "FAILED", updatedAt: new Date() }).where(eq(projects.id, id));
      await addLog(id, "ESKALASI: Regenerasi LLM gagal dan fallback mock tidak aman untuk prompt ini.");

      return Response.json(
        { error: "Regenerasi LLM gagal. Fallback mock ditolak agar kontrak tidak salah arah." },
        { status: 502 },
      );
    }

    generated = createMockContract(refinedPrompt);
    await addLog(id, "Regenerasi LLM gagal; fallback mock capability dipakai.");
  }

  await db
    .update(contracts)
    .set({
      intent: generated.intent,
      userSummary: generated.user_summary,
      clarificationQuestions: [],
      clarificationAnswers: answers,
      included: generated.included,
      excluded: generated.excluded,
      capabilities: generated.capabilities,
      definitionOfDone: generated.definition_of_done,
      batches: generated.batches,
    })
    .where(eq(contracts.projectId, id));

  await db
    .update(projects)
    .set({ status: "CONTRACT_READY", updatedAt: new Date() })
    .where(eq(projects.id, id));

  await addLog(id, "Klarifikasi selesai. Execution Contract siap dikonfirmasi.");

  return Response.json(await serializeProject(id));
}
