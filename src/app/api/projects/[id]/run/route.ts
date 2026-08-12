import { db } from "@/db";
import { artifacts, deliveries, projects, verifications } from "@/db/schema";
import { eq } from "drizzle-orm";
import { buildArtifact, ensureInvariants, repairArtifact } from "@/lib/agent";
import { generateArtifactWithLLM } from "@/lib/llm-builder";
import { addLog, getNextArtifactVersion, serializeProject } from "@/lib/project-store";
import { getPublicOrigin } from "@/lib/request";
import { sanitizeHtml } from "@/lib/security";
import type { Contract, VerificationResult } from "@/lib/types";
import { verifyContract, verifyContractHtml } from "@/lib/verifier";

export const dynamic = "force-dynamic";

const MAX_ITERATIONS = 3;

function countPass(results: VerificationResult[]): number {
  return results.filter((r) => r.status === "pass").length;
}

function blockingResults(results: VerificationResult[]): VerificationResult[] {
  return results.filter((r) => r.status === "fail" || r.status === "error");
}

function failsToTags(results: VerificationResult[], contract: Contract): string[] {
  const tags = new Set<string>();

  for (const r of results) {
    if (r.status !== "fail" && r.status !== "error") continue;

    const item = contract.definition_of_done.find((d) => d.id === r.dodId);
    const cap = contract.capabilities?.find((c) => c.dod_ids.includes(r.dodId));

    if (cap) {
      tags.add(`${cap.id}:${r.checkType}`);
      tags.add(cap.adapter);
    }

    tags.add(r.checkType);

    if (item?.selector?.includes("form") || r.dodId.includes("FORM")) tags.add("form");
    if (item?.selector?.includes("img") || r.dodId.includes("GALLERY")) tags.add("gallery");
    if (item?.selector === "h1" || r.dodId.includes("H1") || item?.description.toLowerCase().includes("judul")) {
      tags.add("h1");
    }
    if (item?.selector?.includes("ul, ol, table") || r.dodId.includes("LIST")) tags.add("list");
    if (item?.selector?.includes("blockquote") || r.dodId.includes("PROOF")) tags.add("proof");
    if (r.dodId.includes("CUSTOM")) tags.add("custom");
    if (r.checkType === "visual_smoke") tags.add("visual");
    if (r.checkType === "constraint_absence") tags.add("constraint");
  }

  return Array.from(tags);
}

async function saveVerifications(projectId: string, results: VerificationResult[]): Promise<void> {
  await db.transaction(async (tx) => {
    await tx.delete(verifications).where(eq(verifications.projectId, projectId));

    if (results.length > 0) {
      await tx.insert(verifications).values(
        results.map((r) => ({
          projectId,
          dodId: r.dodId,
          checkType: r.checkType,
          status: r.status,
          detail: r.detail,
        })),
      );
    }
  });
}

function selfVerifyAllowed(): boolean {
  return Boolean(process.env.APP_BASE_URL);
}

async function verifySafe(
  previewUrl: string,
  html: string,
  dod: Contract["definition_of_done"],
): Promise<VerificationResult[]> {
  if (!selfVerifyAllowed()) {
    return verifyContractHtml(html, dod);
  }

  try {
    return await verifyContract(previewUrl, dod);
  } catch {
    return verifyContractHtml(html, dod);
  }
}

export async function POST(req: Request, ctx: { params: Promise<{ id: string }> }) {
  const { id } = await ctx.params;

  const project = await serializeProject(id);

  if (!project) {
    return Response.json({ error: "Project tidak ditemukan." }, { status: 404 });
  }

  if (!project.contract) {
    return Response.json({ error: "Kontrak belum dibuat." }, { status: 400 });
  }

  if (project.executionMode === "manual") {
    return Response.json(
      {
        error:
          "Project ini sudah memilih jalur manual. Buat project baru agar observasi Agent Runner tidak tercampur.",
      },
      { status: 409 },
    );
  }

  const isInitialAgentRun = project.status === "CONFIRMED" && project.executionMode === null;

  const isAgentRetry =
    (project.status === "COMPLETED" || project.status === "FAILED") && project.executionMode === "agent";

  if (!isInitialAgentRun && !isAgentRetry) {
    return Response.json(
      {
        error: "Kontrak harus dikonfirmasi dan jalur Agent harus konsisten sebelum runner berjalan.",
      },
      { status: 409 },
    );
  }

  if (!process.env.APP_BASE_URL && process.env.NODE_ENV === "production") {
    return Response.json(
      {
        error:
          "APP_BASE_URL wajib diisi untuk Agent Runner di production agar preview URL tidak diturunkan dari header request.",
      },
      { status: 500 },
    );
  }

  const previewUrl = `${getPublicOrigin(req)}/api/projects/${id}/preview`;
  const contract = project.contract;
  const capabilities = contract.capabilities ?? [];
  const allowForm = capabilities.some((c) => c.adapter === "form");

  await db
    .update(projects)
    .set({ status: "RUNNING", executionMode: "agent", updatedAt: new Date() })
    .where(eq(projects.id, id));

  await addLog(id, "MODE DIPILIH: Agent Runner otomatis. Jalur manual dikunci untuk project ini.");

  await addLog(
    id,
    `Agent Runner: capability=${capabilities.map((c) => c.adapter).join(",") || "none"} — mulai membangun artifact.`,
  );

  try {
    let html: string;
    let artifactSource: "template" | "llm";

    if (process.env.OPENAI_API_KEY) {
      try {
        html = await generateArtifactWithLLM(contract, project.prompt, id);
        await addLog(id, "Artifact di-generate oleh LLM.");
        artifactSource = "llm";
      } catch (err) {
        const msg = err instanceof Error ? err.message : "error tidak diketahui";
        await addLog(id, `LLM code-gen gagal (${msg}) — memakai template capability.`);
        html = buildArtifact(contract, project.prompt, id);
        artifactSource = "template";
      }
    } else {
      html = buildArtifact(contract, project.prompt, id);
      await addLog(id, "Tanpa API key → artifact dibuat dari template capability.");
      artifactSource = "template";
    }

    html = sanitizeHtml(ensureInvariants(html, id, capabilities), { allowForm });

    let version = await getNextArtifactVersion(id);

    await db.insert(artifacts).values({ projectId: id, html, version, source: artifactSource });
    await addLog(id, `Artifact v${version} (mesin: ${artifactSource}) disajikan di ${previewUrl}.`);

    await db.delete(deliveries).where(eq(deliveries.projectId, id));
    await db.insert(deliveries).values({
      projectId: id,
      previewUrl,
      summary: "Sedang diverifikasi oleh Agent Runner.",
    });

    await addLog(id, "Menjalankan Verifier terhadap artifact…");

    let results = await verifySafe(previewUrl, html, contract.definition_of_done);
    let iter = 1;

    await addLog(id, `Verifikasi iterasi ${iter}: ${countPass(results)}/${results.length} lolos.`);

    while (iter < MAX_ITERATIONS && blockingResults(results).length > 0) {
      const tags = failsToTags(results, contract);

      if (tags.length === 0) break;

      const repair = repairArtifact(html, tags, capabilities, id);

      if (repair.unrepairable.length > 0) {
        await addLog(id, `Tidak bisa direpair otomatis: ${repair.unrepairable.join(" | ")}.`);
      }

      const repairedHtml = sanitizeHtml(repair.html, { allowForm });

      if (repair.applied.length === 0 || repairedHtml === html) {
        await addLog(
          id,
          "Repair loop dihentikan: tidak ada mutasi HTML yang aman/efektif. Eskalasi diperlukan.",
        );
        break;
      }

      iter++;
      html = repairedHtml;

      version = await getNextArtifactVersion(id);

      await db.insert(artifacts).values({ projectId: id, html, version, source: artifactSource });

      await addLog(id, `Repair iterasi ${iter}: ${repair.applied.join(" | ")} [${tags.join(", ")}].`);

      results = await verifySafe(previewUrl, html, contract.definition_of_done);

      await addLog(id, `Verifikasi iterasi ${iter}: ${countPass(results)}/${results.length} lolos.`);
    }

    const pass = countPass(results);
    const fail = results.filter((r) => r.status === "fail").length;
    const errors = results.filter((r) => r.status === "error").length;
    const manual = results.filter((r) => r.status === "manual").length;
    const hasBlockingFailure = fail > 0 || errors > 0;

    await saveVerifications(id, results);

    const finalStatus = hasBlockingFailure ? "FAILED" : "COMPLETED";

    const deliverySummary = hasBlockingFailure
      ? `Belum selesai: ${fail} DoD gagal dan ${errors} error setelah ${iter} iterasi. Preview tersedia untuk inspeksi.`
      : `Dibuat & diverifikasi otomatis: ${pass}/${results.length} lolos${manual ? `, ${manual} perlu cek manual` : ""}.`;

    await db.transaction(async (tx) => {
      await tx.update(projects).set({ status: finalStatus, updatedAt: new Date() }).where(eq(projects.id, id));
      await tx.update(deliveries).set({ summary: deliverySummary }).where(eq(deliveries.projectId, id));
    });

    if (hasBlockingFailure) {
      const failedIds = blockingResults(results)
        .map((r) => r.dodId)
        .join(", ");

      await addLog(
        id,
        `ESKALASI: Agent Runner berstatus GAGAL. DoD bermasalah: ${failedIds}. Jangan anggap project selesai; user perlu meninjau preview/log lalu retry.`,
      );
    } else {
      await addLog(
        id,
        `Agent Runner selesai: ${pass}/${results.length} lolos, 0 gagal/error, ${manual} perlu cek manual.`,
      );
    }

    return Response.json(await serializeProject(id));
  } catch (err) {
    console.error("Agent runner error:", err);

    await db.update(projects).set({ status: "FAILED", updatedAt: new Date() }).where(eq(projects.id, id));

    await addLog(
      id,
      `ESKALASI: Agent Runner gagal karena error sistem: ${err instanceof Error ? err.message : "error tidak diketahui"}.`,
    );

    return Response.json(await serializeProject(id));
  }
}
