import { db } from "@/db";
import { artifacts, contracts, deliveries, leads, logs, projects, verifications } from "@/db/schema";
import { and, count, desc, eq } from "drizzle-orm";
import type {
  Batch,
  CapabilityAssignment,
  CheckType,
  ClarificationAnswer,
  Contract,
  Delivery,
  DodItem,
  ExecutionMode,
  Lead,
  LogEntry,
  Project,
  VerificationResult,
  VerificationStatus,
} from "./types";

export async function addLog(projectId: string, message: string): Promise<void> {
  await db.insert(logs).values({ projectId, message });
}

export async function serializeProject(id: string): Promise<Project | null> {
  const [p] = await db.select().from(projects).where(eq(projects.id, id));

  if (!p) return null;

  const contractRows = await db.select().from(contracts).where(eq(contracts.projectId, id));

  const [deliveryRow] = await db
    .select()
    .from(deliveries)
    .where(eq(deliveries.projectId, id))
    .orderBy(desc(deliveries.deliveredAt))
    .limit(1);

  const logRows = await db
    .select()
    .from(logs)
    .where(eq(logs.projectId, id))
    .orderBy(desc(logs.createdAt));

  const verRows = await db
    .select()
    .from(verifications)
    .where(eq(verifications.projectId, id))
    .orderBy(desc(verifications.createdAt));

  const verificationsByDod: Record<string, VerificationResult> = {};

  for (const v of verRows) {
    if (!verificationsByDod[v.dodId]) {
      verificationsByDod[v.dodId] = {
        dodId: v.dodId,
        checkType: v.checkType as CheckType,
        status: v.status as VerificationStatus,
        detail: v.detail ?? "",
        createdAt: v.createdAt.toISOString(),
      };
    }
  }

  const verValues = Object.values(verificationsByDod);

  const verificationSummary =
    verValues.length > 0
      ? {
          total: verValues.length,
          pass: verValues.filter((v) => v.status === "pass").length,
          fail: verValues.filter((v) => v.status === "fail").length,
          manual: verValues.filter((v) => v.status === "manual" || v.status === "error").length,
          lastRunAt: verValues.reduce((max, v) => (v.createdAt > max ? v.createdAt : max), ""),
        }
      : null;

  const [artifactRow] = await db
    .select()
    .from(artifacts)
    .where(eq(artifacts.projectId, id))
    .orderBy(desc(artifacts.version))
    .limit(1);

  const [leadCountRow] = await db
    .select({ value: count() })
    .from(leads)
    .where(and(eq(leads.projectId, id), eq(leads.isTest, false)));

  let contract: Contract | null = null;

  const c = contractRows[0];

  if (c) {
    contract = {
      intent: c.intent,
      user_summary: c.userSummary,
      clarification_questions: c.clarificationQuestions ?? [],
      clarification_answers: (c.clarificationAnswers ?? []) as ClarificationAnswer[],
      included: c.included ?? [],
      excluded: c.excluded ?? [],
      capabilities: (c.capabilities ?? []) as CapabilityAssignment[],
      definition_of_done: (c.definitionOfDone ?? []) as DodItem[],
      batches: (c.batches ?? []) as Batch[],
      source: (c.source as "mock" | "llm") ?? "mock",
      confirmation_status: (c.confirmationStatus as "pending" | "confirmed") ?? "pending",
      confirmed_at: c.confirmedAt ? c.confirmedAt.toISOString() : null,
    };
  }

  let delivery: Delivery | null = null;

  if (deliveryRow) {
    delivery = {
      previewUrl: deliveryRow.previewUrl,
      summary: deliveryRow.summary,
      deliveredAt: deliveryRow.deliveredAt.toISOString(),
    };
  }

  return {
    id: p.id,
    prompt: p.prompt,
    status: p.status as Project["status"],
    executionMode: (p.executionMode as ExecutionMode | null) ?? null,
    createdAt: p.createdAt.toISOString(),
    updatedAt: p.updatedAt.toISOString(),
    contract,
    delivery,
    verifications: verificationsByDod,
    verificationSummary,
    agentBuilt: Boolean(artifactRow),
    artifactSource: artifactRow ? ((artifactRow.source as "template" | "llm") ?? "template") : null,
    leadsCount: Number(leadCountRow?.value ?? 0),
    logs: logRows.map((l) => ({
      id: l.id,
      message: l.message,
      createdAt: l.createdAt.toISOString(),
    })) as LogEntry[],
  };
}

export async function listProjects(): Promise<Project[]> {
  const rows = await db.select().from(projects).orderBy(desc(projects.createdAt));
  const serialized = await Promise.all(rows.map((row) => serializeProject(row.id)));
  return serialized.filter((p): p is Project => p !== null);
}

export async function getLatestArtifact(
  projectId: string,
): Promise<{ html: string; version: number } | null> {
  const [row] = await db
    .select()
    .from(artifacts)
    .where(eq(artifacts.projectId, projectId))
    .orderBy(desc(artifacts.version))
    .limit(1);

  return row ? { html: row.html, version: row.version } : null;
}

export async function getNextArtifactVersion(projectId: string): Promise<number> {
  const [row] = await db
    .select()
    .from(artifacts)
    .where(eq(artifacts.projectId, projectId))
    .orderBy(desc(artifacts.version))
    .limit(1);

  return row ? row.version + 1 : 1;
}

export async function addLead(
  projectId: string,
  payload: Record<string, string>,
  isTest = false,
): Promise<void> {
  await db.insert(leads).values({ projectId, payload, isTest });
}

export async function listLeads(projectId: string, includeTest = false): Promise<Lead[]> {
  const rows = await db
    .select()
    .from(leads)
    .where(
      includeTest
        ? eq(leads.projectId, projectId)
        : and(eq(leads.projectId, projectId), eq(leads.isTest, false)),
    )
    .orderBy(desc(leads.createdAt));

  return rows.map((r) => ({
    id: r.id,
    payload: r.payload ?? {},
    isTest: r.isTest,
    createdAt: r.createdAt.toISOString(),
  }));
}
