"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import type { Lead, Project, ProjectStatus } from "@/lib/types";

const EXAMPLES = [
  "Saya butuh halaman checklist SOP onboarding karyawan baru tanpa form.",
  "Buat form pengajuan perbaikan aset internal untuk tim operasional.",
  "Saya ingin katalog layanan konsultasi dengan daftar paket dan form permintaan.",
];

const STATUS_LABELS: Record<ProjectStatus, string> = {
  DRAFT: "Draf",
  AWAITING_CLARIFICATION: "Perlu Klarifikasi",
  CONTRACT_READY: "Kontrak Siap",
  CONFIRMED: "Dikonfirmasi",
  RUNNING: "Sedang Dibangun",
  COMPLETED: "Selesai",
  FAILED: "Gagal",
};

async function api<T>(path: string, method = "GET", body?: unknown): Promise<T> {
  const res = await fetch(path, {
    method,
    headers: body ? { "Content-Type": "application/json" } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const err = (await res.json().catch(() => ({ error: "Request gagal." }))) as { error?: string };
    throw new Error(err.error || "Request gagal.");
  }

  return (await res.json()) as T;
}

function fullTime(iso: string): string {
  return new Date(iso).toLocaleString("id-ID", {
    day: "2-digit",
    month: "short",
    hour: "2-digit",
    minute: "2-digit",
  });
}

export default function HarnessApp({
  llmEnabled,
  pilotProtected,
}: {
  llmEnabled: boolean;
  pilotProtected: boolean;
}) {
  const [prompt, setPrompt] = useState("");
  const [generating, setGenerating] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [project, setProject] = useState<Project | null>(null);
  const [projects, setProjects] = useState<Project[]>([]);
  const [deliverUrl, setDeliverUrl] = useState("");
  const [deliverSummary, setDeliverSummary] = useState("");
  const [verifyLoading, setVerifyLoading] = useState(false);
  const [agentLoading, setAgentLoading] = useState(false);
  const [clarifyAnswers, setClarifyAnswers] = useState<string[]>([]);
  const [clarifyLoading, setClarifyLoading] = useState(false);
  const [leads, setLeads] = useState<Lead[] | null>(null);
  const [leadsLoading, setLeadsLoading] = useState(false);
  const [deletingId, setDeletingId] = useState<string | null>(null);

  const clarifySigRef = useRef("");

  const applyProject = useCallback((next: Project) => {
    const questions = next.contract?.clarification_questions ?? [];
    const signature = `${next.id}|${questions.join("\u0000")}`;

    if (signature !== clarifySigRef.current) {
      clarifySigRef.current = signature;
      setClarifyAnswers(questions.map(() => ""));
    }

    setProject(next);
  }, []);

  const loadProjects = useCallback(async () => {
    try {
      setProjects(await api<Project[]>("/api/projects"));
    } catch {
      // ignore
    }
  }, []);

  const refreshCurrent = useCallback(
    async (id: string) => {
      try {
        applyProject(await api<Project>(`/api/projects/${id}`));
      } catch {
        // ignore
      }
    },
    [applyProject],
  );

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- setState terjadi setelah await fetch (async), bukan sinkron saat mount
    void loadProjects();
  }, [loadProjects]);

  const activeProjectId = project?.id;
  const activeProjectStatus = project?.status;

  useEffect(() => {
    if (!activeProjectId) return;
    if (activeProjectStatus === "COMPLETED" || activeProjectStatus === "FAILED") return;

    const t = setInterval(() => {
      void refreshCurrent(activeProjectId);
      void loadProjects();
    }, 3000);

    return () => clearInterval(t);
  }, [activeProjectId, activeProjectStatus, refreshCurrent, loadProjects]);

  async function handleGenerate() {
    const value = prompt.trim();

    if (!value) {
      setError("Tulis kebutuhan dulu.");
      return;
    }

    setGenerating(true);
    setError(null);

    try {
      const created = await api<Project>("/api/generate", "POST", { prompt: value });
      applyProject(created);
      setLeads(null);
      setDeliverUrl("");
      setDeliverSummary("");
      await loadProjects();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal generate kontrak.");
      await loadProjects();
    } finally {
      setGenerating(false);
    }
  }

  async function handleConfirm() {
    if (!project) return;

    setBusy(true);
    setError(null);

    try {
      applyProject(await api<Project>(`/api/projects/${project.id}/confirm`, "POST"));
      await loadProjects();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal konfirmasi.");
    } finally {
      setBusy(false);
    }
  }

  async function handleDeliver() {
    if (!project) return;

    const url = deliverUrl.trim();

    if (!url) {
      setError("URL preview wajib diisi.");
      return;
    }

    setBusy(true);
    setError(null);

    try {
      applyProject(
        await api<Project>(`/api/projects/${project.id}/deliver`, "POST", {
          preview_url: url,
          summary: deliverSummary.trim(),
        }),
      );

      setDeliverUrl("");
      setDeliverSummary("");
      await loadProjects();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal mengirim hasil.");
    } finally {
      setBusy(false);
    }
  }

  async function handleVerify() {
    if (!project) return;

    setVerifyLoading(true);
    setError(null);

    try {
      applyProject(await api<Project>(`/api/projects/${project.id}/verify`, "POST"));
      await loadProjects();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal menjalankan verifier.");
    } finally {
      setVerifyLoading(false);
    }
  }

  async function handleRunAgent() {
    if (!project) return;

    setAgentLoading(true);
    setError(null);

    try {
      applyProject(await api<Project>(`/api/projects/${project.id}/run`, "POST"));
      await loadProjects();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal menjalankan agent.");
    } finally {
      setAgentLoading(false);
    }
  }

  async function handleClarify() {
    if (!project) return;

    setClarifyLoading(true);
    setError(null);

    try {
      applyProject(await api<Project>(`/api/projects/${project.id}/clarify`, "POST", { answers: clarifyAnswers }));
      await loadProjects();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal memperbarui kontrak.");
    } finally {
      setClarifyLoading(false);
    }
  }

  async function handleLoadLeads() {
    if (!project) return;

    if (leads !== null) {
      setLeads(null);
      return;
    }

    setLeadsLoading(true);
    setError(null);

    try {
      setLeads(await api<Lead[]>(`/api/projects/${project.id}/leads`));
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal mengambil leads.");
    } finally {
      setLeadsLoading(false);
    }
  }

  async function handleDeleteProject(id: string) {
    if (typeof window !== "undefined" && !window.confirm("Hapus project ini beserta semua datanya?")) {
      return;
    }

    setDeletingId(id);
    setError(null);

    try {
      await api<{ ok: boolean }>(`/api/projects/${id}`, "DELETE");

      if (project?.id === id) {
        setProject(null);
        setLeads(null);
        clarifySigRef.current = "";
      }

      await loadProjects();
    } catch (e) {
      setError(e instanceof Error ? e.message : "Gagal menghapus project.");
    } finally {
      setDeletingId(null);
    }
  }

  function selectProject(id: string) {
    const found = projects.find((p) => p.id === id);

    if (found) {
      applyProject(found);
      setLeads(null);
      setDeliverUrl("");
      setDeliverSummary("");
    }
  }

  const contract = project?.contract ?? null;
  const isAwaiting = project?.status === "AWAITING_CLARIFICATION";

  return (
    <div className="mx-auto max-w-6xl px-5 py-8">
      <header className="mb-8 flex flex-wrap items-center justify-between gap-3 border-b border-white/10 pb-4">
        <div>
          <p className="text-lg font-bold">HARNESS</p>
          <p className="text-xs uppercase tracking-widest text-slate-400">Capability Execution Kernel</p>
        </div>
        <div className="flex gap-2 text-xs">
          <span className="rounded-full border border-white/10 px-3 py-1">
            {llmEnabled ? "Mode LLM" : "Mock capability"}
          </span>
          <span className="rounded-full border border-white/10 px-3 py-1">
            {pilotProtected ? "Pilot gate aktif" : "API belum diproteksi"}
          </span>
        </div>
      </header>

      <main className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="space-y-6 lg:col-span-2">
          <section className="rounded-2xl border border-white/10 bg-slate-900/60 p-5">
            <label htmlFor="prompt" className="mb-2 block text-sm font-semibold">
              Ceritakan maunya apa
            </label>

            <textarea
              id="prompt"
              value={prompt}
              onChange={(e) => setPrompt(e.target.value)}
              placeholder="Contoh: Saya mau landing page untuk katering kantor dengan form permintaan penawaran."
              className="min-h-[120px] w-full resize-y rounded-xl border border-white/10 bg-slate-950/60 p-4 text-sm"
            />

            {!llmEnabled && (
              <div className="mt-3 rounded-lg border border-amber-400/20 bg-amber-500/10 px-3 py-2 text-xs text-amber-200">
                Tanpa OPENAI_API_KEY, sistem memakai capability discovery generik. Semua prompt tetap diproses,
                tetapi output template bisa lebih generik.
              </div>
            )}

            <div className="mt-3 flex flex-wrap gap-2">
              {EXAMPLES.map((ex, i) => (
                <button
                  key={i}
                  type="button"
                  onClick={() => setPrompt(ex)}
                  className="max-w-full truncate rounded-full border border-white/10 bg-white/5 px-3 py-1.5 text-left text-xs"
                >
                  {ex.length > 48 ? `${ex.slice(0, 48)}…` : ex}
                </button>
              ))}
            </div>

            <div className="mt-4 flex items-center gap-3">
              <button
                type="button"
                onClick={handleGenerate}
                disabled={generating}
                className="rounded-xl bg-indigo-500 px-5 py-2.5 text-sm font-semibold text-white disabled:opacity-60"
              >
                {generating ? "Menerjemahkan…" : "Generate Execution Contract"}
              </button>
              {error && <span className="text-sm text-rose-300">{error}</span>}
            </div>
          </section>

          <section className="rounded-2xl border border-white/10 bg-slate-900/60 p-5">
            <div className="mb-4 flex items-center justify-between">
              <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400">Execution Contract</h2>
              {contract && <span className="text-xs">{contract.source === "llm" ? "LLM" : "Mock"}</span>}
            </div>

            {!contract ? (
              <p className="text-sm text-slate-400">Belum ada kontrak.</p>
            ) : (
              <div className="space-y-5">
                <div className="rounded-xl border border-indigo-400/20 bg-indigo-500/5 p-4">
                  <p className="text-sm">{contract.user_summary}</p>
                  <p className="mt-2 text-xs italic text-slate-400">Intent: {contract.intent}</p>
                </div>

                {contract.clarification_questions.length > 0 && (
                  <div className="rounded-xl border border-amber-400/30 bg-amber-500/10 p-4">
                    <p className="mb-3 text-sm font-semibold text-amber-200">Butuh keputusan bisnis</p>

                    <div className="space-y-3">
                      {contract.clarification_questions.map((q, i) => (
                        <div key={i}>
                          <label className="mb-1 block text-xs">
                            {i + 1}. {q}
                          </label>
                          <textarea
                            value={clarifyAnswers[i] ?? ""}
                            onChange={(e) =>
                              setClarifyAnswers((prev) => {
                                const next = [...prev];
                                next[i] = e.target.value;
                                return next;
                              })
                            }
                            className="min-h-[56px] w-full rounded-lg border border-amber-400/20 bg-slate-950/50 p-2 text-sm"
                          />
                        </div>
                      ))}
                    </div>

                    <button
                      type="button"
                      onClick={handleClarify}
                      disabled={clarifyLoading}
                      className="mt-3 rounded-xl bg-amber-500 px-4 py-2 text-sm font-semibold text-slate-950 disabled:opacity-60"
                    >
                      {clarifyLoading ? "Memperbarui kontrak…" : "Kirim Jawaban & Perbarui Kontrak"}
                    </button>
                  </div>
                )}

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div className="rounded-xl border border-emerald-400/20 bg-emerald-500/5 p-4">
                    <p className="mb-2 text-sm font-semibold text-emerald-200">Yang dibuat</p>
                    <ul className="list-inside list-disc space-y-1 text-sm">
                      {(contract.included.length ? contract.included : ["-"]).map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="rounded-xl border border-rose-400/20 bg-rose-500/5 p-4">
                    <p className="mb-2 text-sm font-semibold text-rose-200">Yang sengaja tidak dibuat</p>
                    <ul className="list-inside list-disc space-y-1 text-sm">
                      {(contract.excluded.length ? contract.excluded : ["-"]).map((item, i) => (
                        <li key={i}>{item}</li>
                      ))}
                    </ul>
                  </div>
                </div>

                <div>
                  <p className="mb-2 text-sm font-semibold">Definition of Done</p>
                  <ul className="space-y-2">
                    {contract.definition_of_done.map((item) => {
                      const v = project?.verifications?.[item.id];

                      return (
                        <li key={item.id} className="rounded-lg border border-white/5 bg-slate-950/40 p-3">
                          <div className="flex items-start justify-between gap-3">
                            <div>
                              <span className="font-mono text-[11px] text-slate-400">{item.id}</span>
                              <p className="text-sm">{item.description}</p>
                            </div>
                            <span className="text-[10px] uppercase text-slate-400">{item.check_type}</span>
                          </div>

                          {v && (
                            <p className="mt-2 text-xs text-slate-400">
                              <span className="mr-2 font-semibold uppercase">{v.status}</span>
                              {v.detail}
                            </p>
                          )}
                        </li>
                      );
                    })}
                  </ul>
                </div>

                <div>
                  <p className="mb-2 text-sm font-semibold">Rencana batch</p>
                  <ol className="list-inside list-decimal space-y-2 text-sm">
                    {contract.batches.map((batch) => (
                      <li key={batch.id}>
                        <span className="font-semibold">{batch.name}</span> — {batch.goal}
                      </li>
                    ))}
                  </ol>
                </div>
              </div>
            )}
          </section>
        </div>

        <div className="space-y-6">
          <section className="rounded-2xl border border-white/10 bg-slate-900/60 p-5">
            <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-400">Status</h2>

            {!project ? (
              <p className="text-sm text-slate-400">Belum ada project aktif.</p>
            ) : (
              <div className="space-y-4">
                <div className="flex items-center justify-between gap-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="rounded-full border border-white/10 px-2.5 py-1 text-[11px]">
                      {STATUS_LABELS[project.status]}
                    </span>
                    {project.executionMode && (
                      <span className="rounded-full border border-white/10 px-2.5 py-1 text-[10px] uppercase">
                        {project.executionMode === "agent" ? "Jalur: Agent" : "Jalur: Manual"}
                      </span>
                    )}
                  </div>
                  <span className="text-[11px] text-slate-500">{fullTime(project.createdAt)}</span>
                </div>

                {contract && (
                  <div className="rounded-xl border border-white/5 bg-slate-950/40 p-4">
                    {isAwaiting ? (
                      <p className="text-xs text-amber-300">
                        Jawab pertanyaan klarifikasi di panel kontrak di sebelah.
                      </p>
                    ) : contract.confirmation_status === "pending" ? (
                      <button
                        type="button"
                        onClick={handleConfirm}
                        disabled={busy}
                        className="w-full rounded-xl bg-indigo-500 px-4 py-2.5 text-sm font-semibold disabled:opacity-60"
                      >
                        {busy ? "Memproses…" : "Confirm Contract"}
                      </button>
                    ) : (
                      <p className="text-xs font-medium text-emerald-300">✓ Kontrak sudah dikonfirmasi.</p>
                    )}
                  </div>
                )}

                {(project.status === "CONFIRMED" ||
                  (project.status === "FAILED" && project.executionMode === "agent")) && (
                  <div className="rounded-xl border border-violet-400/20 bg-violet-500/5 p-4">
                    <p className="mb-2 text-xs font-semibold text-violet-200">Agent Runner</p>
                    <button
                      type="button"
                      onClick={handleRunAgent}
                      disabled={agentLoading}
                      className="w-full rounded-xl bg-violet-500 px-4 py-2.5 text-sm font-semibold disabled:opacity-60"
                    >
                      {agentLoading
                        ? "Membangun & memverifikasi…"
                        : project.status === "FAILED"
                          ? "↻ Retry Agent Runner"
                          : "⚡ Jalankan Agent Runner"}
                    </button>
                  </div>
                )}

                {(project.status === "CONFIRMED" ||
                  (project.status === "FAILED" && project.executionMode === "manual")) && (
                  <div className="rounded-xl border border-sky-400/20 bg-sky-500/5 p-4">
                    <p className="mb-2 text-xs font-semibold text-sky-200">Atau deliver manual</p>

                    <input
                      type="url"
                      value={deliverUrl}
                      onChange={(e) => setDeliverUrl(e.target.value)}
                      placeholder="https://hasil-preview.vercel.app"
                      className="mb-2 w-full rounded-lg border border-white/10 bg-slate-950/60 px-3 py-2 text-sm"
                    />

                    <textarea
                      value={deliverSummary}
                      onChange={(e) => setDeliverSummary(e.target.value)}
                      placeholder="Ringkasan singkat hasil…"
                      className="mb-2 min-h-[60px] w-full rounded-lg border border-white/10 bg-slate-950/60 px-3 py-2 text-sm"
                    />

                    <button
                      type="button"
                      onClick={handleDeliver}
                      disabled={busy}
                      className="w-full rounded-xl bg-sky-500 px-4 py-2.5 text-sm font-semibold disabled:opacity-60"
                    >
                      {busy ? "Mengirim…" : project.status === "FAILED" ? "Perbarui Hasil Manual" : "Deliver Hasil"}
                    </button>
                  </div>
                )}

                {project.delivery && (
                  <div className="rounded-xl border border-emerald-400/30 bg-emerald-500/10 p-4">
                    <p className="mb-1 text-[11px] font-semibold uppercase text-emerald-300">Hasil siap</p>
                    <a
                      href={project.delivery.previewUrl}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="break-all text-sm font-medium underline-offset-2 hover:underline"
                    >
                      Buka hasil preview ↗
                    </a>
                    <p className="mt-2 text-xs text-slate-300">{project.delivery.summary}</p>

                    <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                      <span className="rounded bg-sky-500/15 px-2 py-0.5">{project.leadsCount} lead asli</span>
                      <button
                        type="button"
                        onClick={handleLoadLeads}
                        disabled={leadsLoading}
                        className="rounded bg-white/10 px-2 py-0.5 disabled:opacity-60"
                      >
                        {leadsLoading ? "Memuat…" : leads === null ? "Lihat leads" : "Tutup leads"}
                      </button>
                    </div>

                    {leads !== null && (
                      <div className="mt-3 max-h-64 space-y-2 overflow-y-auto border-t border-white/10 pt-3">
                        {leads.length === 0 ? (
                          <p className="text-xs text-slate-400">Belum ada lead asli.</p>
                        ) : (
                          leads.map((lead) => (
                            <div key={lead.id} className="rounded-lg bg-slate-950/50 p-2 text-xs">
                              <p className="mb-1 font-mono text-[10px] text-slate-500">{fullTime(lead.createdAt)}</p>
                              {Object.entries(lead.payload).map(([key, value]) => (
                                <p key={key} className="break-words">
                                  <span className="text-slate-500">{key}:</span> {value}
                                </p>
                              ))}
                            </div>
                          ))
                        )}
                      </div>
                    )}
                  </div>
                )}

                {contract && project.delivery && (
                  <div className="rounded-xl border border-white/5 bg-slate-950/40 p-4">
                    <div className="mb-3 flex items-center justify-between gap-2">
                      <p className="text-[11px] font-semibold uppercase text-slate-400">Verifier (DoD)</p>
                      <button
                        type="button"
                        onClick={handleVerify}
                        disabled={verifyLoading}
                        className="rounded-lg bg-white/10 px-3 py-1.5 text-xs font-semibold disabled:opacity-60"
                      >
                        {verifyLoading ? "Memeriksa…" : "Jalankan Verifier"}
                      </button>
                    </div>

                    {project.verificationSummary ? (
                      <div className="flex flex-wrap gap-2 text-[11px]">
                        <span className="rounded bg-emerald-500/15 px-2 py-1">
                          {project.verificationSummary.pass} lolos
                        </span>
                        <span className="rounded bg-rose-500/15 px-2 py-1">
                          {project.verificationSummary.fail} gagal
                        </span>
                        <span className="rounded bg-amber-500/15 px-2 py-1">
                          {project.verificationSummary.manual} manual
                        </span>
                        <span className="rounded bg-white/5 px-2 py-1">/ {project.verificationSummary.total}</span>
                      </div>
                    ) : (
                      <p className="text-xs text-slate-500">Jalankan verifier untuk mengecek DoD otomatis.</p>
                    )}
                  </div>
                )}
              </div>
            )}
          </section>

          <section className="rounded-2xl border border-white/10 bg-slate-900/60 p-5">
            <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-slate-400">Log Harness</h2>
            <div className="max-h-72 space-y-1.5 overflow-y-auto pr-1">
              {(!project || project.logs.length === 0) && <p className="text-sm text-slate-500">Belum ada log.</p>}
              {project?.logs.map((log) => (
                <div key={log.id} className="rounded-md bg-slate-950/40 px-3 py-2 text-xs">
                  <span className="font-mono text-slate-500">[{fullTime(log.createdAt)}]</span>{" "}
                  <span>{log.message}</span>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>

      <section className="mt-8 rounded-2xl border border-white/10 bg-slate-900/60 p-5">
        <h2 className="mb-4 text-sm font-semibold uppercase tracking-wider text-slate-400">Riwayat Project</h2>

        {projects.length === 0 ? (
          <p className="text-sm text-slate-500">Belum ada project.</p>
        ) : (
          <ul className="divide-y divide-white/5">
            {projects.map((p) => (
              <li key={p.id} className="flex items-center gap-2 py-1">
                <button
                  type="button"
                  onClick={() => selectProject(p.id)}
                  className={`flex min-w-0 flex-1 items-center justify-between gap-4 rounded-lg px-3 py-3 text-left transition hover:bg-white/5 ${
                    project?.id === p.id ? "bg-white/5" : ""
                  }`}
                >
                  <div className="min-w-0">
                    <p className="truncate text-sm">{p.prompt}</p>
                    <p className="mt-0.5 text-[11px] text-slate-500">
                      {fullTime(p.createdAt)}
                      {p.executionMode ? ` · ${p.executionMode === "agent" ? "Agent" : "Manual"}` : ""}
                    </p>
                  </div>
                  <span className="rounded-full border border-white/10 px-2.5 py-1 text-[11px]">
                    {STATUS_LABELS[p.status]}
                  </span>
                </button>

                <button
                  type="button"
                  onClick={() => handleDeleteProject(p.id)}
                  disabled={deletingId === p.id}
                  className="shrink-0 rounded-lg border border-rose-400/20 px-3 py-3 text-xs text-rose-300 disabled:opacity-50"
                >
                  {deletingId === p.id ? "…" : "Hapus"}
                </button>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
