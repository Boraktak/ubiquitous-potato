# HARNESS — Capability Execution Kernel (Hardened)

Refactored pilot: capability-based execution system.

## Pipeline
```text
prompt
→ capability discovery
→ Execution Contract
→ artifact (template/LLM)
→ sanitize + ensureInvariants
→ verifyContract
→ repairArtifact
```

## Security hardening
- Next middleware auth + rate limit + origin guard.
- SSRF guard untuk manual deliver dan verifier.
- HTML sanitization + CSP untuk preview.
- Escape prompt/user text pada template.
- Timeout untuk LLM calls.
- Limit clarification, capabilities, DoD, dan lead payload.

## Offline generalization smoke test
```bash
node scripts/generalization-proof.mjs
```

## Deploy
Lihat `DEPLOYMENT.md`.
