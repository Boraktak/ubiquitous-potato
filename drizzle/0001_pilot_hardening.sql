CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS projects (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  prompt text NOT NULL,
  status text NOT NULL DEFAULT 'DRAFT',
  execution_mode text,
  created_at timestamptz NOT NULL DEFAULT now(),
  updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE projects ADD COLUMN IF NOT EXISTS execution_mode text;

CREATE TABLE IF NOT EXISTS contracts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  intent text NOT NULL,
  user_summary text NOT NULL,
  clarification_questions jsonb NOT NULL DEFAULT '[]'::jsonb,
  clarification_answers jsonb NOT NULL DEFAULT '[]'::jsonb,
  included jsonb NOT NULL DEFAULT '[]'::jsonb,
  excluded jsonb NOT NULL DEFAULT '[]'::jsonb,
  capabilities jsonb NOT NULL DEFAULT '[]'::jsonb,
  definition_of_done jsonb NOT NULL DEFAULT '[]'::jsonb,
  batches jsonb NOT NULL DEFAULT '[]'::jsonb,
  source text NOT NULL DEFAULT 'mock',
  confirmation_status text NOT NULL DEFAULT 'pending',
  confirmed_at timestamptz,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE contracts ADD COLUMN IF NOT EXISTS capabilities jsonb NOT NULL DEFAULT '[]'::jsonb;

CREATE TABLE IF NOT EXISTS logs (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  message text NOT NULL,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS deliveries (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  preview_url text NOT NULL,
  summary text NOT NULL,
  delivered_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS verifications (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  dod_id text NOT NULL,
  check_type text NOT NULL,
  status text NOT NULL,
  detail text,
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS artifacts (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  html text NOT NULL,
  version integer NOT NULL DEFAULT 1,
  source text NOT NULL DEFAULT 'template',
  created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS leads (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  project_id uuid NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
  payload jsonb NOT NULL DEFAULT '{}'::jsonb,
  is_test boolean NOT NULL DEFAULT false,
  created_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE leads ADD COLUMN IF NOT EXISTS is_test boolean NOT NULL DEFAULT false;

CREATE INDEX IF NOT EXISTS contracts_project_idx ON contracts(project_id);
CREATE INDEX IF NOT EXISTS logs_project_idx ON logs(project_id);
CREATE INDEX IF NOT EXISTS deliveries_project_idx ON deliveries(project_id);
CREATE INDEX IF NOT EXISTS verifications_project_idx ON verifications(project_id);
CREATE INDEX IF NOT EXISTS artifacts_project_version_idx ON artifacts(project_id, version);
CREATE INDEX IF NOT EXISTS leads_project_test_idx ON leads(project_id, is_test);
