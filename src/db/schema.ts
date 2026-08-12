import { boolean, index, integer, jsonb, pgTable, text, timestamp, uuid } from "drizzle-orm/pg-core";
import type { Batch, CapabilityAssignment, DodItem } from "@/lib/types";

export const projects = pgTable("projects", {
  id: uuid("id").primaryKey().defaultRandom(),
  prompt: text("prompt").notNull(),
  status: text("status").notNull().default("DRAFT"),
  executionMode: text("execution_mode"),
  createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  updatedAt: timestamp("updated_at", { withTimezone: true }).notNull().defaultNow(),
});

export const contracts = pgTable(
  "contracts",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    projectId: uuid("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    intent: text("intent").notNull(),
    userSummary: text("user_summary").notNull(),
    clarificationQuestions: jsonb("clarification_questions").$type<string[]>().notNull().default([]),
    clarificationAnswers: jsonb("clarification_answers")
      .$type<Array<{ question: string; answer: string }>>()
      .notNull()
      .default([]),
    included: jsonb("included").$type<string[]>().notNull().default([]),
    excluded: jsonb("excluded").$type<string[]>().notNull().default([]),
    capabilities: jsonb("capabilities").$type<CapabilityAssignment[]>().notNull().default([]),
    definitionOfDone: jsonb("definition_of_done").$type<DodItem[]>().notNull().default([]),
    batches: jsonb("batches").$type<Batch[]>().notNull().default([]),
    source: text("source").notNull().default("mock"),
    confirmationStatus: text("confirmation_status").notNull().default("pending"),
    confirmedAt: timestamp("confirmed_at", { withTimezone: true }),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("contracts_project_idx").on(table.projectId)],
);

export const logs = pgTable(
  "logs",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    projectId: uuid("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    message: text("message").notNull(),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("logs_project_idx").on(table.projectId)],
);

export const deliveries = pgTable(
  "deliveries",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    projectId: uuid("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    previewUrl: text("preview_url").notNull(),
    summary: text("summary").notNull(),
    deliveredAt: timestamp("delivered_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("deliveries_project_idx").on(table.projectId)],
);

export const verifications = pgTable(
  "verifications",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    projectId: uuid("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    dodId: text("dod_id").notNull(),
    checkType: text("check_type").notNull(),
    status: text("status").notNull(),
    detail: text("detail"),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("verifications_project_idx").on(table.projectId)],
);

export const artifacts = pgTable(
  "artifacts",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    projectId: uuid("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    html: text("html").notNull(),
    version: integer("version").notNull().default(1),
    source: text("source").notNull().default("template"),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("artifacts_project_version_idx").on(table.projectId, table.version)],
);

export const leads = pgTable(
  "leads",
  {
    id: uuid("id").primaryKey().defaultRandom(),
    projectId: uuid("project_id")
      .notNull()
      .references(() => projects.id, { onDelete: "cascade" }),
    payload: jsonb("payload").$type<Record<string, string>>().notNull().default({}),
    isTest: boolean("is_test").notNull().default(false),
    createdAt: timestamp("created_at", { withTimezone: true }).notNull().defaultNow(),
  },
  (table) => [index("leads_project_test_idx").on(table.projectId, table.isTest)],
);
