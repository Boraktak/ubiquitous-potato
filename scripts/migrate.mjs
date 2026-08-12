import "dotenv/config";
import { readFile } from "node:fs/promises";
import pg from "pg";

if (!process.env.DATABASE_URL) throw new Error("DATABASE_URL wajib diisi.");

const sql = await readFile(new URL("../drizzle/0001_pilot_hardening.sql", import.meta.url), "utf8");
const pool = new pg.Pool({ connectionString: process.env.DATABASE_URL });

try {
  await pool.query(sql);
  console.log("Database HARNESS siap.");
} finally {
  await pool.end();
}
