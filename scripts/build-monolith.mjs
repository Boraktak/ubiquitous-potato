import { readdir, readFile, writeFile } from "node:fs/promises";
import path from "node:path";

const root = process.cwd();
const outputName = "MONOLITH_CAPABILITY.txt";

const excludedDirs = new Set([".git", ".next", "node_modules", "build", "dist", "coverage"]);
const excludedFiles = new Set([
  outputName,
  "MONOLITH.txt",
  "package-lock.json",
  "next-env.d.ts",
  ".env",
  ".env.local",
]);

const allowedExtensions = new Set([
  ".ts",
  ".tsx",
  ".js",
  ".mjs",
  ".json",
  ".css",
  ".md",
  ".sql",
  ".example",
]);

async function walk(relative = "") {
  const entries = await readdir(path.join(root, relative), { withFileTypes: true });
  const files = [];

  for (const entry of entries.sort((a, b) => a.name.localeCompare(b.name))) {
    const rel = path.join(relative, entry.name);

    if (entry.isDirectory()) {
      if (!excludedDirs.has(entry.name)) files.push(...(await walk(rel)));
    } else if (!excludedFiles.has(entry.name) && allowedExtensions.has(path.extname(entry.name))) {
      files.push(rel.split(path.sep).join("/"));
    }
  }

  return files;
}

const files = await walk();
const contents = await Promise.all(files.map((file) => readFile(path.join(root, file), "utf8")));
const lineCounts = contents.map((content) => content.replace(/\r?\n$/, "").split(/\r?\n/).length);

const width = Math.max(...files.map((file) => file.length), 4) + 4;
const totalLines = lineCounts.reduce((sum, count) => sum + count, 0);
const divider = "#".repeat(80);
const sectionDivider = "=".repeat(80);

const chunks = [
  divider,
  "# HARNESS — capability execution kernel (hardened)",
  "# Monolith source dump (diperbaiki). Setiap file dipisahkan pembatas jelas.",
  `# Generated: ${new Date().toISOString()}`,
  `# Total file: ${files.length}`,
  divider,
  "",
  sectionDivider,
  "TABLE OF CONTENTS",
  sectionDivider,
  `${"#".padEnd(6)}${"FILE".padEnd(width)}LINES`,
  ...files.map((file, index) => `${String(index + 1).padEnd(6)}${file.padEnd(width)}${lineCounts[index]}`),
  `${"".padEnd(6)}${"TOTAL".padEnd(width)}${totalLines}`,
  "",
  sectionDivider,
  "SOURCE FILES",
  sectionDivider,
  "",
];

for (let i = 0; i < files.length; i++) {
  chunks.push(divider, `### FILE: ${files[i]}`, divider, contents[i].replace(/\r?\n$/, ""), "");
}

await writeFile(path.join(root, outputName), `${chunks.join("\n")}\n`);
console.log(`${outputName}: ${files.length} files, ${totalLines} source lines`);
