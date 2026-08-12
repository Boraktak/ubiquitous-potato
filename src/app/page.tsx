import HarnessApp from "@/components/HarnessApp";

export const dynamic = "force-dynamic";

export default function HomePage() {
  const llmEnabled = Boolean(process.env.OPENAI_API_KEY);
  const pilotProtected = Boolean(process.env.PILOT_ACCESS_KEY);

  return <HarnessApp llmEnabled={llmEnabled} pilotProtected={pilotProtected} />;
}
