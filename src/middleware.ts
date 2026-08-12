import { NextRequest, NextResponse } from "next/server";

interface RateBucket {
  count: number;
  resetAt: number;
}

const buckets = new Map<string, RateBucket>();

const WINDOW_MS = Number(process.env.RATE_LIMIT_WINDOW_MS || 10 * 60 * 1000);
const TRUST_PROXY = process.env.TRUST_PROXY !== "0";

function constantTimeEqual(a: string, b: string): boolean {
  const max = Math.max(a.length, b.length);
  let diff = a.length ^ b.length;

  for (let i = 0; i < max; i++) {
    diff |= (a.charCodeAt(i) || 0) ^ (b.charCodeAt(i) || 0);
  }

  return diff === 0;
}

function hasPilotAccess(request: NextRequest): boolean {
  const expectedPassword = process.env.PILOT_ACCESS_KEY;

  if (!expectedPassword) return true;

  const auth = request.headers.get("authorization");

  if (!auth?.startsWith("Basic ")) return false;

  try {
    const decoded = atob(auth.slice(6));
    const splitAt = decoded.indexOf(":");
    const username = splitAt >= 0 ? decoded.slice(0, splitAt) : "";
    const password = splitAt >= 0 ? decoded.slice(splitAt + 1) : "";
    const expectedUsername = process.env.PILOT_USERNAME || "harness";

    return (
      constantTimeEqual(username, expectedUsername) &&
      constantTimeEqual(password, expectedPassword)
    );
  } catch {
    return false;
  }
}

function clientIp(request: NextRequest): string {
  if (!TRUST_PROXY) return "untrusted-client";

  return (
    request.headers.get("x-forwarded-for")?.split(",")[0]?.trim() ||
    request.headers.get("x-real-ip") ||
    "unknown"
  );
}

function rateRule(
  request: NextRequest,
): { name: string; limit: number } | null {
  if (!["POST", "PUT", "PATCH", "DELETE"].includes(request.method)) {
    return null;
  }

  const path = request.nextUrl.pathname;

  if (/\/api\/projects\/[^/]+\/preview$/.test(path)) {
    return {
      name: "preview-submit",
      limit: Number(process.env.RATE_LIMIT_PREVIEW || 20),
    };
  }

  if (path === "/api/generate") {
    return {
      name: "generate",
      limit: Number(process.env.RATE_LIMIT_GENERATE || 10),
    };
  }

  if (path.startsWith("/api/")) {
    return {
      name: "api-mutation",
      limit: Number(process.env.RATE_LIMIT_MUTATION || 60),
    };
  }

  return null;
}

async function tryUpstashRateLimit(
  key: string,
  limit: number,
  windowMs: number,
): Promise<{ limited: boolean; retryAfter: number } | null> {
  const url = process.env.UPSTASH_REDIS_REST_URL;
  const token = process.env.UPSTASH_REDIS_REST_TOKEN;

  if (!url || !token) return null;

  try {
    const now = Date.now();
    const windowIndex = Math.floor(now / windowMs);
    const redisKey = `harness:rl:${key}:${windowIndex}`;
    const ttlSeconds = Math.ceil(windowMs / 1000);

    const res = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${token}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify([
        ["INCR", redisKey],
        ["EXPIRE", redisKey, ttlSeconds],
      ]),
    });

    if (!res.ok) return null;

    const data = (await res.json()) as unknown;
    const first = Array.isArray(data) ? data[0] : data;
    const count = Number((first as { result?: string | number } | null)?.result ?? 0);

    if (count <= limit) return { limited: false, retryAfter: 0 };

    const retryAfter = Math.max(
      1,
      Math.ceil(((windowIndex + 1) * windowMs - now) / 1000),
    );

    return { limited: true, retryAfter };
  } catch {
    return null;
  }
}

function inMemoryRateLimit(key: string, limit: number): NextResponse | null {
  const now = Date.now();
  const current = buckets.get(key);

  const bucket =
    !current || current.resetAt <= now
      ? { count: 0, resetAt: now + WINDOW_MS }
      : current;

  bucket.count += 1;
  buckets.set(key, bucket);

  if (buckets.size > 5_000) {
    for (const [candidate, value] of buckets) {
      if (value.resetAt <= now) buckets.delete(candidate);
      if (buckets.size <= 4_000) break;
    }
  }

  if (bucket.count <= limit) return null;

  const retryAfter = Math.max(1, Math.ceil((bucket.resetAt - now) / 1_000));

  return NextResponse.json(
    { error: "Terlalu banyak request. Coba lagi setelah rate limit di-reset." },
    {
      status: 429,
      headers: {
        "Retry-After": String(retryAfter),
        "X-RateLimit-Limit": String(limit),
        "X-RateLimit-Remaining": "0",
      },
    },
  );
}

async function applyRateLimit(request: NextRequest): Promise<NextResponse | null> {
  const rule = rateRule(request);

  if (!rule) return null;

  const ip = clientIp(request);
  const key = `${rule.name}:${ip}`;

  const upstash = await tryUpstashRateLimit(key, rule.limit, WINDOW_MS);

  if (upstash) {
    if (!upstash.limited) return null;

    return NextResponse.json(
      { error: "Terlalu banyak request. Coba lagi setelah rate limit di-reset." },
      {
        status: 429,
        headers: {
          "Retry-After": String(upstash.retryAfter),
          "X-RateLimit-Limit": String(rule.limit),
          "X-RateLimit-Remaining": "0",
        },
      },
    );
  }

  return inMemoryRateLimit(key, rule.limit);
}

function originAllowed(request: NextRequest): boolean {
  const origin = request.headers.get("origin");
  const referer = request.headers.get("referer");
  const target = request.nextUrl.origin;

  if (!origin && !referer) return true;

  if (origin) {
    try {
      if (new URL(origin).origin === target) return true;
    } catch {
      // ignore
    }
  }

  if (referer) {
    try {
      if (new URL(referer).origin === target) return true;
    } catch {
      // ignore
    }
  }

  return false;
}

export async function middleware(request: NextRequest) {
  const limited = await applyRateLimit(request);
  if (limited) return limited;

  const method = request.method;
  const path = request.nextUrl.pathname;

  const isHealth = path === "/api/health" && method === "GET";

  const isPublicPreview =
    /\/api\/projects\/[0-9a-f-]+\/preview$/.test(path) &&
    (method === "GET" || method === "POST");

  const isApi = path.startsWith("/api/");

  if (
    isApi &&
    ["POST", "PUT", "PATCH", "DELETE"].includes(method)
  ) {
    if (!originAllowed(request)) {
      return NextResponse.json(
        { error: "Origin request tidak diizinkan." },
        { status: 403 },
      );
    }
  }

  if (!isHealth && !isPublicPreview && !hasPilotAccess(request)) {
    if (isApi) {
      return NextResponse.json(
        { error: "Pilot access key diperlukan." },
        {
          status: 401,
          headers: {
            "WWW-Authenticate": 'Basic realm="HARNESS Pilot", charset="UTF-8"',
          },
        },
      );
    }

    return new NextResponse("Akses pilot diperlukan.", {
      status: 401,
      headers: {
        "WWW-Authenticate": 'Basic realm="HARNESS Pilot", charset="UTF-8"',
      },
    });
  }

  return NextResponse.next();
}

export const config = {
  matcher: ["/((?!_next/static|_next/image|favicon.ico).*)"],
};
