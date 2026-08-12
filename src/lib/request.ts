export function getPublicOrigin(req: Request): string {
  const configured = process.env.APP_BASE_URL?.trim();

  if (configured) {
    const parsed = new URL(configured);

    if (!["http:", "https:"].includes(parsed.protocol)) {
      throw new Error("APP_BASE_URL harus memakai http:// atau https://.");
    }

    return parsed.origin;
  }

  const forwardedHost = req.headers.get("x-forwarded-host")?.split(",")[0]?.trim();
  const host = forwardedHost || req.headers.get("host");
  const forwardedProto = req.headers.get("x-forwarded-proto")?.split(",")[0]?.trim();

  if (host) {
    const protocol =
      forwardedProto || (host.includes("localhost") || host.startsWith("127.") ? "http" : "https");

    return `${protocol}://${host}`;
  }

  const requestUrl = new URL(req.url);
  return requestUrl.origin;
}
