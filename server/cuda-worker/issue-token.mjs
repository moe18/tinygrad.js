// Issue a tinygrad.js remote CUDA session token from a Node.js backend (same format as tgjs_cuda_worker/tokens.py).
//   import { issueSessionToken } from "./issue-token.mjs";
//   res.json({ token: issueSessionToken(process.env.TGJS_CUDA_TOKEN_SECRET, user.id, { ttlS: 900, memoryBytes: 1 << 30 }) });
import { createHmac } from "node:crypto";

const b64url = (b) => Buffer.from(b).toString("base64url");

export function issueSessionToken(secret, sub, { ttlS = 900, memoryBytes, timeLimitMs, maxSessions } = {}) {
  if (!secret || secret.length < 32) throw new Error("the token secret must have at least 32 characters");
  const claims = { sub, exp: Math.floor(Date.now() / 1000) + ttlS };
  if (memoryBytes != null) claims.memoryBytes = memoryBytes;
  if (timeLimitMs != null) claims.timeLimitMs = timeLimitMs;
  if (maxSessions != null) claims.maxSessions = maxSessions;
  const body = "tgjs1." + b64url(JSON.stringify(claims));
  return body + "." + b64url(createHmac("sha256", secret).update(body).digest());
}
