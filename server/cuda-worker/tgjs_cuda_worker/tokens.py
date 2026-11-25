"""Short-lived session tokens: your backend issues them to signed-in users; the worker verifies them with a shared secret.

  token = "tgjs1." + base64url(json claims) + "." + base64url(HMAC-SHA256(secret, "tgjs1." + base64url(json claims)))

Claims: sub (user id, required), exp (unix seconds, required), and optional per-user caps: memoryBytes, timeLimitMs,
maxSessions. Browsers only ever see these tokens, never cloud provider credentials or the signing secret.
"""
from __future__ import annotations
import base64, hashlib, hmac, json, time

PREFIX = "tgjs1."

def _b64(b: bytes) -> str: return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
def _unb64(s: str) -> bytes: return base64.urlsafe_b64decode(s + "=" * (-len(s) % 4))

def issue(secret: str | bytes, sub: str, ttl_s: int = 900, **caps) -> str:
  key = secret.encode() if isinstance(secret, str) else secret
  claims = {"sub": sub, "exp": int(time.time()) + int(ttl_s), **{k: v for k, v in caps.items() if v is not None}}
  body = PREFIX + _b64(json.dumps(claims, separators=(",", ":")).encode())
  return body + "." + _b64(hmac.new(key, body.encode(), hashlib.sha256).digest())

class InvalidToken(Exception): pass

def verify(secret: str | bytes, token: str, now: float | None = None) -> dict:
  key = secret.encode() if isinstance(secret, str) else secret
  if not token.startswith(PREFIX) or token.count(".") != 2: raise InvalidToken("malformed token")
  body, sig = token.rsplit(".", 1)
  try: given = _unb64(sig)
  except Exception as e: raise InvalidToken("malformed token") from e
  if not hmac.compare_digest(given, hmac.new(key, body.encode(), hashlib.sha256).digest()): raise InvalidToken("bad signature")
  try: claims = json.loads(_unb64(body[len(PREFIX):]))
  except Exception as e: raise InvalidToken("malformed claims") from e
  if not isinstance(claims, dict) or not isinstance(claims.get("sub"), str) or not isinstance(claims.get("exp"), (int, float)): raise InvalidToken("missing claims")
  if claims["exp"] < (time.time() if now is None else now): raise InvalidToken("token expired")
  return claims
