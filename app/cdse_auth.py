"""
CDSE (Copernicus Data Space Ecosystem) authentication helper.

Reads your Copernicus credentials from a local file that is NEVER committed or
shipped, exchanges them for a short-lived OAuth access token against the CDSE
Keycloak identity server, and transparently refreshes it.

Credentials file format (default: $CDSE_ENV_FILE, else ./copernicus.env):
    Username: your_email
    Password: your_password
(Standard KEY=VALUE lines with CDSE_USER / CDSE_PASS are also accepted.)

SECURITY
  //* The password is read at runtime and only sent to the official CDSE token
    endpoint over HTTPS. It is never logged, printed, or written anywhere.
  //* Keep the credentials file OUTSIDE any folder you distribute, and git-ignore it.
  //* Prefer S3 keys or a Sentinel Hub OAuth client over your login password when you
    can — they are revocable and scoped.

Usage:
    from cdse_auth import CDSEAuth
    auth = CDSEAuth()                      # reads copernicus.env
    headers = auth.auth_header()           # {"Authorization": "Bearer ..."}
    # e.g. requests.get(odata_url, headers=headers)
"""
from __future__ import annotations

import os
import time
import urllib.parse
import urllib.request

TOKEN_URL = ("https://identity.dataspace.copernicus.eu/auth/realms/CDSE/"
             "protocol/openid-connect/token")
CLIENT_ID = "cdse-public"


def _find_env_file(path: str | None) -> str:
    if path and os.path.exists(path):
        return path
    env = os.environ.get("CDSE_ENV_FILE")
    if env and os.path.exists(env):
        return env
    for cand in ("copernicus.env",
                 os.path.join(os.path.dirname(os.path.dirname(__file__)), "copernicus.env"),
                 os.path.expanduser("~/copernicus.env")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(
        "No credentials file found. Set CDSE_ENV_FILE or create copernicus.env "
        "with 'Username:' and 'Password:' lines (kept outside any release folder).")


def _parse_creds(path: str) -> tuple[str, str]:
    """Parse 'Username:/Password:' or 'CDSE_USER=/CDSE_PASS=' without echoing values."""
    user = pw = None
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for sep in (":", "="):
                if sep in line:
                    k, v = line.split(sep, 1)
                    k, v = k.strip().lower(), v.strip().strip('"').strip("'")
                    if k in ("username", "cdse_user", "user", "email"):
                        user = v
                    elif k in ("password", "cdse_pass", "pass"):
                        pw = v
                    break
    if not user or not pw:
        raise ValueError(f"{os.path.basename(path)} must define a username and password.")
    return user, pw


def _post_form(data: dict) -> dict:
    import json
    from urllib.error import HTTPError
    body = urllib.parse.urlencode(data).encode()
    req = urllib.request.Request(TOKEN_URL, data=body,
                                 headers={"Content-Type": "application/x-www-form-urlencoded"})
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())
    except HTTPError as e:
        # Surface CDSE's reason (error / error_description) — never contains the password.
        detail = ""
        try:
            j = json.loads(e.read().decode())
            detail = f" — {j.get('error')}: {j.get('error_description')}"
        except Exception:
            pass
        raise RuntimeError(f"HTTP {e.code}{detail}") from None


class CDSEAuth:
    """Lazily fetches and refreshes a CDSE access token. Never stores creds on self."""

    def __init__(self, env_file: str | None = None, leeway: int = 30):
        self._env_file = _find_env_file(env_file)
        self._leeway = leeway
        self._token: str | None = None
        self._refresh: str | None = None
        self._expiry: float = 0.0

    def _login(self) -> None:
        user, pw = _parse_creds(self._env_file)         # read only when needed
        tok = _post_form({"client_id": CLIENT_ID, "grant_type": "password",
                          "username": user, "password": pw})
        del user, pw
        self._apply(tok)

    def _do_refresh(self) -> None:
        tok = _post_form({"client_id": CLIENT_ID, "grant_type": "refresh_token",
                          "refresh_token": self._refresh})
        self._apply(tok)

    def _apply(self, tok: dict) -> None:
        self._token = tok["access_token"]
        self._refresh = tok.get("refresh_token")
        self._expiry = time.time() + int(tok.get("expires_in", 600))

    def token(self) -> str:
        if self._token and time.time() < self._expiry - self._leeway:
            return self._token
        try:
            if self._refresh:
                self._do_refresh()
            else:
                self._login()
        except Exception:
            self._login()                                # refresh expired -> full login
        return self._token

    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.token()}"}


def _read_kv(path: str) -> dict:
    kv = {}
    with open(path, "r", encoding="utf-8-sig") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            for sep in (":", "="):
                if sep in line:
                    k, v = line.split(sep, 1)
                    kv[k.strip().lower()] = v.strip().strip('"').strip("'")
                    break
    return kv


class CDSEClientAuth:
    """Sentinel Hub OAuth client-credentials token (client id + secret). Never logs secrets."""

    def __init__(self, env_file: str | None = None, leeway: int = 30):
        # A file is optional here: on a server the id/secret come from env vars
        # (CDSE_CLIENT_ID / CDSE_CLIENT_SECRET), e.g. Hugging Face Space Secrets.
        try:
            self._env_file = _find_env_file(env_file)
        except FileNotFoundError:
            self._env_file = None
        self._leeway = leeway
        self._token: str | None = None
        self._expiry: float = 0.0

    def _creds(self):
        kv = _read_kv(self._env_file) if self._env_file else {}
        cid = os.environ.get("CDSE_CLIENT_ID") or next(
            (kv[k] for k in ("client id", "client_id", "clientid") if k in kv), None)
        sec = os.environ.get("CDSE_CLIENT_SECRET") or next(
            (kv[k] for k in ("client secret", "client_secret", "clientsecret") if k in kv), None)
        if not cid or not sec:
            raise RuntimeError("OAuth 'client id' / 'client secret' not found in the credentials file.")
        return cid, sec

    def token(self) -> str:
        if self._token and time.time() < self._expiry - self._leeway:
            return self._token
        cid, sec = self._creds()
        tok = _post_form({"grant_type": "client_credentials",
                          "client_id": cid, "client_secret": sec})
        del cid, sec
        self._token = tok["access_token"]
        self._expiry = time.time() + int(tok.get("expires_in", 600))
        return self._token

    def auth_header(self) -> dict:
        return {"Authorization": f"Bearer {self.token()}"}


if __name__ == "__main__":
    # Safe self-test: proves auth works WITHOUT printing the token or credentials.
    try:
        a = CDSEAuth()
        t = a.token()
        secs = int(a._expiry - time.time())
        print(f"OK: got a CDSE access token ({len(t)} chars), valid ~{secs}s. "
              f"Creds file: {a._env_file}")
    except Exception as e:
        print(f"FAILED: {type(e).__name__}: {e}")
