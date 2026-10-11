"""Download the Glass-Box CT demo data, verify its SHA-256, and unpack it to <out>/demo. Used by the Dockerfile; standard library only. Exits non-zero (failing the build) when the download
fails, the checksum does not match, the archive contains anything unsafe (absolute or ".." paths, links, devices) or is not what the app expects.
Usage: python scripts/fetch_demo_data.py --url URL --sha256 HEX [--out /app]   (URL may be file:// for local tests)"""
import argparse
import hashlib
import json
import os
import sys
import tarfile
import tempfile
import urllib.error
import urllib.request

MAX_BYTES = 500 * 1024 * 1024  # a sanity cap: the real archive is about 56 MB


def fail(msg: str) -> None:
    print(f"fetch_demo_data: ERROR: {msg}", file=sys.stderr); raise SystemExit(1)


def download(url: str, dest: str) -> str:
    h, n = hashlib.sha256(), 0
    try:
        with urllib.request.urlopen(url, timeout=120) as r, open(dest, "wb") as f:
            while chunk := r.read(1 << 20):
                n += len(chunk)
                if n > MAX_BYTES:
                    fail(f"download larger than {MAX_BYTES // (1 << 20)} MB: refusing")
                h.update(chunk); f.write(chunk)
    except urllib.error.HTTPError as e:
        fail(f"{url} returned HTTP {e.code}" + (" (has the release been published, and is the repository public?)" if e.code in (403, 404) else ""))
    except (urllib.error.URLError, OSError) as e:
        fail(f"could not download {url}: {e}")
    return h.hexdigest()


def safe_extract(archive: str, out: str) -> None:
    root = os.path.realpath(out)
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for m in members:
            parts = m.name.split("/")
            if m.name.startswith("/") or ".." in parts or not parts[0] == "demo":
                fail(f"unsafe or unexpected path in the archive: {m.name!r}")
            if not (m.isfile() or m.isdir()):
                fail(f"archive member {m.name!r} is a link or special file: refusing")
            if not os.path.realpath(os.path.join(root, m.name)).startswith(root + os.sep):
                fail(f"archive member {m.name!r} would leave the target directory")
        for m in members:
            dst = os.path.join(root, m.name)
            if m.isdir():
                os.makedirs(dst, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                with tar.extractfile(m) as src, open(dst, "wb") as f:
                    while chunk := src.read(1 << 20):
                        f.write(chunk)


def check_layout(out: str) -> int:
    d = os.path.join(out, "demo")
    for need in ("index.json", "ATTRIBUTION.md"):
        if not os.path.exists(os.path.join(d, need)):
            fail(f"the archive has no demo/{need}")
    scans = json.load(open(os.path.join(d, "index.json"))).get("scans", [])
    for s in scans:
        for f in ("ct.nii.gz", "lesions.json", "quality_audit.json"):
            if not os.path.exists(os.path.join(d, s["id"], f)):
                fail(f"scan {s['id']} is missing {f}")
    return len(scans)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--url", required=True); ap.add_argument("--sha256", required=True); ap.add_argument("--out", default="/app"); a = ap.parse_args(argv)
    want = a.sha256.strip().lower()
    if len(want) != 64 or any(c not in "0123456789abcdef" for c in want):
        fail("--sha256 must be 64 hexadecimal characters")
    with tempfile.TemporaryDirectory() as tmp:
        arc = os.path.join(tmp, "demo-data.tar.gz"); got = download(a.url, arc)
        if got != want:
            fail(f"checksum mismatch for {a.url}\n  expected {want}\n  got      {got}\n  The build is stopped: the file is not the one this repository was built against.")
        print(f"fetch_demo_data: checksum ok ({got})"); os.makedirs(a.out, exist_ok=True); safe_extract(arc, a.out)
    n = check_layout(a.out); print(f"fetch_demo_data: unpacked {n} scans to {os.path.join(a.out, 'demo')}"); return 0


if __name__ == "__main__":
    sys.exit(main())
