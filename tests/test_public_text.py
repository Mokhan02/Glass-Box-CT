"""Public text rule: a named third-party company must never appear in anything tracked in the repository (code, docs, app text, screenshots' alt text, scripts)."""
import subprocess

FORBIDDEN = ["nu" + "cleo"]  # built from fragments so this file does not contain the name itself


def test_no_third_party_company_names_in_tracked_files():
    files = subprocess.run(["git", "ls-files", "-z"], capture_output=True, check=True).stdout.decode().split("\0")
    hits = []
    for f in filter(None, files):
        try:
            text = open(f, encoding="utf-8").read().lower()
        except (UnicodeDecodeError, OSError):
            continue  # binary files (images, NIfTI)
        hits += [f for name in FORBIDDEN if name in text]
    assert not hits, f"forbidden name found in: {hits}"
