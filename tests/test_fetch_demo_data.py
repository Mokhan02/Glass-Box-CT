import hashlib
import importlib.util
import io
import json
import os
import tarfile

import pytest

spec = importlib.util.spec_from_file_location("fetch_demo_data", os.path.join(os.path.dirname(os.path.dirname(__file__)), "scripts", "fetch_demo_data.py"))
fdd = importlib.util.module_from_spec(spec); spec.loader.exec_module(fdd)


def _tar(path, extra=None, omit=()):
    files = {"demo/index.json": json.dumps({"scans": [{"id": "demo_1"}]}).encode(), "demo/ATTRIBUTION.md": b"cc by 4.0", "demo/demo_1/ct.nii.gz": b"x", "demo/demo_1/lesions.json": b"{}", "demo/demo_1/quality_audit.json": b"{}"}
    with tarfile.open(path, "w:gz") as t:
        for n, b in files.items():
            if n in omit:
                continue
            ti = tarfile.TarInfo(n); ti.size = len(b); t.addfile(ti, io.BytesIO(b))
        for ti, data in (extra or []):
            t.addfile(ti, io.BytesIO(data) if data is not None else None)
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(tmp_path, sha=None, **kw):
    arc = str(tmp_path / "a.tar.gz"); real = _tar(arc, **kw); out = tmp_path / "out"
    return fdd.main(["--url", "file://" + arc, "--sha256", sha or real, "--out", str(out)]), out


def test_a_matching_checksum_unpacks_the_demo_data(tmp_path):
    rc, out = run(tmp_path); assert rc == 0 and (out / "demo" / "index.json").exists() and (out / "demo" / "demo_1" / "ct.nii.gz").read_bytes() == b"x"


def test_a_wrong_checksum_fails_the_build_and_extracts_nothing(tmp_path, capsys):
    with pytest.raises(SystemExit) as e:
        run(tmp_path, sha="0" * 64)
    assert e.value.code == 1 and "checksum mismatch" in capsys.readouterr().err and not (tmp_path / "out" / "demo").exists()


def test_malformed_checksums_and_unreachable_urls_fail(tmp_path):
    for sha in ("abc", "g" * 64):
        with pytest.raises(SystemExit):
            fdd.main(["--url", "file:///nonexistent", "--sha256", sha, "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        fdd.main(["--url", "file:///definitely/not/here.tar.gz", "--sha256", "a" * 64, "--out", str(tmp_path)])


def test_unsafe_archives_are_refused_even_with_a_correct_checksum(tmp_path):
    for name in ("/etc/evil", "demo/../../evil", "other/x"):
        ti = tarfile.TarInfo(name); ti.size = 1
        with pytest.raises(SystemExit):
            run(tmp_path, extra=[(ti, b"x")])
    ln = tarfile.TarInfo("demo/link"); ln.type = tarfile.SYMTYPE; ln.linkname = "/etc/passwd"
    with pytest.raises(SystemExit):
        run(tmp_path, extra=[(ln, None)])
    assert not (tmp_path / "evil").exists()


def test_an_archive_that_is_not_what_the_app_expects_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        run(tmp_path, omit=("demo/ATTRIBUTION.md",))
