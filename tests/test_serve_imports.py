"""The slim serve image (requirements-serve.txt) has no scipy, sklearn, torch, nibabel, matplotlib or pandas.
Importing and building the app with those blocked catches a top-level import that would crash the container at start-up."""
import pathlib
import subprocess
import sys
import textwrap

ROOT = pathlib.Path(__file__).resolve().parents[1]

CODE = textwrap.dedent("""
    import sys
    sys.path.insert(0, ROOT_PLACEHOLDER)
    BLOCKED = {"scipy", "sklearn", "torch", "monai", "nibabel", "matplotlib", "pandas", "skimage", "SimpleITK"}
    class Block:
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in BLOCKED:
                raise ModuleNotFoundError("No module named %r (blocked: not in the serve image)" % name)
    sys.meta_path.insert(0, Block())
    import glassbox.app.main
    from glassbox.app.main import build_app
    import glassbox.trust.audit, glassbox.trust.verdict, glassbox.trust.longitudinal
    print("ok")
""").replace("ROOT_PLACEHOLDER", repr(str(ROOT)))


def test_server_imports_without_heavy_dependencies():
    r = subprocess.run([sys.executable, "-c", CODE], capture_output=True, text=True, cwd=str(ROOT))
    assert r.returncode == 0 and "ok" in r.stdout, r.stderr[-1500:]
