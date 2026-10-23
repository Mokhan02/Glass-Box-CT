import numpy as np
import pytest

torch = pytest.importorskip("torch")

from glassbox.seg import w3s2_engines as EN  # noqa: E402
from glassbox.seg import w3s2_stats as S  # noqa: E402


def test_torch_eraser_matches_numpy_formula_and_identity_is_exact():
    rng = np.random.default_rng(0); X = rng.normal(size=(500, 16)); z = (rng.random(500) > 0.5).astype(float); X[:, 2] += 2 * z; e = S.leace(X, z)
    t = torch.from_numpy(rng.normal(size=(1, 16, 4, 5, 6)).astype(np.float32)); p = {"mu": torch.as_tensor(e["mu"], dtype=torch.float32), "d": torch.as_tensor(e["d"], dtype=torch.float32), "a": torch.as_tensor(e["a"], dtype=torch.float32), "c": float(e["c"])}
    got = EN.apply_eraser_torch(t, p)[0].permute(1, 2, 3, 0).reshape(-1, 16).numpy(); want = S.apply_eraser(t[0].permute(1, 2, 3, 0).reshape(-1, 16).numpy().astype(np.float64), e)
    assert np.allclose(got, want, atol=1e-3)
    i = EN.identity_eraser(16); site_p = {k: torch.as_tensor(v, dtype=torch.float32) if k != "c" else v for k, v in i.items()}; assert torch.equal(EN.apply_eraser_torch(t, site_p), t)      # d = 0 -> bit-identical


def test_site_hook_returns_nothing_without_an_eraser_and_edits_the_output_with_one():
    conv = torch.nn.Conv3d(1, 8, 1); site = EN.Site(conv, torch.device("cpu")); x = torch.randn(1, 1, 4, 4, 4)
    base = conv(x); assert torch.equal(base, site.last) and site.eraser is None                       # no replacement
    rng = np.random.default_rng(1); X = rng.normal(size=(300, 8)); e = S.leace(X, (rng.random(300) > 0.5).astype(float)); site.set_eraser(e); out = conv(x)
    assert not torch.allclose(out, base) and torch.equal(site.last, base)                                # the hook saw the unedited tensor and returned an edited one
    site.set_eraser(None); assert torch.equal(conv(x), base)
