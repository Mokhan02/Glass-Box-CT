import torch

from glassbox.sae.batchtopk import BatchTopKSAE, dead_fraction, fit_scale, fvu, train


def _synthetic(n_feat=64, d=32, k_true=3, seed=0):
    g = torch.Generator().manual_seed(seed)
    D = torch.randn(n_feat, d, generator=g)
    D = D / D.norm(dim=1, keepdim=True)

    def batch(b):
        idx = torch.stack([torch.randperm(n_feat, generator=g)[:k_true] for _ in range(b)])
        coef = torch.rand(b, k_true, generator=g) + 0.5
        x = torch.zeros(b, d)
        for j in range(k_true):
            x += coef[:, j:j + 1] * D[idx[:, j]]
        return x

    return D, batch


def test_batchtopk_keeps_exactly_k_per_token_on_average():
    sae = BatchTopKSAE(16, expansion=4, k=5)
    z = sae.encode(torch.randn(50, 16))
    assert int((z > 0).sum()) == 5 * 50


def test_decoder_rows_stay_unit_norm_and_threshold_set():
    _, batch = _synthetic()
    sae = BatchTopKSAE(32, expansion=4, k=3)
    train(sae, (batch(256) for _ in range(30)), log_every=10**9)
    assert torch.allclose(sae.W_dec.norm(dim=1), torch.ones(sae.n), atol=1e-4)
    assert float(sae.threshold) > 0


def test_sae_learns_synthetic_sparse_data():
    torch.manual_seed(0)
    _, batch = _synthetic()
    sae = BatchTopKSAE(32, expansion=8, k=3)
    sample = batch(2048)
    sae.scale.fill_(fit_scale(sample))
    train(sae, (batch(512) * sae.scale for _ in range(1500)), lr=1e-3, log_every=10**9)
    x = batch(1024) * sae.scale
    z = sae.encode(x, use_threshold=True)
    assert fvu(x, sae.decode(z)) < 0.2
    assert dead_fraction(sae, window=500) < 0.9
