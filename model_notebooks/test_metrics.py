"""
Verify Metrics numerical correctness by comparing outputs against
independently computed reference values (numpy/scipy).
"""

import torch
import numpy as np
import unittest
import sys
import os
from scipy.linalg import sqrtm

sys.path.insert(0, os.path.dirname(__file__))
from datasets import Metrics


def reference_frechet_distance(emb_p, emb_t, eps=1e-6):
    """
    Reference FD computed with numpy/scipy, independent of torchaudio.
    FD = ||mu_p - mu_t||^2 + Tr(sigma_p + sigma_t - 2 * sqrtm(sigma_p @ sigma_t))
    """
    mu_p = np.mean(emb_p, axis=0)
    mu_t = np.mean(emb_t, axis=0)
    sigma_p = np.cov(emb_p, rowvar=False) + eps * np.eye(emb_p.shape[1])
    sigma_t = np.cov(emb_t, rowvar=False) + eps * np.eye(emb_t.shape[1])

    diff = mu_p - mu_t
    covmean, _ = sqrtm(sigma_p @ sigma_t, disp=False)
    # sqrtm can return complex if numerically unstable, take real part
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fd = diff @ diff + np.trace(sigma_p + sigma_t - 2.0 * covmean)
    return float(fd)


class TestEmbedFrechetDistanceValues(unittest.TestCase):
    """Compare embed_avg_frechet_distance output vs scipy reference."""

    def setUp(self):
        self.metrics = Metrics(model_metrics=False)

    def test_identical_distributions(self):
        """FD(X, X) should be 0."""
        torch.manual_seed(42)
        emb = torch.randn(2, 50, 32)  # (batch, seq_len, embed_dim)
        result = self.metrics.embed_avg_frechet_distance(emb, emb)
        self.assertAlmostEqual(result, 0.0, places=3,
                               msg=f"FD(X,X) should be 0, got {result}")

    def test_shifted_mean(self):
        """Shift mean by known amount, verify FD matches scipy reference."""
        torch.manual_seed(42)
        emb_p = torch.randn(1, 100, 16)  # (1, 100, 16)
        emb_t = emb_p.clone() + 2.0  # shift mean by 2 in every dim

        result = self.metrics.embed_avg_frechet_distance(emb_p, emb_t)

        # scipy reference
        flat_p = emb_p.reshape(-1, 16).numpy()
        flat_t = emb_t.reshape(-1, 16).numpy()
        expected = reference_frechet_distance(flat_p, flat_t)

        self.assertAlmostEqual(result, expected, places=2,
                               msg=f"FD mismatch: got {result}, expected {expected}")

    def test_different_variance(self):
        """Different variance distributions, verify FD matches scipy."""
        torch.manual_seed(123)
        emb_p = torch.randn(1, 200, 8)
        emb_t = torch.randn(1, 200, 8) * 3.0  # wider spread

        result = self.metrics.embed_avg_frechet_distance(emb_p, emb_t)

        flat_p = emb_p.reshape(-1, 8).numpy()
        flat_t = emb_t.reshape(-1, 8).numpy()
        expected = reference_frechet_distance(flat_p, flat_t)

        self.assertAlmostEqual(result, expected, places=1,
                               msg=f"FD mismatch: got {result}, expected {expected}")

    def test_known_gaussians(self):
        """Two unit Gaussians with known mean shift: FD = ||mu_diff||^2."""
        # If both have identity covariance, FD = ||mu_p - mu_t||^2
        torch.manual_seed(0)
        d = 4
        n = 5000  # large enough to approximate identity cov

        emb_p = torch.randn(1, n, d)
        # shift mean by [1,0,0,...,0] => expected FD ≈ 1
        shift = torch.zeros(d)
        shift[0] = 1.0
        emb_t = torch.randn(1, n, d) + shift

        result = self.metrics.embed_avg_frechet_distance(emb_p, emb_t)

        flat_p = emb_p.reshape(-1, d).numpy()
        flat_t = emb_t.reshape(-1, d).numpy()
        expected = reference_frechet_distance(flat_p, flat_t)

        # result should be close to 1.0 (mean shift squared) with matching covariance
        self.assertAlmostEqual(result, expected, places=1,
                               msg=f"FD mismatch: got {result}, expected {expected}")
        # sanity: should be approximately 1.0
        self.assertAlmostEqual(result, 1.0, delta=0.5,
                               msg=f"FD should be near 1.0 for unit shift, got {result}")


class TestCosineSimValues(unittest.TestCase):
    """Compare wav_avg_cos_sim output vs manual computation."""

    def setUp(self):
        self.metrics = Metrics(model_metrics=False)

    def test_identical(self):
        """cos_sim(x, x) = 1"""
        x = torch.tensor([[[1.0, 2.0, 3.0]], [[4.0, 5.0, 6.0]]])  # (2, 1, 3)
        result = self.metrics.wav_avg_cos_sim(x, x)
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_orthogonal(self):
        """cos_sim of orthogonal vectors = 0"""
        x = torch.tensor([[[1.0, 0.0, 0.0]]])  # (1, 1, 3)
        y = torch.tensor([[[0.0, 1.0, 0.0]]])  # (1, 1, 3)
        result = self.metrics.wav_avg_cos_sim(x, y)
        self.assertAlmostEqual(result, 0.0, places=5,
                               msg=f"Orthogonal cos_sim should be 0, got {result}")

    def test_opposite(self):
        """cos_sim(x, -x) = -1"""
        x = torch.tensor([[[3.0, 4.0]]])  # (1, 1, 2)
        result = self.metrics.wav_avg_cos_sim(x, -x)
        self.assertAlmostEqual(result, -1.0, places=5)

    def test_known_angle(self):
        """cos_sim of [1,0] and [1,1] = 1/sqrt(2) ≈ 0.7071"""
        x = torch.tensor([[[1.0, 0.0]]])  # (1, 1, 2)
        y = torch.tensor([[[1.0, 1.0]]])  # (1, 1, 2)
        result = self.metrics.wav_avg_cos_sim(x, y)
        expected = 1.0 / np.sqrt(2)
        self.assertAlmostEqual(result, expected, places=4,
                               msg=f"cos_sim([1,0],[1,1]) should be {expected:.4f}, got {result}")

    def test_batch_average(self):
        """Verify batch averaging is correct."""
        # batch of 2: first pair identical (sim=1), second pair orthogonal (sim=0)
        x = torch.tensor([[[1.0, 0.0]], [[1.0, 0.0]]])  # (2, 1, 2)
        y = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])  # (2, 1, 2)

        result = self.metrics.wav_avg_cos_sim(x, y)

        # manual: sim[0]=1.0, sim[1]=0.0, mean=0.5
        expected = 0.5
        self.assertAlmostEqual(result, expected, places=5,
                               msg=f"Batch avg cos_sim should be {expected}, got {result}")


class TestFDMValues(unittest.TestCase):
    """Verify FDM by manually computing MFCC + FD and comparing."""

    def setUp(self):
        self.metrics = Metrics(model_metrics=False)
        self.device = self.metrics.device

    def test_identical_audio(self):
        """FDM(x, x) = 0"""
        torch.manual_seed(42)
        audio = torch.randn(2, 1, 44100, device=self.device)  # (batch, 1, audio_len)
        result = self.metrics.FDM(audio, audio)
        self.assertAlmostEqual(result, 0.0, places=2,
                               msg=f"FDM(x,x) should be 0, got {result}")

    def test_fdm_matches_manual_pipeline(self):
        """Compute MFCC externally, then FD via scipy, compare to FDM."""
        torch.manual_seed(42)
        audio_a = torch.randn(2, 1, 44100, device=self.device)  # (batch, 1, audio_len)
        audio_b = torch.randn(2, 1, 44100, device=self.device) * 2.0

        # FDM result
        result = self.metrics.FDM(audio_a, audio_b)

        # Manual pipeline: same MFCC transform, then scipy FD
        # FDM internally does squeeze(1), so manual pipeline uses squeezed audio
        mfcc_a = self.metrics.mfcc_transform(audio_a.squeeze(1)).squeeze(1).permute(0, 2, 1)
        mfcc_b = self.metrics.mfcc_transform(audio_b.squeeze(1)).squeeze(1).permute(0, 2, 1)

        flat_a = mfcc_a.reshape(-1, mfcc_a.shape[-1]).cpu().double().numpy()
        flat_b = mfcc_b.reshape(-1, mfcc_b.shape[-1]).cpu().double().numpy()
        expected = reference_frechet_distance(flat_a, flat_b)

        self.assertAlmostEqual(result, expected, places=1,
                               msg=f"FDM mismatch: got {result}, expected {expected}")


@unittest.skipUnless(torch.cuda.is_available(), "Requires CUDA")
class TestNoveltyScoreValues(unittest.TestCase):
    """Verify novelty_score against manual computation."""

    def setUp(self):
        self.metrics = Metrics(model_metrics=False)

    def test_identical_inputs(self):
        """novelty_score(x, x) = 1 (perfect correlation of novelty curves)."""
        torch.manual_seed(42)
        emb = torch.randn(1, 64, 32, device="cuda")
        result = self.metrics.novelty_score(emb, emb).item()
        self.assertAlmostEqual(result, 1.0, places=3,
                               msg=f"NS(x,x) should be 1.0, got {result}")

    def test_manual_novelty_curve(self):
        """Compute novelty curve step-by-step and compare."""
        torch.manual_seed(42)
        emb = torch.randn(1, 32, 16, device="cuda")

        k = 16
        kernel = torch.ones(k, k)
        kernel[:k//2, :k//2] = -1
        kernel[k//2:, k//2:] = -1
        kernel = kernel.unsqueeze(0).unsqueeze(0).to("cuda")

        # manual SSM
        ssm = torch.matmul(emb, emb.transpose(-1, -2))  # (1, 32, 32)

        # manual conv2d on SSM
        import torch.nn.functional as F
        ssm_4d = ssm.unsqueeze(1)
        conv_out = F.conv2d(ssm_4d, kernel, padding=k//2).squeeze(1)
        nov = torch.diagonal(conv_out, dim1=1, dim2=2)
        nov = nov - nov.mean(dim=-1, keepdim=True)
        nov = F.normalize(nov, dim=-1)

        # correlation with itself = 1
        corr = (nov * nov).sum(dim=-1, keepdim=True).mean().item()

        result = self.metrics.novelty_score(emb, emb).item()
        self.assertAlmostEqual(result, corr, places=4,
                               msg=f"NS mismatch: got {result}, manual {corr}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
