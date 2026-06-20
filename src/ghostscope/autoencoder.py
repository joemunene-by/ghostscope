"""PyTorch autoencoder anomaly detector.

Torch is imported lazily inside this module so that the iforest detector
works even when torch is not installed. A high reconstruction error means
the record does not resemble the learned baseline of normal traffic, and
per-feature reconstruction error provides natural explainability.
"""

from __future__ import annotations

import numpy as np


def _require_torch():
    try:
        import torch  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "The autoencoder model requires PyTorch. Install it with "
            "'pip install torch' (CPU wheel) or use --model iforest."
        ) from exc
    import torch

    return torch


def _build_net(torch, n_features: int, hidden: int, latent: int):
    nn = torch.nn
    return nn.Sequential(
        nn.Linear(n_features, hidden),
        nn.ReLU(),
        nn.Linear(hidden, latent),
        nn.ReLU(),
        nn.Linear(latent, hidden),
        nn.ReLU(),
        nn.Linear(hidden, n_features),
    )


class AutoencoderDetector:
    """Reconstruction-error anomaly detector backed by a small autoencoder."""

    def __init__(
        self,
        hidden: int = 16,
        latent: int = 4,
        epochs: int = 40,
        lr: float = 1e-2,
        batch_size: int = 64,
        seed: int = 1337,
    ) -> None:
        self.hidden = hidden
        self.latent = latent
        self.epochs = epochs
        self.lr = lr
        self.batch_size = batch_size
        self.seed = seed
        self.n_features: int | None = None
        self._net = None
        self._state_dict = None

    def fit(self, x: np.ndarray) -> AutoencoderDetector:
        torch = _require_torch()
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)

        self.n_features = x.shape[1]
        net = _build_net(torch, self.n_features, self.hidden, self.latent)
        net.train()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()

        tensor = torch.tensor(x, dtype=torch.float32)
        n = tensor.shape[0]
        generator = torch.Generator().manual_seed(self.seed)
        for _ in range(self.epochs):
            perm = torch.randperm(n, generator=generator)
            for start in range(0, n, self.batch_size):
                idx = perm[start : start + self.batch_size]
                batch = tensor[idx]
                opt.zero_grad()
                out = net(batch)
                loss = loss_fn(out, batch)
                loss.backward()
                opt.step()

        net.eval()
        self._net = net
        self._state_dict = net.state_dict()
        return self

    def _ensure_net(self):
        torch = _require_torch()
        if self._net is None:
            if self._state_dict is None or self.n_features is None:
                raise RuntimeError("AutoencoderDetector is not trained.")
            net = _build_net(torch, self.n_features, self.hidden, self.latent)
            net.load_state_dict(self._state_dict)
            net.eval()
            self._net = net
        return torch, self._net

    def reconstruction_error(self, x: np.ndarray) -> np.ndarray:
        """Per-record mean squared reconstruction error."""
        torch, net = self._ensure_net()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            out = net(tensor)
            err = ((out - tensor) ** 2).mean(dim=1)
        return err.numpy()

    def per_feature_error(self, x: np.ndarray) -> np.ndarray:
        """Per-record, per-feature squared reconstruction error matrix."""
        torch, net = self._ensure_net()
        with torch.no_grad():
            tensor = torch.tensor(x, dtype=torch.float32)
            out = net(tensor)
            err = (out - tensor) ** 2
        return err.numpy()

    def score(self, x: np.ndarray) -> np.ndarray:
        """Anomaly score: higher means more anomalous (reconstruction error)."""
        return self.reconstruction_error(x)

    def to_payload(self) -> dict:
        """Serialize hyperparameters and weights to a plain dict."""
        torch = _require_torch()
        self._ensure_net()
        buffer = {}
        for key, value in self._state_dict.items():
            buffer[key] = value.cpu().numpy()
        return {
            "hidden": self.hidden,
            "latent": self.latent,
            "epochs": self.epochs,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "n_features": self.n_features,
            "state_dict": buffer,
            "_torch_version": torch.__version__,
        }

    @classmethod
    def from_payload(cls, payload: dict) -> AutoencoderDetector:
        torch = _require_torch()
        det = cls(
            hidden=payload["hidden"],
            latent=payload["latent"],
            epochs=payload["epochs"],
            lr=payload["lr"],
            batch_size=payload["batch_size"],
            seed=payload["seed"],
        )
        det.n_features = payload["n_features"]
        state = {k: torch.tensor(v) for k, v in payload["state_dict"].items()}
        det._state_dict = state
        det._ensure_net()
        return det
