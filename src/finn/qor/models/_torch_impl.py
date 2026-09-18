"""PyTorch implementation details of :class:`finn.qor.models.transformer.TabTransformerRegressor`.

Imported lazily, so that the rest of :mod:`finn.qor` stays importable without torch.
"""

import copy
import numpy as np
import torch
from torch import nn


class FTTransformer(nn.Module):
    """Feature-tokenizer transformer for tabular regression (Gorishniy et al., 2021): every
    input column becomes one token via its own linear embedding, a CLS token is prepended,
    the sequence goes through a pre-norm transformer encoder and the CLS (or mean) token is
    mapped to the scalar output."""

    def __init__(
        self,
        n_features: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffn_mult: int,
        dropout: float,
        pooling: str,
    ):
        super().__init__()
        self.pooling = pooling
        self.weight = nn.Parameter(torch.empty(n_features, d_model))
        self.bias = nn.Parameter(torch.empty(n_features, d_model))
        self.cls = nn.Parameter(torch.empty(1, 1, d_model))
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        nn.init.kaiming_uniform_(self.bias, a=5**0.5)
        nn.init.normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model,
            n_heads,
            dim_feedforward=ffn_mult * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.unsqueeze(-1) * self.weight + self.bias  # (batch, features, d_model)
        tokens = torch.cat([self.cls.expand(x.shape[0], -1, -1), tokens], dim=1)
        encoded = self.encoder(tokens)
        pooled = encoded[:, 0] if self.pooling == "cls" else encoded[:, 1:].mean(dim=1)
        return self.head(self.norm(pooled)).squeeze(-1)


def build_module(n_features: int, hp: dict) -> FTTransformer:
    return FTTransformer(
        n_features,
        hp["d_model"],
        hp["n_layers"],
        hp["n_heads"],
        hp["ffn_mult"],
        hp["dropout"],
        hp["pooling"],
    )


def _loss_fn(name: str):
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    if name == "huber":
        return nn.HuberLoss()
    raise ValueError(f"unknown loss {name!r}")


def train(
    X: np.ndarray,
    y: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    hp: dict,
    seed: int,
    threads: int,
    verbose: int = 0,
) -> tuple[dict, dict]:
    """Train on (X, y) with early stopping on (X_val, y_val); returns the best state_dict
    (as numpy arrays) and training info."""
    torch.manual_seed(seed)
    torch.set_num_threads(max(1, threads))
    device = torch.device("cpu")
    model = build_module(X.shape[1], hp).to(device)
    loss_fn = _loss_fn(hp["loss"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=hp["lr_factor"], patience=hp["lr_patience"]
    )
    Xt = torch.as_tensor(X, dtype=torch.float32, device=device)
    yt = torch.as_tensor(y, dtype=torch.float32, device=device)
    Xv = torch.as_tensor(X_val, dtype=torch.float32, device=device)
    yv = torch.as_tensor(y_val, dtype=torch.float32, device=device)
    generator = torch.Generator().manual_seed(seed)

    best_loss, best_state, best_epoch, bad_epochs = float("inf"), None, 0, 0
    n = len(Xt)
    batch_size = max(1, min(hp["batch_size"], n))
    for epoch in range(hp["max_epochs"]):
        model.train()
        perm = torch.randperm(n, generator=generator)
        for start in range(0, n, batch_size):
            idx = perm[start : start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(Xt[idx]), yt[idx])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(Xv), yv))
        scheduler.step(val_loss)
        if verbose and epoch % 10 == 0:
            print(f"epoch {epoch}: val loss {val_loss:.5f}")
        if val_loss < best_loss - 1e-7:
            best_loss, best_epoch, bad_epochs = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= hp["patience"]:
                break
    if best_state is None:
        best_state = model.state_dict()
    state = {k: v.detach().cpu().numpy() for k, v in best_state.items()}
    info = {
        "torch_version": torch.__version__,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "epochs_trained": epoch + 1,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
    }
    return state, info


def load_module(n_features: int, hp: dict, state: dict) -> FTTransformer:
    model = build_module(n_features, hp)
    model.load_state_dict({k: torch.as_tensor(v) for k, v in state.items()})
    model.eval()
    return model


def predict(model: FTTransformer, X: np.ndarray, threads: int) -> np.ndarray:
    torch.set_num_threads(max(1, threads))
    with torch.no_grad():
        out = model(torch.as_tensor(X, dtype=torch.float32))
    return out.cpu().numpy().astype(float)
