"""Training script for Hybrid-v4 (V4-0 NoGuide, V4-1 DirectMap, V4-2 MF7) on Smoke or Pilot splits."""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
GDRIVE = Path(r"G:\My Drive\SonicMaster_Reproducing")
sys.path.insert(0, str(ROOT))

from scripts.extension.branch_r import estimate_lag_samples  # noqa: E402
from scripts.extension.v4_losses import HybridV4Loss  # noqa: E402
from scripts.extension.v4_models import HybridV4Net, compute_track_norm  # noqa: E402


def res(p: str | Path) -> Path:
    dp = ROOT / p
    return dp if dp.is_file() else GDRIVE / p


def load_stereo_np(path: Path, frames: int = 1320960) -> np.ndarray:
    for attempt in range(3):
        try:
            y, _ = sf.read(io.BytesIO(path.read_bytes()), dtype="float32", always_2d=True)
            if y.shape[1] == 1:
                y = np.repeat(y, 2, axis=1)
            return y[:frames].T.copy()  # (2, N)
        except Exception:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError(f"Failed to load {path}")


def align_s_to_x(x_np: np.ndarray, s_np: np.ndarray) -> np.ndarray:
    """Integer lag alignment of s to x without reading clean."""
    lag, _ = estimate_lag_samples(x_np.T, s_np.T, max_lag=512)
    if lag != 0:
        # Positive lag means s is delayed relative to x -> shift s left by lag
        s_np = np.roll(s_np, -lag, axis=-1)
    return s_np


class V4ChunkDataset(Dataset):
    """Random 4s chunk dataset with track-level normalization & optional RAM/disk cache."""

    def __init__(
        self,
        rows: list[dict[str, str]],
        chunk_samples: int = 44100 * 4,
        has_teacher: bool = True,
        train: bool = True,
        cache_dir: Path | None = None,
    ) -> None:
        self.rows = rows
        self.chunk_samples = chunk_samples
        self.has_teacher = has_teacher
        self.train = train
        self.cache_dir = cache_dir
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def __len__(self) -> int:
        return len(self.rows)

    def _load_or_cache(self, r: dict[str, str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sid = r["sample_id"]
        if self.cache_dir is not None:
            cp = self.cache_dir / f"{sid}.npz"
            if cp.is_file():
                z = np.load(cp)
                return z["x"].astype(np.float32), z["s"].astype(np.float32), z["c"].astype(np.float32)

        x = load_stereo_np(res(r["path_x"]))
        s = load_stereo_np(res(r["path_s"]))
        c = load_stereo_np(res(r["path_clean"]))
        n = min(x.shape[-1], s.shape[-1], c.shape[-1])
        x, s, c = x[:, :n], s[:, :n], c[:, :n]
        s = align_s_to_x(x, s)

        if self.cache_dir is not None:
            np.savez(self.cache_dir / f"{sid}.npz", x=x.astype(np.float16), s=s.astype(np.float16), c=c.astype(np.float16))
        return x, s, c

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        r = self.rows[idx]
        x_np, s_np, c_np = self._load_or_cache(r)
        n = x_np.shape[-1]

        x_t = torch.from_numpy(x_np).unsqueeze(0)
        s_t = torch.from_numpy(s_np).unsqueeze(0)
        c_t = torch.from_numpy(c_np).unsqueeze(0)

        # Track-level normalization
        x_in, s_in, scale, raw_stats = compute_track_norm(x_t, s_t if self.has_teacher else None)
        c_in = c_t / scale

        x_in = x_in.squeeze(0)
        s_in = s_in.squeeze(0) if s_in is not None else torch.zeros_like(x_in)
        c_in = c_in.squeeze(0)
        raw_stats = raw_stats.squeeze(0)

        if n > self.chunk_samples:
            if self.train:
                best_start = 0
                best_rms = -1.0
                for _ in range(5):
                    st = random.randint(0, n - self.chunk_samples)
                    rms_c = float(torch.sqrt(torch.mean(c_in[:, st : st + self.chunk_samples] ** 2)).item())
                    if rms_c >= 0.05:
                        best_start = st
                        break
                    if rms_c > best_rms:
                        best_rms = rms_c
                        best_start = st
                start = best_start
            else:
                start = (n - self.chunk_samples) // 2
            end = start + self.chunk_samples
            x_in = x_in[:, start:end]
            s_in = s_in[:, start:end]
            c_in = c_in[:, start:end]

        return x_in, s_in, c_in, raw_stats


def run_epoch(
    model: HybridV4Net,
    loader: DataLoader,
    criterion: HybridV4Loss,
    optimizer: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler | None,
    device: torch.device,
    stage: int,
) -> dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    stats_sum: dict[str, float] = {}
    n_batches = 0

    for x_in, s_in, c_in, raw_stats in loader:
        x_in = x_in.to(device, non_blocking=True)
        s_in = s_in.to(device, non_blocking=True) if model.has_teacher else None
        c_in = c_in.to(device, non_blocking=True)
        raw_stats = raw_stats.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                y_hat, diag = model.forward_normalized(x_in, s_in, raw_stats)
            loss, parts = criterion(y_hat, c_in, diag, stage=stage)

        if is_train:
            if scaler is not None and device.type == "cuda":
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                optimizer.step()

        for k, v in parts.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + v
        for k, v in diag.items():
            stats_sum[k] = stats_sum.get(k, 0.0) + float(v.item())
        stats_sum["clip_rate"] = stats_sum.get("clip_rate", 0.0) + float((y_hat.abs() >= 0.999).float().mean().item())
        n_batches += 1

    return {k: v / max(1, n_batches) for k, v in stats_sum.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["v4_0", "v4_1", "v4_2"])
    ap.add_argument("--split-mode", default="smoke", choices=["smoke", "pilot"])
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--stage2-epoch", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--hop", type=int, default=512)
    ap.add_argument("--gate-bias-init", type=float, default=-3.5)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--cache-dir", default=str(ROOT / "artifacts/10_extension/v4/smoke_cache"))
    args = ap.parse_args()

    torch.manual_seed(20261122)
    random.seed(20261122)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    splits = json.loads((ROOT / "artifacts/10_extension/v4/pilot_splits.json").read_text(encoding="utf-8"))[args.split_mode]
    tr_srcs = set(splits["train_sources"])
    va_srcs = set(splits["val_sources"])

    all_rows = [
        r for r in csv.DictReader(res("artifacts/10_extension/inputs_manifest.csv").open("r", encoding="utf-8"))
        if r["path_x"] and r["path_s"] and r["path_clean"]
    ]
    tr_rows = [r for r in all_rows if r["source_id"] in tr_srcs]
    va_rows = [r for r in all_rows if r["source_id"] in va_srcs]

    cache_p = Path(args.cache_dir) if args.cache_dir else None
    has_teacher = args.variant in ("v4_1", "v4_2")
    ds_tr = V4ChunkDataset(tr_rows, has_teacher=has_teacher, train=True, cache_dir=cache_p)
    ds_va = V4ChunkDataset(va_rows, has_teacher=has_teacher, train=False, cache_dir=cache_p)

    dl_tr = DataLoader(ds_tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    dl_va = DataLoader(ds_va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    model = HybridV4Net(args.variant, hop=args.hop, gate_bias_init=args.gate_bias_init).to(device)
    criterion = HybridV4Loss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda") if device.type == "cuda" else None

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    history = []

    print(f"Training {args.variant} ({args.split_mode}: train={len(tr_rows)}, val={len(va_rows)}) on {device}", flush=True)
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        stage = 2 if ep >= args.stage2_epoch else 1
        tr_stats = run_epoch(model, dl_tr, criterion, optimizer, scaler, device, stage=stage)
        va_stats = run_epoch(model, dl_va, criterion, None, None, device, stage=stage)
        scheduler.step()

        dt = time.time() - t0
        rec = {"epoch": ep, "stage": stage, "sec": round(dt, 1), "train": tr_stats, "val": va_stats}
        history.append(rec)

        g_str = f" gM={va_stats['g_mean_M']:.4f} dw0={va_stats['dw0_rms']:.4f}" if "g_mean_M" in va_stats else ""
        print(
            f"[Ep {ep:02d}/{args.epochs} S{stage}] {dt:.1f}s | "
            f"tr_tot={tr_stats['total']:.4f} tr_stft={tr_stats['l_stft']:.4f} | "
            f"va_tot={va_stats['total']:.4f} va_stft={va_stats['l_stft']:.4f}{g_str}",
            flush=True,
        )
        if va_stats["l_stft"] < best_val:
            best_val = va_stats["l_stft"]
            torch.save({"epoch": ep, "variant": args.variant, "model": model.state_dict(), "val": va_stats}, out_dir / "best.pt")

    torch.save({"epoch": args.epochs, "variant": args.variant, "model": model.state_dict()}, out_dir / "last.pt")
    (out_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"Done! Best val l_stft={best_val:.4f} saved to {out_dir}")


if __name__ == "__main__":
    main()
