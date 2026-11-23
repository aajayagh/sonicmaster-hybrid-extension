"""Push Kaggle T4 GPU kernel to train Pilot V4-0, V4-1, and V4-2 on sonicmaster-v4-pilot-5s."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from kaggle.api.kaggle_api_extended import KaggleApi

ROOT = Path(__file__).resolve().parents[2]


def get_api(line_idx: int = 0) -> tuple[KaggleApi, str]:
    lines = [l.strip() for l in Path("C:/Users/ADMIN/.kaggle/access_token.txt").read_text().splitlines() if l.strip()]
    os.environ["KAGGLE_API_TOKEN"] = lines[line_idx]
    api = KaggleApi()
    api.authenticate()
    return api, str(api.config_values["username"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account-line", type=int, default=0)
    ap.add_argument("--ds-slug", default="sonicmaster-v4-pilot-5s")
    ap.add_argument("--k-slug", default="sonicmaster-v4-pilot-train-run")
    ap.add_argument("--variants", default="v4_0,v4_1,v4_2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stage2-epoch", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--gate-bias-init", type=float, default=-3.5)
    ap.add_argument("--lambda-ms", type=float, default=0.2)
    ap.add_argument("--lambda-g", type=float, default=0.002)
    ap.add_argument("--lambda-kl", type=float, default=0.0)
    args = ap.parse_args()

    api, owner = get_api(args.account_line)
    ds_ref = args.ds_slug if "/" in args.ds_slug else f"{owner}/{args.ds_slug}"
    k_slug = args.k_slug
    k_dir = ROOT / "artifacts" / "10_extension" / "v4" / "kaggle_kernel_pilot_train"
    k_dir.mkdir(parents=True, exist_ok=True)

    # Inline v4_models.py & v4_losses.py into self-contained Kaggle script
    models_src = (ROOT / "scripts" / "extension" / "v4_models.py").read_text(encoding="utf-8")
    losses_src = (ROOT / "scripts" / "extension" / "v4_losses.py").read_text(encoding="utf-8")
    losses_src = losses_src.replace("from __future__ import annotations\n", "")
    losses_src = losses_src.replace("from scripts.extension.v4_models import compress_stft, lr_to_ms, stft_ms", "")

    driver = f'''
import json, random, time, zipfile
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

INPUT_ROOT = Path("/kaggle/input")
DATA_DIR = None
for _ in range(60):
    found = list(INPUT_ROOT.rglob("pilot_meta.json"))
    if found:
        DATA_DIR = found[0].parent
        break
    time.sleep(5)
if DATA_DIR is None:
    raise RuntimeError(f"pilot_meta.json not found under {{list(INPUT_ROOT.rglob('*'))[:20]}}")

if not (DATA_DIR / "clips").is_dir():
    WORK_DATA = Path("/kaggle/working/pilot_data")
    WORK_DATA.mkdir(parents=True, exist_ok=True)
    for zp in DATA_DIR.glob("*.zip"):
        print(f"Extracting {{zp.name}}...", flush=True)
        with zipfile.ZipFile(zp, "r") as zf:
            zf.extractall(WORK_DATA / zp.stem)
    if (DATA_DIR / "pilot_meta.json").is_file():
        (WORK_DATA / "pilot_meta.json").write_text((DATA_DIR / "pilot_meta.json").read_text())
    DATA_DIR = WORK_DATA

meta = json.loads((DATA_DIR / "pilot_meta.json").read_text())
print(f"Loaded meta: train={{len(meta['train_rows'])}} val={{len(meta['val_rows'])}}", flush=True)

CHUNK_SAMPLES = 44100 * 4

class PilotCropDataset(Dataset):
    def __init__(self, rows, data_dir: Path, has_teacher: bool, train: bool):
        self.rows = rows
        self.data_dir = data_dir
        self.has_teacher = has_teacher
        self.train = train

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        sid = r["sample_id"]
        src = r["source_id"]
        z = np.load(self.data_dir / "clips" / f"{{sid}}.npz")
        x_np = z["x"].astype(np.float32)
        s_np = z["s"].astype(np.float32)
        rx = float(z["rms_x"])
        rs = float(z["rms_s"])
        c_np = np.load(self.data_dir / "clean" / f"{{src}}.npy").astype(np.float32)

        n = min(x_np.shape[-1], s_np.shape[-1], c_np.shape[-1])
        if n > CHUNK_SAMPLES:
            st = random.randint(0, n - CHUNK_SAMPLES) if self.train else (n - CHUNK_SAMPLES) // 2
            en = st + CHUNK_SAMPLES
            x_np, s_np, c_np = x_np[:, st:en], s_np[:, st:en], c_np[:, st:en]

        x_t = torch.from_numpy(x_np)
        s_t = torch.from_numpy(s_np)
        c_t = torch.from_numpy(c_np)

        if not self.has_teacher:
            scale = max(rx, 1e-8)
            x_in = x_t / scale
            s_in = torch.zeros_like(x_in)
            c_in = c_t / scale
            raw_stats = torch.tensor([np.log(rx + 1e-8), 0.0, 0.0], dtype=torch.float32)
        else:
            scale = max(rs, 1e-8)
            g_align = float(np.clip(rs / max(rx, 1e-8), 0.1, 16.0))
            x_in = (x_t * g_align) / scale
            s_in = s_t / scale
            c_in = c_t / scale
            gain_norm = float(np.clip(20.0 * np.log10(max(rs, 1e-8) / max(rx, 1e-8)) / 24.0, -2.0, 2.0))
            raw_stats = torch.tensor([np.log(rx + 1e-8), np.log(rs + 1e-8), gain_norm], dtype=torch.float32)

        return x_in, s_in, c_in, raw_stats, r["degradation_types"]


def run_epoch(model, loader, criterion, optimizer, scaler, device, stage):
    is_train = optimizer is not None
    model.train(is_train)
    sums = {{}}
    nb = 0
    g_by_deg = {{}}
    for x_in, s_in, c_in, raw_stats, deg_strs in loader:
        x_in = x_in.to(device, non_blocking=True)
        s_in = s_in.to(device, non_blocking=True) if model.has_teacher else None
        c_in = c_in.to(device, non_blocking=True)
        raw_stats = raw_stats.to(device, non_blocking=True)
        if is_train:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", enabled=True):
                y_hat, diag = model.forward_normalized(x_in, s_in, raw_stats)
            loss, parts = criterion(
                y_hat,
                c_in,
                diag,
                stage=stage,
                lambda_ms={args.lambda_ms},
                lambda_g={args.lambda_g},
                lambda_kl={args.lambda_kl},
            )
        if is_train:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            scaler.step(optimizer)
            scaler.update()
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        for k, v in diag.items():
            sums[k] = sums.get(k, 0.0) + float(v.item())
        sums["clip_rate"] = sums.get("clip_rate", 0.0) + float((y_hat.abs() >= 0.999).float().mean().item())
        nb += 1
    return {{k: v / max(1, nb) for k, v in sums.items()}}


device = torch.device("cuda")
OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
variants = {json.dumps([v.strip() for v in args.variants.split(",") if v.strip()])}

for var in variants:
    torch.manual_seed(20261122)
    random.seed(20261122)
    vdir = OUT / var
    vdir.mkdir(parents=True, exist_ok=True)
    has_t = var in ("v4_1", "v4_2", "v4_3", "v4_4")
    ds_tr = PilotCropDataset(meta["train_rows"], DATA_DIR, has_t, True)
    ds_va = PilotCropDataset(meta["val_rows"], DATA_DIR, has_t, False)
    dl_tr = DataLoader(ds_tr, batch_size={args.batch_size}, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size={args.batch_size}, shuffle=False, num_workers=2, pin_memory=True)

    model = HybridV4Net(var, hop=512, gate_bias_init={args.gate_bias_init}).to(device)
    print(f"=== {{var}} params: {{count_parameters_breakdown(model)}} ===", flush=True)
    criterion = HybridV4Loss().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max={args.epochs}, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    hist = []
    for ep in range(1, {args.epochs} + 1):
        t0 = time.time()
        stage = 2 if ep >= {args.stage2_epoch} else 1
        tr_s = run_epoch(model, dl_tr, criterion, opt, scaler, device, stage)
        va_s = run_epoch(model, dl_va, criterion, None, None, device, stage)
        sched.step()
        dt = time.time() - t0
        hist.append({{"epoch": ep, "stage": stage, "sec": round(dt, 1), "train": tr_s, "val": va_s}})
        g_info = f" gM={{va_s['g_mean_M']:.4f}} gS={{va_s['g_mean_S']:.4f}} dw0={{va_s['dw0_rms']:.4f}} dwk={{va_s['dwk_rms']:.4f}}" if "g_mean_M" in va_s else ""
        print(f"[{{var}} Ep {{ep:02d}}/{args.epochs} S{{stage}}] {{dt:.1f}}s tr={{tr_s['l_stft']:.4f}} va={{va_s['l_stft']:.4f}} wav={{va_s['l_wav']:.4f}} clip={{va_s['clip_rate']:.5f}}{{g_info}}", flush=True)
        if va_s["l_stft"] < best_val:
            best_val = va_s["l_stft"]
            torch.save({{"epoch": ep, "variant": var, "model": model.state_dict(), "val": va_s}}, vdir / "best.pt")
    (vdir / "history.json").write_text(json.dumps(hist, indent=2))
print("ALL_PILOT_DONE", flush=True)
'''
    full_script = models_src + "\n" + losses_src + "\n" + driver
    (k_dir / "train_pilot.py").write_text(full_script, encoding="utf-8")
    meta = {
        "id": f"{owner}/{k_slug}",
        "title": k_slug,
        "code_file": "train_pilot.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": False,
        "dataset_sources": [ds_ref],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (k_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    for attempt in range(5):
        try:
            api.kernels_push(str(k_dir))
            break
        except Exception as e:
            print(f"kernels_push retry {attempt+1}: {e}")
            time.sleep(15)
    print(f"Pushed {owner}/{k_slug}!")


if __name__ == "__main__":
    main()
