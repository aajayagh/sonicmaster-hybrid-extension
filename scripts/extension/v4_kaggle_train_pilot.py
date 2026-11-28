"""Push Kaggle T4 GPU kernel to train Pilot V4 variants (Stages 1-2 or Stage 3 fine-tuning with Dual-Domain GAN)."""
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


def _strip_local_header(src: str, extra_strip: tuple[str, ...] = ()) -> str:
    lines = src.splitlines()
    drop = {"from __future__ import annotations", *extra_strip}
    kept = [ln for ln in lines if ln.strip() not in drop]
    return "\n".join(kept) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--account-line", type=int, default=0)
    ap.add_argument("--ds-slug", default="sonicmaster-v4-pilot-5s")
    ap.add_argument("--k-slug", default="sonicmaster-v4-pilot-train-run")
    ap.add_argument("--variants", default="v4_0,v4_1,v4_2")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--stage2-epoch", type=int, default=13)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--grad-accum", type=int, default=1)
    ap.add_argument("--gate-bias-init", type=float, default=-3.5)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lr-head", type=float, default=5e-4)
    ap.add_argument("--lr-d", type=float, default=2e-4)
    ap.add_argument("--lambda-ms", type=float, default=0.2)
    ap.add_argument("--lambda-g", type=float, default=0.002)
    ap.add_argument("--lambda-kl", type=float, default=0.0)
    ap.add_argument("--lambda-hf", type=float, default=0.0)
    ap.add_argument("--lambda-br", type=float, default=0.0)
    ap.add_argument("--lambda-br-p", type=float, default=0.0)
    ap.add_argument("--br-quad-scale", type=float, default=0.0)
    ap.add_argument("--br-log-weight", type=float, default=0.0)
    ap.add_argument("--lambda-rev", type=float, default=0.0)
    ap.add_argument("--rev-fp-weight", type=float, default=2.0)
    ap.add_argument("--freeze-reverb-head", action="store_true")
    ap.add_argument("--freeze-backbone-and-rev-head", action="store_true")
    ap.add_argument("--w-comp", type=float, default=0.5)
    ap.add_argument("--w-wav", type=float, default=0.5)
    ap.add_argument("--lambda-mod", type=float, default=0.0)
    ap.add_argument("--init-kernel-slug", default="")
    ap.add_argument("--init-ds-slug", default="")
    ap.add_argument("--init-ckpt-glob", default="*best.pt")
    ap.add_argument("--init-allow-new-head", action="store_true")
    ap.add_argument("--use-gan", action="store_true")
    ap.add_argument("--gan-warmup-epochs", type=int, default=2)
    ap.add_argument("--lambda-adv", type=float, default=0.05)
    ap.add_argument("--lambda-fm", type=float, default=0.50)
    ap.add_argument("--dry-run", action="store_true", help="Write train_pilot.py & metadata locally without pushing to Kaggle")
    args = ap.parse_args()

    k_slug = args.k_slug
    k_dir = ROOT / "artifacts" / "10_extension" / "v4" / "kaggle_kernel_pilot_train"
    k_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        owner = "dryrun"
    else:
        api, owner = get_api(args.account_line)
    ds_ref = args.ds_slug if "/" in args.ds_slug else f"{owner}/{args.ds_slug}"

    # Inline v4_models.py, v4_losses.py, and v4_discriminators.py into self-contained Kaggle script
    models_src = _strip_local_header((ROOT / "scripts" / "extension" / "v4_models.py").read_text(encoding="utf-8"))
    losses_src = _strip_local_header(
        (ROOT / "scripts" / "extension" / "v4_losses.py").read_text(encoding="utf-8"),
        ("from scripts.extension.v4_models import compress_stft, lr_to_ms, stft_ms",),
    )
    disc_src = _strip_local_header(
        (ROOT / "scripts" / "extension" / "v4_discriminators.py").read_text(encoding="utf-8"),
        ("from scripts.extension.v4_models import compress_stft, lr_to_ms",),
    )
    script_header = (
        "from __future__ import annotations\n\n"
        "import os\n"
        'os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")\n\n'
    )

    init_slug = args.init_kernel_slug.strip()
    init_ds_slug = args.init_ds_slug.strip()
    use_init_ckpt = bool(init_slug or init_ds_slug)
    use_gan = bool(args.use_gan)
    init_allow_new_head = bool(args.init_allow_new_head)

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

INIT_CKPT_PATH = None
USE_INIT_CKPT = {repr(use_init_ckpt)}
INIT_ALLOW_NEW_HEAD = {repr(init_allow_new_head)}
INIT_CKPT_GLOB = {repr(args.init_ckpt_glob)}
INIT_KERNEL_LEAF = {repr((init_slug or init_ds_slug).split("/")[-1] if (init_slug or init_ds_slug) else "")}
if USE_INIT_CKPT:
    matches = sorted(INPUT_ROOT.rglob(INIT_CKPT_GLOB))
    if not matches:
        for _ in range(30):
            matches = sorted(INPUT_ROOT.rglob(INIT_CKPT_GLOB))
            if matches:
                break
            time.sleep(2)
    if not matches:
        raise RuntimeError(f"Init checkpoint glob {{INIT_CKPT_GLOB!r}} matched nothing under /kaggle/input")
    preferred = [m for m in matches if INIT_KERNEL_LEAF and INIT_KERNEL_LEAF in str(m)]
    INIT_CKPT_PATH = preferred[0] if preferred else matches[0]
    print(f"Found init checkpoint: {{INIT_CKPT_PATH}}", flush=True)

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


REVERB_TAGS = {{"small", "big", "mix", "real"}}


def _np_auroc(y_t: np.ndarray, y_s: np.ndarray) -> float:
    y_b = y_t > 0.5
    n_pos = int(y_b.sum())
    n_neg = int((~y_b).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(y_s, kind="mergesort")
    sorted_s = y_s[order]
    ranks = np.empty_like(sorted_s, dtype=np.float64)
    i, n = 0, len(sorted_s)
    while i < n:
        j = i + 1
        while j < n and sorted_s[j] == sorted_s[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + 1 + j)
        i = j
    return float((ranks[y_b].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    stage,
    disc=None,
    opt_d=None,
    scaler_d=None,
    gan_scale: float = 0.0,
    grad_accum: int = 1,
    use_rev_mask: bool = False,
):
    is_train = optimizer is not None
    model.train(is_train)
    if disc is not None:
        disc.train(is_train)
    sums = {{}}
    nb = 0
    val_rev_y, val_rev_p, val_g_ph, val_g_mag = [], [], [], []
    n_batches = len(loader)
    eff_accum = max(1, int(grad_accum)) if is_train else 1
    for step_idx, (x_in, s_in, c_in, raw_stats, deg_strs) in enumerate(loader):
        x_in = x_in.to(device, non_blocking=True)
        s_in = s_in.to(device, non_blocking=True) if model.has_teacher else None
        c_in = c_in.to(device, non_blocking=True)
        raw_stats = raw_stats.to(device, non_blocking=True)
        rev_vals = [1.0 if (set(s.split("|")) & REVERB_TAGS) else 0.0 for s in deg_strs]
        rev_t = torch.tensor(rev_vals, dtype=torch.float32, device=device)

        if is_train and (step_idx % eff_accum == 0):
            optimizer.zero_grad(set_to_none=True)
            if disc is not None and opt_d is not None:
                opt_d.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_train):
            with torch.amp.autocast(device_type="cuda", enabled=True):
                y_hat, diag = model.forward_normalized(x_in, s_in, raw_stats)

        # 1. Accumulate Discriminator gradients on detached generator outputs, then free D graph before G losses
        d_loss_val = 0.0
        if disc is not None and is_train:
            for p in disc.parameters():
                p.requires_grad_(True)
            with torch.amp.autocast(device_type="cuda", enabled=True):
                r_logits_d, _ = disc(c_in)
                f_logits_d, _ = disc(y_hat.detach())
                l_d = hinge_disc_loss(r_logits_d, f_logits_d)
            d_loss_val = float(l_d.item())
            scaler_d.scale(l_d / float(eff_accum)).backward()
            del r_logits_d, f_logits_d, l_d

        # 2. Generator reconstruction + optional adversarial + feature-matching loss
        with torch.set_grad_enabled(is_train):
            loss, parts = criterion(
                y_hat,
                c_in,
                diag,
                stage=stage,
                lambda_ms={args.lambda_ms},
                lambda_g={args.lambda_g},
                lambda_kl={args.lambda_kl},
                lambda_hf={args.lambda_hf},
                lambda_br={args.lambda_br},
                w_comp={args.w_comp},
                w_wav={args.w_wav},
                lambda_mod={args.lambda_mod},
                rev_target=rev_t if use_rev_mask else None,
                lambda_rev={args.lambda_rev},
                rev_fp_weight={args.rev_fp_weight},
                lambda_br_p={args.lambda_br_p},
                br_quad_scale={args.br_quad_scale},
                br_log_weight={args.br_log_weight},
            )
            r_fmaps_g = None
            f_fmaps_g = None
            f_logits_g = None
            if disc is not None:
                if is_train and gan_scale <= 0.0:
                    with torch.no_grad(), torch.amp.autocast(device_type="cuda", enabled=True):
                        _, r_fmaps_g = disc(c_in)
                        f_logits_g, f_fmaps_g = disc(y_hat.detach())
                        l_adv = hinge_gen_loss(f_logits_g)
                        l_fm = normalized_feature_matching_loss(r_fmaps_g, f_fmaps_g)
                else:
                    if is_train:
                        for p in disc.parameters():
                            p.requires_grad_(False)
                    with torch.amp.autocast(device_type="cuda", enabled=True):
                        with torch.no_grad():
                            r_logits_g, r_fmaps_g = disc(c_in)
                        f_logits_g, f_fmaps_g = disc(y_hat)
                        l_adv = hinge_gen_loss(f_logits_g)
                        l_fm = normalized_feature_matching_loss(r_fmaps_g, f_fmaps_g)
                        if not is_train:
                            d_loss_val = float(hinge_disc_loss(r_logits_g, f_logits_g).item())
                        del r_logits_g
                parts["l_d"] = d_loss_val
                parts["l_adv"] = float(l_adv.item())
                parts["l_fm"] = float(l_fm.item())
                if gan_scale > 0.0:
                    loss = loss + gan_scale * ({args.lambda_adv} * l_adv + {args.lambda_fm} * l_fm)
                parts["total"] = float(loss.item())

        clip_rate_val = float((y_hat.detach().abs() >= 0.999).float().mean().item())
        for k, v in parts.items():
            sums[k] = sums.get(k, 0.0) + v
        for k, v in diag.items():
            if v.ndim == 0:
                sums[k] = sums.get(k, 0.0) + float(v.item())
        if (not is_train) and ("rev_logit" in diag):
            val_rev_y.append(rev_t.detach().cpu().numpy())
            val_rev_p.append(torch.sigmoid(diag["rev_logit"].detach()).cpu().numpy())
            val_g_ph.append(diag["g_ph_item"].detach().cpu().numpy())
            val_g_mag.append(diag["g_mag_item"].detach().cpu().numpy())
        sums["clip_rate"] = sums.get("clip_rate", 0.0) + clip_rate_val
        nb += 1

        if is_train:
            scaler.scale(loss / float(eff_accum)).backward()
            if disc is not None:
                for p in disc.parameters():
                    p.requires_grad_(True)
            step_now = ((step_idx + 1) % eff_accum == 0) or ((step_idx + 1) == n_batches)
            if step_now:
                if disc is not None and opt_d is not None:
                    scaler_d.unscale_(opt_d)
                    torch.nn.utils.clip_grad_norm_(disc.parameters(), 5.0)
                    scaler_d.step(opt_d)
                    scaler_d.update()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
                scaler.step(optimizer)
                scaler.update()

        if disc is not None:
            del r_fmaps_g, f_fmaps_g, f_logits_g
        del y_hat, loss
    out_d = {{k: v / max(1, nb) for k, v in sums.items()}}
    if val_rev_y:
        yt = np.concatenate(val_rev_y)
        yp = np.concatenate(val_rev_p)
        gp = np.concatenate(val_g_ph)
        gm = np.concatenate(val_g_mag)
        pos_m = yt > 0.5
        neg_m = ~pos_m
        out_d["val_rev_auroc"] = _np_auroc(yt, yp)
        out_d["val_rev_tpr"] = float((yp[pos_m] >= 0.5).mean()) if pos_m.any() else 0.0
        out_d["val_nonrev_fpr"] = float((yp[neg_m] >= 0.5).mean()) if neg_m.any() else 0.0
        out_d["g_ph_rev"] = float(gp[pos_m].mean()) if pos_m.any() else 0.0
        out_d["g_ph_nonrev"] = float(gp[neg_m].mean()) if neg_m.any() else 0.0
        out_d["g_mag_rev"] = float(gm[pos_m].mean()) if pos_m.any() else 0.0
        out_d["g_mag_nonrev"] = float(gm[neg_m].mean()) if neg_m.any() else 0.0
    return out_d


device = torch.device("cuda")
OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
variants = {repr([v.strip() for v in args.variants.split(",") if v.strip()])}
USE_GAN = {repr(use_gan)}
GAN_WARMUP_EPOCHS = {int(args.gan_warmup_epochs)}
EPOCHS = {int(args.epochs)}
FREEZE_REVERB_HEAD = {repr(bool(args.freeze_reverb_head))}
FREEZE_UPSTREAM = {repr(bool(args.freeze_backbone_and_rev_head))}

for var in variants:
    torch.manual_seed(20261122)
    random.seed(20261122)
    vdir = OUT / var
    vdir.mkdir(parents=True, exist_ok=True)
    has_t = var in ("v4_1", "v4_2", "v4_3", "v4_4", "v4_5p", "v4_6", "v4_7", "v4_7c")
    ds_tr = PilotCropDataset(meta["train_rows"], DATA_DIR, has_t, True)
    ds_va = PilotCropDataset(meta["val_rows"], DATA_DIR, has_t, False)
    dl_tr = DataLoader(ds_tr, batch_size={args.batch_size}, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size={args.batch_size}, shuffle=False, num_workers=2, pin_memory=True)

    model = HybridV4Net(var, hop=512, gate_bias_init={args.gate_bias_init}).to(device)
    if INIT_CKPT_PATH is not None:
        ck = torch.load(INIT_CKPT_PATH, map_location="cpu", weights_only=False)
        if INIT_ALLOW_NEW_HEAD:
            incompat = model.load_state_dict(ck["model"], strict=False)
            assert len(incompat.unexpected_keys) == 0, f"Unexpected keys: {{incompat.unexpected_keys}}"
            bad_missing = [
                k for k in incompat.missing_keys
                if not (
                    k.startswith("decoder.reverb_head.")
                    or k in ("decoder.gamma_band_logits", "decoder.hf_def_alpha_logit", "decoder.hf_def_tau")
                )
            ]
            assert len(bad_missing) == 0, f"Missing non-head keys: {{bad_missing}}"
            print(f"Loaded init weights strict=False from {{INIT_CKPT_PATH}} (new head missing={{len(incompat.missing_keys)}})", flush=True)
        else:
            model.load_state_dict(ck["model"], strict=True)
            print(f"Loaded init weights strict=True from {{INIT_CKPT_PATH}} (src_var={{ck.get('variant')}}, ep={{ck.get('epoch')}})", flush=True)
    print(f"=== {{var}} params: {{count_parameters_breakdown(model)}} ===", flush=True)

    criterion = HybridV4Loss().to(device)
    use_rev_routing = var in ("v4_7", "v4_7c")
    if use_rev_routing and FREEZE_UPSTREAM:
        for n, p in model.named_parameters():
            if n.startswith("encoder.") or n.startswith("blocks.") or n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits":
                p.requires_grad_(False)
        hf_params = [p for n, p in model.named_parameters() if p.requires_grad and n in ("decoder.hf_def_alpha_logit", "decoder.hf_def_tau")]
        dec_params = [p for n, p in model.named_parameters() if p.requires_grad and n not in ("decoder.hf_def_alpha_logit", "decoder.hf_def_tau")]
        param_groups = [{{"params": dec_params, "lr": {args.lr}}}]
        if hf_params:
            param_groups.append({{"params": hf_params, "lr": {args.lr_head}}})
        opt = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    elif use_rev_routing and FREEZE_REVERB_HEAD:
        for n, p in model.named_parameters():
            if n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits":
                p.requires_grad_(False)
        opt = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr={args.lr},
            weight_decay=1e-4,
        )
    elif use_rev_routing:
        head_params = [p for n, p in model.named_parameters() if p.requires_grad and (n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits")]
        base_params = [p for n, p in model.named_parameters() if p.requires_grad and not (n.startswith("decoder.reverb_head.") or n == "decoder.gamma_band_logits")]
        opt = torch.optim.AdamW(
            [{{"params": base_params, "lr": {args.lr}}}, {{"params": head_params, "lr": {args.lr_head}}}],
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr={args.lr}, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS, eta_min=1e-5)
    scaler = torch.amp.GradScaler("cuda")

    disc = None
    opt_d = None
    sched_d = None
    scaler_d = None
    if USE_GAN:
        disc = HybridV4Discriminator().to(device)
        d_params = sum(p.numel() for p in disc.parameters())
        print(f"=== {{var}} discriminator params: {{d_params}} ===", flush=True)
        opt_d = torch.optim.AdamW(disc.parameters(), lr={args.lr_d}, betas=(0.5, 0.9), weight_decay=1e-4)
        sched_d = torch.optim.lr_scheduler.CosineAnnealingLR(opt_d, T_max=EPOCHS, eta_min=1e-5)
        scaler_d = torch.amp.GradScaler("cuda")

    best_val = float("inf")
    best_qual_val = float("inf")
    has_qual_best = False
    hist = []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        stage = 2 if ep >= {args.stage2_epoch} else 1
        if not USE_GAN:
            gan_scale = 0.0
        elif GAN_WARMUP_EPOCHS <= 0:
            gan_scale = 1.0
        else:
            gan_scale = min(1.0, max(0.0, float(ep - 1) / float(GAN_WARMUP_EPOCHS)))

        tr_s = run_epoch(model, dl_tr, criterion, opt, scaler, device, stage, disc=disc, opt_d=opt_d, scaler_d=scaler_d, gan_scale=gan_scale, grad_accum={int(args.grad_accum)}, use_rev_mask=use_rev_routing)
        va_s = run_epoch(model, dl_va, criterion, None, None, device, stage, disc=disc, gan_scale=gan_scale, grad_accum=1, use_rev_mask=use_rev_routing)
        sched.step()
        if sched_d is not None:
            sched_d.step()
        dt = time.time() - t0
        hist.append({{"epoch": ep, "stage": stage, "gan_scale": gan_scale, "sec": round(dt, 1), "train": tr_s, "val": va_s}})
        g_info = f" gM={{va_s['g_mean_M']:.4f}} gS={{va_s['g_mean_S']:.4f}} dw0={{va_s['dw0_rms']:.4f}} dwk={{va_s['dwk_rms']:.4f}}" if "g_mean_M" in va_s else ""
        ph_info = f" gphM={{va_s['g_ph_mean_M']:.4f}} gphS={{va_s['g_ph_mean_S']:.4f}}" if "g_ph_mean_M" in va_s else ""
        gan_info = f" fm={{va_s.get('l_fm', 0.0):.4f}} adv={{va_s.get('l_adv', 0.0):.4f}} D={{va_s.get('l_d', 0.0):.4f}} gs={{gan_scale:.2f}}" if USE_GAN else ""
        br_info = f" br={{va_s['l_br']:.4f}}" if "l_br" in va_s else ""
        brp_info = f" brP={{va_s['l_br_p']:.4f}}" if "l_br_p" in va_s else ""
        hf_info = f" hfA={{va_s['hf_def_alpha']:.3f}} hfT={{va_s['hf_def_tau']:.3f}} phf={{va_s['p_hf_def_mean']:.3f}}" if "hf_def_alpha" in va_s else ""
        rev_info = (
            f" auroc={{va_s['val_rev_auroc']:.4f}} tpr={{va_s['val_rev_tpr']:.3f}} fpr={{va_s['val_nonrev_fpr']:.3f}}"
            f" gphR={{va_s['g_ph_rev']:.3f}} gphNR={{va_s['g_ph_nonrev']:.3f}} gmR={{va_s['g_mag_rev']:.3f}}"
            if "val_rev_auroc" in va_s else ""
        )
        print(f"[{{var}} Ep {{ep:02d}}/{{EPOCHS}} S{{stage}}] {{dt:.1f}}s tr={{tr_s['l_stft']:.4f}} va={{va_s['l_stft']:.4f}} wav={{va_s['l_wav']:.4f}} clip={{va_s['clip_rate']:.5f}}{{g_info}}{{ph_info}}{{br_info}}{{brp_info}}{{hf_info}}{{rev_info}}{{gan_info}}", flush=True)

        ckpt_obj = {{
            "epoch": ep,
            "variant": var,
            "model": model.state_dict(),
            "val": va_s,
            "inference_config": {{"hop": 512, "gate_bias_init": {args.gate_bias_init}, "chunk_sec": 8.0}},
        }}
        if USE_GAN:
            torch.save({{**ckpt_obj, "discriminator": disc.state_dict()}}, vdir / "last.pt")
            if ep in (6, 9, 12):
                torch.save(ckpt_obj, vdir / f"ep{{ep:02d}}.pt")
            eligible_best = ep >= GAN_WARMUP_EPOCHS + 1
            val_score = va_s["l_stft"] + 0.5 * va_s.get("l_kl", 0.0) + 0.5 * va_s.get("l_mod", 0.0) + 0.25 * va_s.get("l_fm", 0.0)
            if eligible_best and val_score < best_val:
                best_val = val_score
                torch.save(ckpt_obj, vdir / "best.pt")
        elif use_rev_routing:
            torch.save(ckpt_obj, vdir / "last.pt")
            if FREEZE_UPSTREAM or FREEZE_REVERB_HEAD:
                val_score = va_s["l_stft"] + 0.5 * va_s.get("l_kl", 0.0) + 0.5 * va_s.get("l_mod", 0.0) + 0.5 * va_s.get("l_br", 0.0) + 0.25 * va_s.get("l_ms", 0.0)
            else:
                val_score = va_s["l_stft"] + 0.5 * va_s.get("l_kl", 0.0) + 0.5 * va_s.get("l_mod", 0.0) + 0.25 * va_s.get("l_br", 0.0) + 0.25 * va_s.get("l_rev", 0.0)
            if va_s.get("val_nonrev_fpr", 1.0) <= 0.04:
                if val_score < best_qual_val:
                    best_qual_val = val_score
                    has_qual_best = True
                    torch.save(ckpt_obj, vdir / "best.pt")
            elif (not has_qual_best) and (val_score < best_val):
                best_val = val_score
                torch.save(ckpt_obj, vdir / "best.pt")
        else:
            if va_s["l_stft"] < best_val:
                best_val = va_s["l_stft"]
                torch.save(ckpt_obj, vdir / "best.pt")
    if not (vdir / "best.pt").is_file():
        torch.save(ckpt_obj, vdir / "best.pt")
    (vdir / "history.json").write_text(json.dumps(hist, indent=2))
print("ALL_PILOT_DONE", flush=True)
'''
    full_script = script_header + models_src + "\n" + losses_src + "\n" + disc_src + "\n" + driver
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
        "dataset_sources": [ds_ref] + ([init_ds_slug] if init_ds_slug else []),
        "competition_sources": [],
        "kernel_sources": [init_slug] if init_slug else [],
        "model_sources": [],
    }
    (k_dir / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"DRY_RUN wrote {k_dir / 'train_pilot.py'} and {k_dir / 'kernel-metadata.json'}")
        return
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
