"""Step 0 Compute Benchmark: Parameter count, GMACs, Peak VRAM, and 30s Latency (hop 512 vs 1024, chunk 4s/2s vs 8s/4s)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch
from torch.utils.flop_counter import FlopCounterMode

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from scripts.extension.v4_models import HybridV4Net, count_parameters_breakdown  # noqa: E402


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Benchmarking on device: {device}" + (f" ({torch.cuda.get_device_name(0)})" if device.type == "cuda" else ""))

    param_table = {}
    for var in ("v4_0", "v4_1", "v4_2"):
        m = HybridV4Net(var, hop=512)
        c = count_parameters_breakdown(m)
        param_table[var] = c
        print(f"Params {var}: total={c['total']:,} (enc={c['encoder']:,}, backbone={c['backbone']:,}, dec={c['decoder']:,})")

    # Benchmark V4-2 across hop={512, 1024} and chunk={4s/2s, 8s/4s} on a 30s stereo 44.1kHz clip
    x_30 = torch.randn(1, 2, 44100 * 30, device=device) * 0.1
    s_30 = torch.randn(1, 2, 44100 * 30, device=device) * 0.1
    x_4 = torch.randn(1, 2, 44100 * 4, device=device) * 0.1
    s_4 = torch.randn(1, 2, 44100 * 4, device=device) * 0.1
    stats = torch.zeros(1, 3, device=device)

    bench_rows = []
    for hop in (512, 1024):
        m = HybridV4Net("v4_2", hop=hop).to(device)
        m.eval()

        # Measure GMACs on 4s chunk
        with FlopCounterMode(display=False) as fc:
            with torch.no_grad():
                _ = m.forward_normalized(x_4, s_4, stats)
        gmacs_4s = fc.get_total_flops() / 2e9  # 1 MAC ~= 2 FLOPs

        # Measure training peak VRAM (batch=2, 4s chunk, AMP fp16)
        m.train()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        xb = x_4.repeat(2, 1, 1)
        sb = s_4.repeat(2, 1, 1)
        stb = stats.repeat(2, 1)
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            yb, _ = m.forward_normalized(xb, sb, stb)
            loss = yb.abs().mean()
        loss.backward()
        m.zero_grad(set_to_none=True)
        train_vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

        m.eval()
        for chunk_sec in (4.0, 8.0):
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
            # Warmup
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                _ = m.forward_track(x_30, s_30, chunk_sec=chunk_sec, overlap_ratio=0.5)
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                _ = m.forward_track(x_30, s_30, chunk_sec=chunk_sec, overlap_ratio=0.5)
            if device.type == "cuda":
                torch.cuda.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0
            inf_vram_mb = (torch.cuda.max_memory_allocated() / 1e6) if device.type == "cuda" else 0.0

            row = {
                "hop": hop,
                "chunk_sec": chunk_sec,
                "gmacs_4s": round(gmacs_4s, 3),
                "train_b2_4s_vram_mb": round(train_vram_mb, 1),
                "infer_30s_vram_mb": round(inf_vram_mb, 1),
                "infer_30s_latency_ms": round(lat_ms, 1),
            }
            bench_rows.append(row)
            print(f"Bench hop={hop} chunk={chunk_sec}s: {row}")

    out_dir = ROOT / "artifacts" / "10_extension" / "v4"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "step0_benchmark.json").write_text(
        json.dumps({"device": str(device), "params": param_table, "benchmarks": bench_rows}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
