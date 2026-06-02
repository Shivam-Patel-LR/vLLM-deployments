"""Concurrent throughput sweep for DeepSeek-OCR vLLM container.

Renders PDF pages once, then at each target concurrency level N keeps exactly N
requests in-flight for a fixed duration and reports pages/sec. Outputs are
discarded (only completion and token counts are kept).
"""

import argparse
import base64
import io
import random
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

import fitz
import httpx
from PIL import Image


def render_pages(pdf_path: Path, dpi: int) -> list[str]:
    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
    b64_pages = []
    for page in doc:
        pix = page.get_pixmap(matrix=mat, alpha=False)
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        b64_pages.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    doc.close()
    return b64_pages


def make_request(client: httpx.Client, url: str, model: str, b64: str) -> dict:
    # temperature=0.1/top_p=0.95 + ngram no-repeat logits processor matches
    # the production retry config and keeps the decoder off the deterministic
    # loop attractor that plagues temperature=0 (see test-deepseek.py
    # docstring). max_tokens=2048 hard-caps any outlier; observed mean is
    # <700 tokens/page so this does not truncate real content.
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    {"type": "text", "text": "Free OCR."},
                ],
            }
        ],
        "temperature": 0.1,
        "top_p": 0.95,
        "max_tokens": 2048,
        "stream": False,
        "vllm_xargs": {
            "ngram_size": 10,
            "window_size": 300,
            "whitelist_token_ids": [128821, 128822],
        },
    }
    t0 = time.perf_counter()
    r = client.post(url, json=payload, timeout=300.0)
    latency = time.perf_counter() - t0
    r.raise_for_status()
    data = r.json()
    usage = data.get("usage") or {}
    return {
        "latency": latency,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "finish_reason": data["choices"][0].get("finish_reason"),
    }


def run_level(
    pages_b64: list[str],
    base_url: str,
    model: str,
    concurrency: int,
    duration_s: float,
    warmup_s: float,
) -> dict:
    url = f"{base_url}/chat/completions"
    limits = httpx.Limits(
        max_connections=concurrency + 8,
        max_keepalive_connections=concurrency + 8,
    )
    client = httpx.Client(limits=limits, http2=False)

    state_lock = threading.Lock()
    results: list[dict] = []
    stop_flag = threading.Event()
    t_warm_end = [0.0]
    t_measure_end = [0.0]
    # Per-thread RNG seeded per level so each level draws a fresh, decorrelated
    # page sequence. Uniform random sampling with replacement breaks the
    # round-robin correlation where every level repeatedly hit the same
    # high-truncation pages in the same order.
    rng_seed = random.randrange(2**31)
    thread_local = threading.local()

    def worker():
        nonlocal results
        thread_local.rng = random.Random(rng_seed ^ threading.get_ident())
        while not stop_flag.is_set():
            b64 = thread_local.rng.choice(pages_b64)
            try:
                r = make_request(client, url, model, b64)
            except Exception as e:
                r = {"error": str(e), "latency": 0.0,
                     "prompt_tokens": 0, "completion_tokens": 0}
            t_done = time.perf_counter()
            with state_lock:
                if t_warm_end[0] <= t_done <= t_measure_end[0]:
                    results.append(r)

    t_start = time.perf_counter()
    t_warm_end[0] = t_start + warmup_s
    t_measure_end[0] = t_warm_end[0] + duration_s

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures: list[Future] = [pool.submit(worker) for _ in range(concurrency)]
        # Let workers run until measurement window closes; then tell them to stop
        # after their current request completes.
        while time.perf_counter() < t_measure_end[0]:
            time.sleep(0.25)
        stop_flag.set()
        for f in futures:
            f.result()

    client.close()

    completed = [r for r in results if "error" not in r]
    errors = [r for r in results if "error" in r]
    n = len(completed)
    if n == 0:
        return {"concurrency": concurrency, "n_requests": 0,
                "errors": len(errors), "pages_per_sec": 0.0}

    total_completion = sum(r["completion_tokens"] for r in completed)
    total_prompt = sum(r["prompt_tokens"] for r in completed)
    latencies = [r["latency"] for r in completed]
    duration_actual = duration_s  # by construction
    return {
        "concurrency": concurrency,
        "n_requests": n,
        "errors": len(errors),
        "duration_s": duration_actual,
        "pages_per_sec": n / duration_actual,
        "output_tok_per_sec": total_completion / duration_actual,
        "total_tok_per_sec": (total_completion + total_prompt) / duration_actual,
        "mean_completion_tok": total_completion / n,
        "latency_p50": statistics.median(latencies),
        "latency_p95": sorted(latencies)[int(0.95 * (len(latencies) - 1))]
        if len(latencies) > 1 else latencies[0],
        "latency_mean": statistics.mean(latencies),
        "length_trunc_rate": sum(
            1 for r in completed if r["finish_reason"] == "length"
        ) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf",
                    default="deepseek-OCR/R8.0_GX_System_Description_Guide SNIPPET.pdf")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-OCR")
    ap.add_argument("--dpi", type=int, default=144)
    ap.add_argument("--levels", default="1,2,4,8,16,24,32,48,64,80,100,128")
    ap.add_argument("--duration", type=float, default=60.0,
                    help="measurement window seconds")
    ap.add_argument("--warmup", type=float, default=15.0)
    args = ap.parse_args()

    levels = [int(x) for x in args.levels.split(",") if x]
    print(f"[render] rendering {args.pdf} @ {args.dpi} DPI ...", flush=True)
    pages_b64 = render_pages(Path(args.pdf), args.dpi)
    print(f"[render] {len(pages_b64)} pages ready "
          f"(avg {sum(len(p) for p in pages_b64)//len(pages_b64)//1024} KB b64)",
          flush=True)

    # Warm the model once so the first level isn't penalized.
    client = httpx.Client(timeout=300.0)
    try:
        make_request(client, f"{args.base_url}/chat/completions",
                     args.model, pages_b64[0])
    finally:
        client.close()

    header = (f"{'conc':>4} {'n':>5} {'err':>4} {'pg/s':>7} "
              f"{'out_tok/s':>10} {'lat_p50':>8} {'lat_p95':>8} "
              f"{'mean_out':>9} {'trunc':>6}")
    print(header, flush=True)
    print("-" * len(header), flush=True)

    all_results = []
    for c in levels:
        r = run_level(pages_b64, args.base_url, args.model, c,
                      args.duration, args.warmup)
        all_results.append(r)
        if r["n_requests"] == 0:
            print(f"{c:>4d}   -- level failed ({r['errors']} errors)",
                  flush=True)
            continue
        print(f"{c:>4d} {r['n_requests']:>5d} {r['errors']:>4d} "
              f"{r['pages_per_sec']:>7.2f} "
              f"{r['output_tok_per_sec']:>10.1f} "
              f"{r['latency_p50']:>8.2f} {r['latency_p95']:>8.2f} "
              f"{r['mean_completion_tok']:>9.0f} "
              f"{r['length_trunc_rate']*100:>5.1f}%", flush=True)

    print("\n[done]", flush=True)


if __name__ == "__main__":
    main()
