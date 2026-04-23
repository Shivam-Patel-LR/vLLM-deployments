"""Compare standalone test-deepseek.py harness vs docling-serve for DeepSeek-OCR.

Runs both systems against a given PDF, captures timing, word counts, and
figure/picture detection, then prints a side-by-side summary.
"""

import argparse
import io
import json
import re
import subprocess
import sys
import time
import zipfile
from pathlib import Path

import httpx

DOCLING_SERVE_URL = "http://localhost:5001"
# docling-serve reaches vLLM via the Docker gateway for the open-webui network
VLLM_URL_FROM_DOCLING = "http://172.25.0.1:8000/v1/chat/completions"
VLLM_MODEL = "deepseek-ai/DeepSeek-OCR"
GROUNDED_PROMPT = "<|grounding|>Convert the document to markdown."


# ---------------------------------------------------------------------------
# System 1: standalone harness
# ---------------------------------------------------------------------------

def run_standalone(pdf_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_txt = output_dir / "output.txt"
    layout_dir = output_dir / "layout"

    cmd = [
        sys.executable, str(Path(__file__).parent / "test-deepseek.py"),
        "--pdf", str(pdf_path),
        "--output", str(out_txt),
        "--extract-layout",
        "--layout-dir", str(layout_dir),
        "--workers", "32",
    ]

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0:
        print("[standalone] stderr:", result.stderr[-2000:])
        return {"error": result.stderr[-500:], "elapsed": elapsed}

    text = out_txt.read_text() if out_txt.exists() else ""
    metadata_path = layout_dir / "metadata.json" if layout_dir.exists() else None
    metadata = {}
    if metadata_path and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())

    per_page = metadata.get("per_page", [])
    num_pages = metadata.get("num_pages", len(per_page))
    truncated = len(metadata.get("truncated_pages", []))
    retried = sum(1 for p in per_page if p.get("n_attempts", 1) > 1)
    figure_count = sum(
        sum(1 for r in p.get("regions", []) if r.get("label") in ("figure", "image"))
        for p in per_page
    )

    return {
        "elapsed": elapsed,
        "word_count": len(text.split()),
        "char_count": len(text),
        "num_pages": num_pages,
        "truncated_pages": truncated,
        "retried_pages": retried,
        "figures_extracted": figure_count,
        "output_file": str(out_txt),
        "layout_dir": str(layout_dir),
        "preview": text[:800],
    }


# ---------------------------------------------------------------------------
# System 2: docling-serve
# ---------------------------------------------------------------------------

def run_docling(pdf_path: Path, output_dir: Path) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)

    # VlmModelApi JSON maps directly to ApiVlmOptions in docling-jobkit
    vlm_api_config = {
        "url": VLLM_URL_FROM_DOCLING,
        "headers": {},
        "params": {
            "model": VLLM_MODEL,
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": 10,
                "window_size": 300,
                "whitelist_token_ids": [128821, 128822],
            },
        },
        "prompt": GROUNDED_PROMPT,
        "scale": 2.0,
        "response_format": "deepseekocr_markdown",
        "temperature": 0.0,
        "timeout": 300,
        "concurrency": 8,
    }

    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    multipart_data = [
        ("pipeline", (None, "vlm")),
        ("vlm_pipeline_model_api", (None, json.dumps(vlm_api_config))),
        ("to_formats", (None, "md")),
        ("image_export_mode", (None, "embedded")),
        ("target_type", (None, "zip")),
        ("include_images", (None, "true")),
        ("document_timeout", (None, "600")),
        ("files", (pdf_path.name, pdf_bytes, "application/pdf")),
    ]

    t0 = time.perf_counter()
    with httpx.Client(timeout=60) as client:
        submit = client.post(
            f"{DOCLING_SERVE_URL}/v1/convert/file/async",
            files=multipart_data,
        )

    if not submit.is_success:
        elapsed = time.perf_counter() - t0
        return {"error": f"Submit HTTP {submit.status_code}: {submit.text[:400]}", "elapsed": elapsed}

    task_id = submit.json().get("task_id")
    print(f"  docling task_id: {task_id}")

    result_resp = None
    with httpx.Client(timeout=30) as client:
        for _ in range(240):  # up to 20 min
            time.sleep(5)
            poll = client.get(f"{DOCLING_SERVE_URL}/v1/status/poll/{task_id}")
            status = poll.json().get("task_status", "unknown")
            if status in ("success", "failure", "partial_success"):
                result_resp = client.get(f"{DOCLING_SERVE_URL}/v1/result/{task_id}")
                break

    elapsed = time.perf_counter() - t0

    if result_resp is None or not result_resp.is_success:
        err = result_resp.text[:400] if result_resp else "polling timed out"
        return {"error": err, "elapsed": elapsed}

    zip_path = output_dir / "docling_output.zip"
    zip_path.write_bytes(result_resp.content)

    md_text = ""
    figure_count = 0
    num_pages = 0
    md_file = output_dir / (pdf_path.stem + ".md")

    try:
        with zipfile.ZipFile(io.BytesIO(result_resp.content)) as zf:
            for name in zf.namelist():
                data = zf.read(name)
                if name.endswith(".md"):
                    md_text = data.decode("utf-8")
                    md_file.write_bytes(data)
                elif name.endswith((".png", ".jpg", ".jpeg")):
                    figure_count += 1
                    img_path = output_dir / name
                    img_path.parent.mkdir(parents=True, exist_ok=True)
                    img_path.write_bytes(data)
        if md_text:
            # Count embedded base64 images (image_export_mode=embedded stores
            # figures inline rather than as separate zip entries)
            embedded = len(re.findall(r"!\[.*?\]\(data:image/", md_text))
            figure_count += embedded
            # Strip base64 payloads before counting words/chars
            md_stripped = re.sub(
                r"data:image/[^;]+;base64,[A-Za-z0-9+/=]+", "IMAGE", md_text
            )
            # Page count: docling uses \n\n---\n\n or bare --- on its own line
            num_pages = (
                len(re.findall(r"\n-{3,}\n", md_text)) + 1
                if re.search(r"\n-{3,}\n", md_text)
                else 0
            )
    except zipfile.BadZipFile:
        md_stripped = ""
        try:
            md_text = result_resp.json().get("md_content", "") or ""
            md_stripped = md_text
        except Exception:
            md_text = result_resp.text
            md_stripped = md_text

    return {
        "elapsed": elapsed,
        "word_count": len(md_stripped.split()),
        "char_count": len(md_stripped),
        "num_pages": num_pages,
        "figures_extracted": figure_count,
        "output_file": str(md_file),
        "preview": md_text[:800],
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _fmt(label: str, v1, v2) -> str:
    return f"  {label:<28} {str(v1):<30} {str(v2)}"


def main():
    parser = argparse.ArgumentParser(description="Compare standalone vs docling-serve DeepSeek-OCR pipelines")
    parser.add_argument("--pdf", required=True, help="Path to input PDF")
    parser.add_argument("--out", default="/tmp/deepseek_compare", help="Output root directory")
    args = parser.parse_args()

    pdf_path = Path(args.pdf).expanduser().resolve()
    out_root = Path(args.out)

    print(f"PDF: {pdf_path.name}  ({pdf_path.stat().st_size / 1e6:.1f} MB)")
    print()

    print("Running System 1: standalone harness (test-deepseek.py) ...")
    r1 = run_standalone(pdf_path, out_root / "standalone")

    print("Running System 2: docling-serve + DeepSeek-OCR ...")
    r2 = run_docling(pdf_path, out_root / "docling")

    print()
    print("=" * 72)
    print(f"{'Metric':<28} {'System 1 (standalone)':<30} {'System 2 (docling-serve)'}")
    print("=" * 72)

    for label, k in [
        ("Elapsed (s)", "elapsed"),
        ("Pages processed", "num_pages"),
        ("Word count", "word_count"),
        ("Char count", "char_count"),
        ("Figures/pictures", "figures_extracted"),
        ("Truncated pages", "truncated_pages"),
        ("Retried pages", "retried_pages"),
        ("Error", "error"),
    ]:
        v1 = r1.get(k, "n/a")
        v2 = r2.get(k, "n/a")
        if isinstance(v1, float):
            v1 = f"{v1:.1f}"
        if isinstance(v2, float):
            v2 = f"{v2:.1f}"
        if v1 == "n/a" and v2 == "n/a":
            continue
        print(_fmt(label, v1, v2))

    print("=" * 72)
    print()

    if "error" not in r1:
        print("--- System 1 output preview ---")
        print(r1.get("preview", "")[:600])
        print()

    if "error" not in r2:
        print("--- System 2 output preview ---")
        print(r2.get("preview", "")[:600])
        print()

    if "error" not in r1:
        print(f"System 1 full output: {r1.get('output_file')}")
        print(f"System 1 layout dir:  {r1.get('layout_dir')}")
    if "error" not in r2:
        print(f"System 2 full output: {r2.get('output_file')}")


if __name__ == "__main__":
    main()
