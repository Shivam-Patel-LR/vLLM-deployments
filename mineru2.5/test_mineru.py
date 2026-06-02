"""MinerU2.5 smoke test against the vLLM OpenAI-compatible endpoint.

Always checks /v1/models. If an image is passed and mineru-vl-utils + Pillow
are importable, runs one two-step extraction and prints the block count.

    uv run test_mineru.py                       # health only
    uv run test_mineru.py --image page1.png     # health + single-page extract
"""
import argparse
import sys
import urllib.request

MODEL = "MinerU2.5-Pro-2604-1.2B"


def check_models(base_url):
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as resp:
        body = resp.read().decode()
    if MODEL not in body:
        sys.exit(f"FAIL: {MODEL!r} not served at {base_url}\n{body}")
    print(f"OK: {MODEL} is serving at {base_url}")


def extract(base_url, image_path):
    try:
        from PIL import Image
        from mineru_vl_utils import MinerUClient
    except ImportError as exc:
        sys.exit(f"SKIP extract: {exc} (pip install 'mineru-vl-utils[vllm]' Pillow)")
    client = MinerUClient(backend="http-client", server_url=base_url,
                          model_name=MODEL, max_concurrency=64)
    [result] = client.concurrent_two_step_extract([Image.open(image_path)])
    print(f"OK: extracted {len(result.blocks)} blocks from {image_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--image", help="page image to extract (PNG/JPG)")
    args = ap.parse_args()
    check_models(args.base_url)
    if args.image:
        extract(args.base_url, args.image)


if __name__ == "__main__":
    main()
