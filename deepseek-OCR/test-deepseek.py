"""DeepSeek-OCR PDF harness (OpenAI-compatible endpoint).

Renders a PDF to per-page PIL images, sends each to a vLLM-served DeepSeek-OCR
model, aggregates the transcriptions into a .txt file, and optionally parses
grounding tokens to produce cropped image regions + an annotated overlay PDF +
a metadata.json audit trail.

Architecture and tuning choices below are driven by three findings established
empirically during bring-up; they are documented inline because none of them
are obvious from the DeepSeek-OCR reference code or paper alone:

1.  Input tokens are hard-capped by the vision encoder, not by image size.
    DeepEncoder uses mode-quantized token budgets (Tiny=64, Small=100,
    Base=256, Large=400, Gundam=n*100+256, Gundam-M=n*256+400). Our server
    runs in Gundam with MAX_CROPS=6 (n ceiling), so every request consumes
    ~919 input tokens: 856 vision + ~63 chat-template/prompt overhead. Since
    max_model_len is 8192, ~7000 output tokens are always available.
    Consequently we do NOT pass `max_tokens` to the server; vLLM defaults it
    to `max_model_len - prompt_tokens` automatically.

2.  At temperature=0 the decoder occasionally falls into a repetition
    attractor (typically a blank `<td></td>` loop in degenerate tables or a
    variant-repetition loop in dense product catalogs). vLLM's continuous
    batching means temperature=0 is only locally deterministic: adjacent
    requests in the batch change the attention GEMM's floating-point
    summation order, flipping argmax at ambiguous logits and diverging
    downstream. Loops therefore appear and disappear run-to-run at the same
    nominal config.

3.  Neither `frequency_penalty` nor `presence_penalty` is a good loop
    mitigation. frequency_penalty is count-weighted and unbounded, so it
    penalizes legitimate grammar tokens alongside pathological ones and
    destabilizes output above ~0.225 (empirical sweep, step=0.025, n=21
    pages). presence_penalty is well-behaved but doesn't meaningfully
    reduce loop rate. The working mitigation is: first attempt deterministic
    for reproducibility, then on `finish_reason='length'` retry with
    progressively jittered sampling. In our testing every loop recovered on
    attempt 2 (temperature=0.1, top_p=0.95); attempt 3 has not been needed.
"""

import argparse
import ast
import base64
import io
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import fitz
import img2pdf
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

DEFAULT_MODEL = "deepseek-ai/DeepSeek-OCR"
DEFAULT_BASE_URL = "http://localhost:8000/v1"

# Two prompt modes. `Free OCR.` emits plain markdown with no layout metadata;
# `<|grounding|>Convert the document to markdown.` instructs the model to
# interleave `<|ref|>label<|/ref|><|det|>[[x1,y1,x2,y2]]<|/det|>` triples into
# the output, which we parse for the --extract-layout path. The chat template
# automatically prepends an `<image>` placeholder from the `image_url` content
# block, matching the reference script's literal `'<image>\n' + PROMPT`.
DEFAULT_PROMPT = "Free OCR."
GROUNDED_PROMPT = "<|grounding|>Convert the document to markdown."

# 144 DPI (2x the PDF base of 72) produces crisp images with legible small
# fonts. Raising this does NOT increase input tokens - the encoder resizes/
# pads to its mode's native resolution regardless (see item 1 in module
# docstring). 144 DPI is just enough detail for the vision model to OCR
# reliably while keeping rendering fast.
DEFAULT_DPI = 144

# Matches the server's `--max-num-seqs=100` admission cap with DP=2 (2 model
# replicas on 2 L40S GPUs). Worker count is thread-pool concurrency for the
# HTTPS requests; the server does its own intra-replica continuous batching.
DEFAULT_WORKERS = 32

# N-gram no-repeat filter (DeepSeek-OCR's custom vLLM logits processor). At
# each decode step it forbids any token that would complete an n-gram already
# present in the last `window_size` tokens, except for whitelisted IDs.
# Reference values (30 / 90) are too loose: pages with densely-varied
# repetition (e.g. `nx4000Muxponder (n<=2)` with minor token perturbations)
# never trigger the 30-gram match and loop until max_tokens. We use 10 / 300:
# small n-gram catches short loop units, wide window catches loops as soon as
# they nucleate.
DEFAULT_NGRAM_SIZE = 10
DEFAULT_WINDOW_SIZE = 300

# Both penalties default to 0 after extensive sweeps (see module docstring
# item 3). Left as knobs because they may be useful for future workloads.
DEFAULT_FREQUENCY_PENALTY = 0.0
DEFAULT_PRESENCE_PENALTY = 0.0

# Number of retry attempts after the initial deterministic call.
# MAX_RETRIES=2 means up to 3 total API calls per page. In 5 consecutive
# full-PDF runs (21 pages each) we saw 3 truncation events total, all
# recovered on attempt 2. Attempt 3 is a safety margin we haven't cashed in.
DEFAULT_MAX_RETRIES = 2

# Progressive jitter schedule: each retry increases temperature and tightens
# nucleus sampling further. The idea is to nudge the decoder off whatever
# deterministic attractor caused the loop, while staying near the high-
# probability mass so we don't go off-task (hallucinate or skip content).
RETRY_SAMPLING: list[dict] = [
    {"temperature": 0.1, "top_p": 0.95},
    {"temperature": 0.3, "top_p": 0.90},
    {"temperature": 0.5, "top_p": 0.85},
]

# Grounded-mode output format. The model emits triples like:
#   <|ref|>title<|/ref|><|det|>[[91, 37, 362, 50]]<|/det|>
# The `det` payload is a list of 4-tuples (a single ref may cover multiple
# boxes, e.g. multi-line titles). Coordinates are in a fixed 0-999 normalized
# space regardless of actual page dimensions (see COORD_SPACE).
REF_PATTERN = re.compile(r"<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>", re.DOTALL)

# Per the DeepSeek-OCR paper, detection coordinates are emitted on a 0-999
# integer grid to keep output tokens compact. We scale them to pixel space
# using the actual rendered page dimensions at runtime.
COORD_SPACE = 999


# ---------------------------------------------------------------------------
# PDF -> images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: Path, dpi: int = DEFAULT_DPI) -> list[Image.Image]:
    """Rasterize each PDF page to a PIL Image at the given DPI.

    PyMuPDF's default matrix is 1:1 with 72 DPI (PDF user-space units). We
    scale by dpi/72 to get the requested resolution. `alpha=False` yields a
    3-channel RGB pixmap directly so we avoid an alpha-compositing step.

    `Image.MAX_IMAGE_PIXELS = None` disables Pillow's decompression-bomb
    guard; legitimate scanned pages at 300+ DPI can exceed the default cap.
    """
    images: list[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)
    Image.MAX_IMAGE_PIXELS = None

    with fitz.open(pdf_path) as doc:
        for page in doc:
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            img = Image.open(io.BytesIO(pixmap.tobytes("png")))
            images.append(img.convert("RGB"))
    return images


def image_to_data_url(image: Image.Image) -> str:
    """Encode a PIL image as a base64 PNG data URL.

    vLLM's OpenAI-compatible `/v1/chat/completions` endpoint accepts images
    via the `image_url` content block. The reference spec allows either a
    URL the server can fetch or an inline data URL; we use the latter to
    avoid needing a static file server. PNG (lossless) preserves thin glyph
    strokes that JPEG compression tends to blur at OCR-relevant scales.
    """
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


# ---------------------------------------------------------------------------
# Single-page OCR call + retry wrapper
# ---------------------------------------------------------------------------

def ocr_page(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    client: OpenAI,
    model: str,
    prompt: str,
    image: Image.Image,
    ngram_size: int,
    window_size: int,
    frequency_penalty: float,
    presence_penalty: float = 0.0,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> tuple[str, int, int, str]:
    """Send one page to DeepSeek-OCR and return (text, prompt_tok, out_tok, finish_reason).

    Deliberately does NOT pass `max_tokens`. vLLM auto-fills it with
    `max_model_len - prompt_tokens`, which for DeepSeek-OCR in Gundam mode
    means ~7273 output tokens of budget for every request. Passing an
    explicit value only risks under-sizing (we tried 4096 and 8192 early on;
    the former truncated legitimate dense pages, the latter exceeded
    max_model_len once prompt tokens were added).

    `skip_special_tokens=False` is REQUIRED for the grounded prompt so the
    `<|ref|>` / `<|det|>` sentinels reach the client and can be parsed.

    `whitelist_token_ids=[128821, 128822]` exempts `<td>` and `</td>` from
    the no-repeat filter so legitimate tables can emit many cell tags. This
    whitelist is a double-edged sword: it creates an escape hatch the decoder
    can exploit when it falls into a blank-table loop, which is the exact
    failure mode the retry wrapper is designed to catch.
    """
    data_url = image_to_data_url(image)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        temperature=temperature,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty,
        extra_body={
            "skip_special_tokens": False,
            "vllm_xargs": {
                "ngram_size": ngram_size,
                "window_size": window_size,
                "whitelist_token_ids": [128821, 128822],
            },
        },
    )
    choice = response.choices[0]
    text = choice.message.content or ""
    finish_reason = choice.finish_reason or "unknown"
    usage = response.usage
    prompt_tokens = getattr(usage, "prompt_tokens", 0) if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) if usage else 0
    return text, prompt_tokens, completion_tokens, finish_reason


def ocr_page_with_retry(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
    client: OpenAI,
    model: str,
    prompt: str,
    image: Image.Image,
    ngram_size: int,
    window_size: int,
    frequency_penalty: float,
    presence_penalty: float,
    max_retries: int,
) -> tuple[str, int, int, str, list[dict]]:
    """Call `ocr_page`; on truncation, retry with jittered sampling.

    The first attempt is always deterministic (temperature=0, top_p=1) so
    normal pages produce reproducible output. When `finish_reason == 'length'`
    we interpret that as "the decoder hit an attractor and ran to max_tokens"
    - the output is almost always unusable garbage in that case - and retry
    with progressively more jitter from RETRY_SAMPLING. We stop as soon as
    any attempt returns a non-length finish_reason.

    Returns the final attempt's (text, prompt_tok, out_tok, finish_reason)
    plus the full per-attempt log for audit.
    """
    attempts: list[dict] = []
    configs = [{"temperature": 0.0, "top_p": 1.0}] + RETRY_SAMPLING[:max_retries]
    text, in_tok, out_tok, finish = "", 0, 0, "unknown"
    for i, cfg in enumerate(configs):
        text, in_tok, out_tok, finish = ocr_page(
            client, model, prompt, image,
            ngram_size, window_size,
            frequency_penalty, presence_penalty,
            temperature=cfg["temperature"], top_p=cfg["top_p"],
        )
        attempts.append({
            "attempt": i + 1,
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "finish_reason": finish,
            "output_tokens": out_tok,
        })
        if finish != "length":
            break
    return text, in_tok, out_tok, finish, attempts


# ---------------------------------------------------------------------------
# Grounding-token parsing (--extract-layout path only)
# ---------------------------------------------------------------------------

def parse_refs(text: str) -> list[tuple[str, list[tuple[int, int, int, int]]]]:
    """Extract (label, boxes) pairs from grounded-prompt output.

    The model's output format is:
        <|ref|>LABEL<|/ref|><|det|>[[x1,y1,x2,y2], ...]<|/det|>
    where the det payload is a Python-literal list of 4-tuples. We use
    `ast.literal_eval` rather than `eval` because the payload comes from an
    LLM's tokens - not trusted input - and literal_eval only parses Python
    constants. Malformed groups are silently dropped (the model sometimes
    emits truncated or malformed coord lists near the output tail).
    """
    parsed: list[tuple[str, list[tuple[int, int, int, int]]]] = []
    for label, raw_boxes in REF_PATTERN.findall(text):
        try:
            boxes = ast.literal_eval(raw_boxes)
        except (ValueError, SyntaxError):
            continue
        if not isinstance(boxes, list):
            continue
        clean: list[tuple[int, int, int, int]] = []
        for box in boxes:
            if isinstance(box, (list, tuple)) and len(box) == 4:
                clean.append(tuple(int(v) for v in box))  # type: ignore[arg-type]
        if clean:
            parsed.append((label, clean))
    return parsed


def scale_box(
    box: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int]:
    """Map a 0-999 normalized bbox to pixel coordinates of the rendered page.

    The 999 grid is coarse - roughly 0.1% of image dimensions per unit - so
    crops are accurate to a few pixels at 144 DPI. This is fine for region
    extraction but should not be relied on for pixel-precise layout
    reconstruction.
    """
    x1, y1, x2, y2 = box
    return (
        int(x1 / COORD_SPACE * width),
        int(y1 / COORD_SPACE * height),
        int(x2 / COORD_SPACE * width),
        int(y2 / COORD_SPACE * height),
    )


# ---------------------------------------------------------------------------
# Overlay PDF rendering
# ---------------------------------------------------------------------------

def render_layout(  # pylint: disable=too-many-locals
    image: Image.Image,
    refs: list[tuple[str, list[tuple[int, int, int, int]]]],
    crops_dir: Path,
    page_idx: int,
) -> tuple[Image.Image, list[dict]]:
    """Draw bounding boxes onto a page and crop out regions labeled `image`.

    Colors are drawn from a per-page seeded RNG so re-running the pipeline
    produces the same overlay palette for a given page - makes diffs between
    runs easier to eyeball. Title regions get a thicker outline (4px vs 2px)
    because they are the visual landmarks a reviewer looks for first.

    Regions labeled `image` in the ground-truth schema are the model's
    identification of figures/diagrams; we crop those from the original
    (un-annotated) page and save as JPGs so they can be linked from markdown
    downstream. The JPG quality=92 keeps files small without introducing
    visible compression on diagram-style content.

    Returns the annotated canvas plus a list of per-region records that feed
    metadata.json.
    """
    width, height = image.size
    canvas = image.copy()
    draw = ImageDraw.Draw(canvas)

    # Semi-transparent fill layer is a separate RGBA image pasted over the
    # canvas at the end. Drawing fills directly on the base RGB image would
    # permanently tint the pixels; the overlay approach keeps the underlying
    # text readable through the tint.
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()

    rng = random.Random(page_idx)
    records: list[dict] = []
    img_idx = 0

    for label, boxes in refs:
        # Keep colors muted (max 200 on R, G; up to 255 on B) for contrast
        # against document white; alpha=40 is barely visible but enough to
        # convey region groupings.
        color = (rng.randint(0, 200), rng.randint(0, 200), rng.randint(0, 255))
        fill = color + (40,)
        for box in boxes:
            raw = scale_box(box, width, height)
            # Clamp so x0<=x1 and y0<=y1; degenerate model output can invert coords.
            scaled = (
                min(raw[0], raw[2]), min(raw[1], raw[3]),
                max(raw[0], raw[2]), max(raw[1], raw[3]),
            )
            record: dict = {
                "label": label,
                "box_999": list(box),
                "box_px": list(scaled),
            }
            if label == "image":
                try:
                    crop = image.crop(scaled)
                    fname = f"page{page_idx + 1:03d}_img{img_idx:03d}.jpg"
                    crop.save(crops_dir / fname, quality=92)
                    record["file"] = fname
                    img_idx += 1
                except Exception as exc:  # pylint: disable=broad-except
                    # Degenerate boxes (e.g. negative size from a misparsed
                    # coord triple) can raise; record the error and continue
                    # so one bad box doesn't kill the whole page.
                    record["crop_error"] = str(exc)

            width_px = 4 if label == "title" else 2
            draw.rectangle(scaled, outline=color, width=width_px)
            overlay_draw.rectangle(scaled, fill=fill, outline=(0, 0, 0, 0), width=1)

            # Label text sits just above the top-left corner of the box,
            # with a white backdrop so it's legible against both light page
            # backgrounds and the semi-transparent region tint.
            text_x = scaled[0]
            text_y = max(0, scaled[1] - 15)
            bbox = draw.textbbox((0, 0), label, font=font)
            draw.rectangle(
                [text_x, text_y, text_x + (bbox[2] - bbox[0]), text_y + (bbox[3] - bbox[1])],
                fill=(255, 255, 255),
            )
            draw.text((text_x, text_y), label, font=font, fill=color)
            records.append(record)

    canvas.paste(overlay, (0, 0), overlay)
    return canvas, records


def pil_images_to_pdf(images: list[Image.Image], output_path: Path) -> None:
    """Encode a list of PIL images into a single PDF via img2pdf.

    img2pdf losslessly wraps each JPEG (or other supported image) as a PDF
    page - no re-encoding, no raster transform. The JPEG quality=92 matches
    the per-page overlay save quality so the annotated PDF faithfully
    represents what render_layout produced.
    """
    payload: list[bytes] = []
    for img in images:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, format="JPEG", quality=92)
        payload.append(buf.getvalue())
    output_path.write_bytes(img2pdf.convert(payload))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Build the CLI parser. Defaults reflect the empirically-validated values
    from bring-up; each knob has a doc-string-level explanation above."""
    parser = argparse.ArgumentParser(
        description="Run DeepSeek-OCR on a PDF via an OpenAI-compatible endpoint."
    )
    parser.add_argument("--pdf", type=Path, required=True, help="Path to input PDF")
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Path to output text file (default: <pdf>.txt, or inside layout dir when --extract-layout)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument(
        "--prompt", default=None,
        help=f"Prompt text (default: '{DEFAULT_PROMPT}', or '{GROUNDED_PROMPT}' when --extract-layout is set)",
    )
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument(
        "--workers", type=int, default=DEFAULT_WORKERS,
        help="Concurrent page requests; server's max_num_seqs is the ceiling.",
    )
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument(
        "--ngram-size", type=int, default=DEFAULT_NGRAM_SIZE,
        help="No-repeat n-gram length (smaller catches shorter loop units)",
    )
    parser.add_argument(
        "--window-size", type=int, default=DEFAULT_WINDOW_SIZE,
        help="No-repeat look-back window (larger detects loops sooner)",
    )
    parser.add_argument(
        "--frequency-penalty", type=float, default=DEFAULT_FREQUENCY_PENALTY,
        help="Sampler-level frequency penalty; destabilizes output above ~0.225.",
    )
    parser.add_argument(
        "--presence-penalty", type=float, default=DEFAULT_PRESENCE_PENALTY,
        help="Sampler-level presence penalty; benign but ineffective for loop mitigation.",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"Retries on finish_reason='length' with jittered sampling (max {len(RETRY_SAMPLING)}).",
    )
    parser.add_argument(
        "--extract-layout", action="store_true",
        help="Parse grounding tokens, crop image regions, and write an annotated PDF.",
    )
    parser.add_argument(
        "--layout-dir", type=Path, default=None,
        help="Output directory for layout artifacts (default: <pdf_stem>_layout_<timestamp>/).",
    )
    parser.add_argument(
        "--skip-truncated", action="store_true",
        help="Omit still-truncated pages from the .txt output (metadata.json retains them).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Pipeline driver
# ---------------------------------------------------------------------------

def main() -> None:  # pylint: disable=too-many-locals,too-many-branches,too-many-statements
    """End-to-end pipeline: render PDF, OCR every page concurrently, optionally
    extract layout artifacts, write outputs.

    Layout of output artifacts when --extract-layout is set:

        <pdf_stem>_layout_<YYYYMMDDTHHMMSS>/
            <pdf_stem>.txt                   -- concatenated transcriptions
            <pdf_stem>_annotated.pdf         -- overlay PDF with bboxes + labels
            images/                          -- JPG crops of figure regions
                page001_img000.jpg, ...
            metadata.json                    -- full audit trail (see below)

    metadata.json captures both the run configuration (model, prompt,
    sampling knobs, retry schedule, timing) and per-page detail (finish
    reason, every attempt's sampling + finish + output tokens, region list
    with pixel and 999-space coordinates). It is the primary artifact for
    diagnosing quality issues after the fact.
    """
    args = parse_args()

    if not args.pdf.exists():
        raise FileNotFoundError(f"PDF not found: {args.pdf}")

    # Default to grounded mode when the user asks for layout extraction;
    # the plain `Free OCR.` prompt does not emit `<|ref|>`/`<|det|>` sentinels
    # so the layout pipeline would have nothing to parse.
    prompt = args.prompt or (GROUNDED_PROMPT if args.extract_layout else DEFAULT_PROMPT)

    layout_dir: Path | None = None
    crops_dir: Path | None = None
    if args.extract_layout:
        # Timestamp each layout run so consecutive invocations on the same PDF
        # don't clobber each other (useful when comparing retry outcomes).
        stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
        layout_dir = args.layout_dir or args.pdf.parent / f"{args.pdf.stem}_layout_{stamp}"
        crops_dir = layout_dir / "images"
        crops_dir.mkdir(parents=True, exist_ok=True)
        output_path = args.output or layout_dir / f"{args.pdf.stem}.txt"
    else:
        output_path = args.output or args.pdf.with_suffix(".txt")

    print(f"Loading PDF: {args.pdf}")
    images = pdf_to_images(args.pdf, dpi=args.dpi)
    print(f"Rendered {len(images)} page(s) at {args.dpi} DPI")
    print(f"Prompt: {prompt!r}")

    client = OpenAI(api_key=args.api_key, base_url=args.base_url, timeout=args.timeout)

    # Per-page accumulators, indexed by page number (0-based). We pre-allocate
    # so out-of-order ThreadPoolExecutor completion can write into the right
    # slot without needing post-sort.
    results: list[str] = [""] * len(images)
    usages: list[tuple[int, int]] = [(0, 0)] * len(images)
    finish_reasons: list[str] = ["unknown"] * len(images)
    attempt_logs: list[list[dict]] = [[] for _ in images]
    failures: list[tuple[int, str]] = []
    started_at = datetime.now(timezone.utc)
    start = time.time()

    # Pages are processed concurrently at the HTTP level; the server does its
    # own continuous batching so requests can be in-flight without being
    # strictly one-per-GPU. `ocr_page_with_retry` handles truncation internally
    # so failures here are only connection/5xx/etc errors, not retryable loops.
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                ocr_page_with_retry,
                client,
                args.model,
                prompt,
                img,
                args.ngram_size,
                args.window_size,
                args.frequency_penalty,
                args.presence_penalty,
                args.max_retries,
            ): idx
            for idx, img in enumerate(images)
        }
        with tqdm(total=len(futures), desc="OCR pages") as pbar:
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    text, in_tok, out_tok, finish_reason, attempts = future.result()
                except Exception as exc:  # pylint: disable=broad-except
                    # Network/server exceptions bubble up here; treat them as
                    # a per-page soft failure so the rest of the PDF still
                    # processes. The stub string goes into the .txt output
                    # so downstream consumers can spot the missing page.
                    failures.append((idx, str(exc)))
                    tqdm.write(f"[page {idx + 1}] FAILED: {exc}")
                    results[idx] = f"[OCR FAILED: {exc}]"
                else:
                    results[idx] = text
                    usages[idx] = (in_tok, out_tok)
                    finish_reasons[idx] = finish_reason
                    attempt_logs[idx] = attempts
                    n_att = len(attempts)
                    if finish_reason == "length":
                        marker = f", STILL TRUNCATED after {n_att} attempts"
                    elif n_att > 1:
                        marker = f", recovered on attempt {n_att}"
                    else:
                        marker = ""
                    tqdm.write(
                        f"\n===== Page {idx + 1} "
                        f"(input={in_tok} tokens, output={out_tok} tokens{marker}) =====\n"
                        f"{text}\n"
                    )
                finally:
                    pbar.update(1)

    elapsed = time.time() - start
    completed_at = datetime.now(timezone.utc)
    total_in = sum(u[0] for u in usages)
    total_out = sum(u[1] for u in usages)
    truncated = [i for i, fr in enumerate(finish_reasons) if fr == "length"]
    attempts_per_page = [len(log) for log in attempt_logs]

    # "Recovered on attempt N" means the page eventually produced a non-
    # length finish on attempt N > 1. Grouping by N tells us how much of our
    # retry budget we're actually using - empirically attempt 2 has always
    # been sufficient.
    recovered_on: dict[int, list[int]] = {}
    for i, log in enumerate(attempt_logs):
        if len(log) > 1 and log[-1].get("finish_reason") != "length":
            recovered_on.setdefault(len(log), []).append(i + 1)

    print(
        f"OCR completed in {elapsed:.2f}s ({len(failures)} failure(s), "
        f"{len(truncated)} still truncated); total input={total_in} tokens, "
        f"total output={total_out} tokens"
    )
    for att, pages in sorted(recovered_on.items()):
        print(f"  recovered on attempt {att}: pages {pages}")

    # Assemble the human-readable .txt. Each page gets a banner header that
    # records its token budget and truncation/recovery state; if the user
    # passed --skip-truncated, still-failing pages are dropped from the .txt
    # but retained in metadata.json so evidence of failure isn't lost.
    page_sep = "\n<--- Page Split --->\n"
    annotated_pages: list[str] = []
    for idx, text in enumerate(results):
        is_trunc = finish_reasons[idx] == "length"
        if is_trunc and args.skip_truncated:
            continue
        suffix = ", TRUNCATED" if is_trunc else ""
        annotated_pages.append(
            f"===== Page {idx + 1} "
            f"(input={usages[idx][0]} tokens, output={usages[idx][1]} tokens{suffix}) =====\n"
            f"{text}"
        )
    combined = page_sep.join(annotated_pages)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(combined, encoding="utf-8")
    print(f"Wrote output to {output_path}")

    if args.extract_layout and layout_dir is not None and crops_dir is not None:
        # Layout pipeline: parse each page's grounding tokens, render overlay,
        # and crop figure regions. This runs serially because rendering is
        # CPU-bound in PIL and we've already saturated GPU-side parallelism
        # during the OCR phase.
        print(f"Extracting layout artifacts into {layout_dir}")
        per_page: list[dict] = []
        overlays: list[Image.Image] = []
        for idx, (img, text) in enumerate(zip(images, results)):
            refs = parse_refs(text)
            overlay_img, records = render_layout(img, refs, crops_dir, idx)
            overlays.append(overlay_img)
            per_page.append(
                {
                    "page": idx + 1,
                    "input_tokens": usages[idx][0],
                    "output_tokens": usages[idx][1],
                    "finish_reason": finish_reasons[idx],
                    "truncated": finish_reasons[idx] == "length",
                    "attempts": attempt_logs[idx],
                    "n_attempts": attempts_per_page[idx],
                    "page_width_px": img.width,
                    "page_height_px": img.height,
                    "regions": records,
                }
            )

        annotated_pdf = layout_dir / f"{args.pdf.stem}_annotated.pdf"
        pil_images_to_pdf(overlays, annotated_pdf)

        metadata = {
            "source_pdf": str(args.pdf.resolve()),
            "source_pdf_bytes": args.pdf.stat().st_size,
            "num_pages": len(images),
            "dpi": args.dpi,
            "model": args.model,
            "base_url": args.base_url,
            "prompt": prompt,
            "workers": args.workers,
            "ngram_size": args.ngram_size,
            "window_size": args.window_size,
            "frequency_penalty": args.frequency_penalty,
            "presence_penalty": args.presence_penalty,
            "max_retries": args.max_retries,
            "retry_sampling": RETRY_SAMPLING[: args.max_retries],
            "truncated_pages": [i + 1 for i in truncated],
            "recovered_on_retry": {
                str(att): pages for att, pages in sorted(recovered_on.items())
            },
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "elapsed_seconds": round(elapsed, 3),
            "total_input_tokens": total_in,
            "total_output_tokens": total_out,
            "failures": [{"page": i + 1, "error": e} for i, e in failures],
            "annotated_pdf": annotated_pdf.name,
            "per_page": per_page,
        }
        (layout_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        print(f"  crops -> {crops_dir}")
        print(f"  annotated PDF -> {annotated_pdf}")
        print(f"  metadata -> {layout_dir / 'metadata.json'}")

    if failures:
        print("\nFailed pages:")
        for idx, err in sorted(failures):
            print(f"  page {idx + 1}: {err}")


if __name__ == "__main__":
    main()
