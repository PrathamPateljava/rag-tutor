"""
Image Extraction, OCR, and Description Module
==============================================
1. Extracts embedded images from PDFs → describes with LLaVA
2. Detects scanned/image-only pages → runs OCR with Tesseract
3. Returns all results as RAG-ready chunks

Setup:
  ollama pull llava
  brew install tesseract
  pip install pymupdf pillow ollama pytesseract
"""

import fitz  # PyMuPDF
import os
import io
import base64
from PIL import Image

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False


IMAGE_OUTPUT_DIR = "./data/processed/images"
VISION_MODEL = "llava"
MIN_IMAGE_SIZE = (50, 50)      # skip tiny images like icons/bullets
OCR_MIN_TEXT_LENGTH = 20       # if page text is shorter than this, treat as scanned


# ─────────────────────────────────────────────
# IMAGE EXTRACTION
# ─────────────────────────────────────────────
def extract_images(filepath: str) -> list[dict]:
    """
    Extract embedded images from a PDF, save them, return metadata.
    Skips tiny images (icons, bullets, decorative elements).
    """
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True)

    doc = fitz.open(filepath)
    pdf_name = os.path.splitext(os.path.basename(filepath))[0]
    extracted = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        images = page.get_images(full=True)

        for img_idx, img_info in enumerate(images):
            xref = img_info[0]

            try:
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                image_ext = base_image["ext"]

                pil_image = Image.open(io.BytesIO(image_bytes))
                width, height = pil_image.size

                if width < MIN_IMAGE_SIZE[0] or height < MIN_IMAGE_SIZE[1]:
                    continue

                img_filename = f"{pdf_name}_p{page_num + 1}_img{img_idx + 1}.{image_ext}"
                img_path = os.path.join(IMAGE_OUTPUT_DIR, img_filename)
                pil_image.save(img_path)

                extracted.append({
                    "filepath": img_path,
                    "page": page_num + 1,
                    "index": img_idx + 1,
                    "width": width,
                    "height": height,
                    "source": os.path.basename(filepath),
                })

            except Exception as e:
                print(f"  Warning: Failed to extract image {img_idx} on page {page_num + 1}: {e}")
                continue

    doc.close()
    print(f"Extracted {len(extracted)} images from {filepath}")
    return extracted


# ─────────────────────────────────────────────
# LLaVA IMAGE DESCRIPTION
# ─────────────────────────────────────────────
def describe_image(image_path: str, context_hint: str = "") -> str:
    """
    Generate a text description of an image using LLaVA via Ollama.
    Falls back to a placeholder if Ollama/LLaVA is unavailable.
    """
    if not OLLAMA_AVAILABLE:
        return f"[Image at {os.path.basename(image_path)} — LLaVA not available]"

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    prompt = (
        "You are describing an image from an educational textbook. "
        "Provide a clear, detailed description that captures all important information "
        "a student would need to understand the concept being illustrated. "
        "Include any text, labels, numbers, axes, or relationships shown. "
        "Be specific and factual — do not speculate beyond what is visible."
    )

    if context_hint:
        prompt += f"\n\nThis image appears in a section about: {context_hint}"

    try:
        response = ollama.chat(
            model=VISION_MODEL,
            messages=[{
                "role": "user",
                "content": prompt,
                "images": [image_data],
            }],
        )
        return response["message"]["content"]

    except Exception as e:
        print(f"  Warning: LLaVA description failed for {image_path}: {e}")
        return f"[Image at {os.path.basename(image_path)} — description failed]"


# ─────────────────────────────────────────────
# OCR FOR SCANNED PAGES
# ─────────────────────────────────────────────
def detect_scanned_pages(filepath: str) -> list[int]:
    """
    Detect pages that are scanned (image-only, no extractable text).
    Returns list of 1-indexed page numbers.
    """
    doc = fitz.open(filepath)
    scanned = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text().strip()

        # If page has almost no text but has images, it's likely scanned
        if len(text) < OCR_MIN_TEXT_LENGTH and page.get_images(full=True):
            scanned.append(page_num + 1)

    doc.close()
    return scanned


def ocr_page(filepath: str, page_num: int, dpi: int = 300) -> str:
    """
    Run OCR on a single PDF page using Tesseract.
    Renders the page as an image first, then extracts text.

    Args:
        filepath: path to PDF
        page_num: 0-indexed page number
        dpi: resolution for rendering (higher = better OCR, slower)
    """
    if not TESSERACT_AVAILABLE:
        return ""

    doc = fitz.open(filepath)
    page = doc[page_num]

    # Render page to image at specified DPI
    zoom = dpi / 72  # 72 is default PDF resolution
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat)

    # Convert to PIL Image
    img = Image.open(io.BytesIO(pix.tobytes("png")))

    # Run Tesseract OCR
    try:
        text = pytesseract.image_to_string(img)
        doc.close()
        return text.strip()
    except Exception as e:
        print(f"  Warning: OCR failed on page {page_num + 1}: {e}")
        doc.close()
        return ""


def ocr_scanned_pages(filepath: str) -> list[dict]:
    """
    Detect and OCR all scanned pages in a PDF.
    Returns list of {"text": ..., "page": N} for each scanned page.
    """
    scanned_pages = detect_scanned_pages(filepath)

    if not scanned_pages:
        return []

    if not TESSERACT_AVAILABLE:
        print("  Warning: Tesseract not available. Skipping OCR for scanned pages.")
        return []

    print(f"  Found {len(scanned_pages)} scanned page(s): {scanned_pages}")

    results = []
    for page_num in scanned_pages:
        print(f"  Running OCR on page {page_num}...")
        text = ocr_page(filepath, page_num - 1)  # 0-indexed for fitz

        if text and len(text) > OCR_MIN_TEXT_LENGTH:
            results.append({
                "text": text,
                "page": page_num,
            })

    print(f"  OCR extracted text from {len(results)} page(s)")
    return results


# ─────────────────────────────────────────────
# FULL IMAGE PIPELINE
# ─────────────────────────────────────────────
def process_pdf_images(filepath: str, page_texts: list[dict] = None) -> list[dict]:
    """
    Full pipeline: extract images + describe with LLaVA + OCR scanned pages.

    Args:
        filepath: path to the PDF
        page_texts: optional list of {"text": ..., "page": N} for context hints

    Returns:
        List of chunk dicts matching the parse_pdf() contract:
        [{"text": description, "metadata": {"source": ..., "page": N, "type": "image"|"ocr"}}]
    """
    filename = os.path.basename(filepath)
    all_chunks = []

    # 1. Extract and describe embedded images
    images = extract_images(filepath)

    if images:
        # Build page context lookup for better descriptions
        page_context = {}
        if page_texts:
            for pt in page_texts:
                page_context[pt["page"]] = pt["text"][:200]

        for img in images:
            context_hint = page_context.get(img["page"], "")

            print(f"  Describing image: {os.path.basename(img['filepath'])}...")
            description = describe_image(img["filepath"], context_hint)

            if description.startswith("[Image at"):
                continue

            all_chunks.append({
                "text": description,
                "metadata": {
                    "source": filename,
                    "page": img["page"],
                    "type": "image",
                    "image_path": img["filepath"],
                },
            })

    # 2. OCR scanned pages
    ocr_results = ocr_scanned_pages(filepath)

    for ocr_page_data in ocr_results:
        all_chunks.append({
            "text": ocr_page_data["text"],
            "metadata": {
                "source": filename,
                "page": ocr_page_data["page"],
                "type": "ocr",
            },
        })

    print(f"  Image chunks: {len([c for c in all_chunks if c['metadata']['type'] == 'image'])}")
    print(f"  OCR chunks: {len([c for c in all_chunks if c['metadata']['type'] == 'ocr'])}")
    print(f"  Total image/OCR chunks: {len(all_chunks)}")

    return all_chunks