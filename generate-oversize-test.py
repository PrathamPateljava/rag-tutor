"""
Generate an oversized PDF (>30MB) to test file size rejection.
Run: python generate_oversized_pdf.py
"""

import os

os.makedirs("data/raw_pdfs", exist_ok=True)
path = "data/raw_pdfs/oversized_test.pdf"

# Create a ~31MB dummy file with a .pdf extension
target_size = 31 * 1024 * 1024  # 31MB
with open(path, "wb") as f:
    f.write(b"%PDF-1.4\n")  # minimal PDF header
    f.write(b"A" * target_size)

size_mb = os.path.getsize(path) / (1024 * 1024)
print(f"Created: {path} ({size_mb:.1f}MB)")
print(f"\nNow run: python main.py ingest")
print(f"Expected: REJECTED — exceeds 30MB limit")