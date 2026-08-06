"""Generate a one-page PDF with no text layer at all - stands in for a
scanned document (a photograph or scan of a page, saved as a PDF, with no
embedded text for PyMuPDF to extract).

Why this exists: indexing a document that yields zero chunks was never
exercised end to end. Reading Chroma's own validation code showed that
`collection.upsert()` rejects an empty ids list outright - and neither
app.py nor server.py caught that before it was found. Both now raise a
clear, actionable error instead. See BUILD_LOG.md, "the empty-chunk bug".

    python tools\\make_blank_pdf.py
"""

from pathlib import Path

import fitz

OUT = Path(__file__).parent.parent / "data" / "blank_scanned.pdf"


def build():
    OUT.parent.mkdir(exist_ok=True)
    doc = fitz.open()
    doc.new_page()  # a page with no text objects on it at all
    doc.save(str(OUT))
    doc.close()
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    build()
