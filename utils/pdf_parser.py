# utils/pdf_parser.py

import fitz  # PyMuPDF
import requests
import tempfile


def extract_text_from_pdf(pdf_url, max_pages=10):
    try:
        response = requests.get(pdf_url, timeout=10)

        if response.status_code != 200:
            return None

        with tempfile.NamedTemporaryFile(delete=True, suffix=".pdf") as tmp:
            tmp.write(response.content)
            tmp.flush()

            doc = fitz.open(tmp.name)

            text = ""

            for page_num in range(min(max_pages, len(doc))):
                page = doc[page_num]
                text += page.get_text()

            return text

    except Exception as e:
        print("PDF extraction error:", e)
        return None