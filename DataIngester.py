import os
from typing import List
import pdfplumber
import markdown
from bs4 import BeautifulSoup


class DataIngester:
    def __init__(self, filepaths: List[str]):
        self.filepaths = filepaths

    def read_file(self, filepath: str) -> str:
        ext = os.path.splitext(filepath)[1]

        if ext == ".pdf":
            with pdfplumber.open(filepath) as pdf:
                return " ".join(page.extract_text() for page in pdf.pages)

        elif ext == ".md":
            with open(filepath, "r") as f:
                md = f.read()
                html = markdown.markdown(md)
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text()

        elif ext == ".html":
            with open(filepath, "r") as f:
                html = f.read()
                soup = BeautifulSoup(html, "html.parser")
                return soup.get_text()

        elif ext == ".txt":
            with open(filepath, "r") as f:
                return f.read()

        else:
            raise ValueError(f"Unsupported file extension: {ext}")

    def ingest(self) -> List[str]:
        documents = []

        for filepath in self.filepaths:
            document = self.read_file(filepath)
            documents.append(document)

        return documents
