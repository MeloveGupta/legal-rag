import logging
from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

def load_pdf(file_path: str | Path) -> List[Document]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")
    
    logger.info(f"Loading PDF: {file_path.name}")
    loader = PyPDFLoader(str(file_path))
    docs = loader.load()

    total_pages = len(docs)
    for doc in docs:
        doc.metadata.update({
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "pdf",
            "total_pages": total_pages,
        })

    logger.info(f"Loaded {total_pages} pages from {file_path.name}")
    return docs

def load_markdown(file_path: str | Path) -> List[Document]:
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Markdown file not found: {file_path}")
    logger.info(f"Loading Markdown: {file_path.name}")
    text = file_path.read_text(encoding="utf-8")

    doc = Document(
        page_content=text,
        metadata={
            "source": str(file_path),
            "file_name": file_path.name,
            "file_type": "markdown",
            "page": 1,
            "total_pages": 1,
        }
    )

    logger.info(f"Loaded {len(text)} characters from {file_path.name}")
    return [doc]

def load_web_page(url: str) -> List[Document]:
    logger.info(f"Loading web page: {url}")
    loader = WebBaseLoader(url)
    docs = loader.load()

    for doc in docs:
        doc.metadata.update({
            "source": url,
            "file_type": "web",
            "page": 1,
            "total_pages": 1,
        })

    logger.info(f"Loaded {len(docs)} document(s) from {url}")
    return docs

def load_documents(source: str | Path) -> List[Document]:
    source_str = str(source)

    if source_str.startswith("http://") or source_str.startswith("https://"):
        return load_web_page(source_str)
    
    path = Path(source)
    extension = path.suffix.lower()

    if extension == ".pdf":
        return load_pdf(path)
    elif extension in (".md", ".markdown"):
        return load_markdown(path)
    else:
        raise ValueError(
            f"Unsupported file type: '{extension}'. "
            f"Supported: .pdf, .md, .markdown, or a https:// URL."
        )