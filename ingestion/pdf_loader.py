from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(pdf_paths):
    documents = []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    for path in pdf_paths:
        reader = PdfReader(path)
        text = ""

        for page in reader.pages:
            text += page.extract_text() or ""

        chunks = splitter.split_text(text)
        documents.extend(chunks)

    return documents
