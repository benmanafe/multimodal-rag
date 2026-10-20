def test_pdf_reader_is_installed():
    # Without llama-index-readers-file, LlamaIndex silently reads .pdf as raw bytes.
    from llama_index.readers.file import PDFReader  # noqa: F401  (ImportError = missing package)
