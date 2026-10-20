"""Extract page text and visual evidence; captioning and indexing happen elsewhere."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import uuid

import pymupdf
from llama_index.core import Document

from app.config import settings


class PdfProcessingError(Exception):
    """A document-level failure that the ingestion/API layer can report clearly."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


@dataclass
class PdfContent:
    text_documents: list[Document] = field(default_factory=list)
    images: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)


def get_document_id(source_path: Path, content: bytes) -> str:
    """Stable across directory migrations; changing filename/content changes identity."""
    return hashlib.sha256(source_path.name.encode("utf-8") + b"\0" + content).hexdigest()


def make_record_id(document_id: str, page: int, modality: str, ordinal: int) -> str:
    name = f"multimodal-rag:{document_id}:p{page}:{modality}:{ordinal}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, name))


def _image_destination(images_dir: Path, document_id: str, filename: str) -> tuple[Path, str]:
    app_root = Path.cwd().resolve()
    image_root = images_dir.resolve()
    destination = (image_root / document_id / filename).resolve()
    try:
        image_root.relative_to(app_root)
        destination.relative_to(image_root)
        relative_path = destination.relative_to(app_root).as_posix()
    except ValueError:
        raise PdfProcessingError(
            "invalid_image_directory", "Generated images must stay within the application image directory."
        ) from None
    return destination, relative_path


def _raster_bytes(block: dict) -> tuple[bytes, str]:
    """Preserve supported embedded bytes; restore alpha or normalize other formats."""
    extension = block["ext"].lower()
    mask = block.get("mask")
    if not mask and extension in {"png", "jpeg", "jpg", "gif", "webp"}:
        return block["image"], extension
    pixmap = pymupdf.Pixmap(block["image"])
    if pixmap.colorspace and pixmap.colorspace.n > 3:
        pixmap = pymupdf.Pixmap(pymupdf.csRGB, pixmap)
    if mask:
        if pixmap.alpha:
            pixmap = pymupdf.Pixmap(pixmap, 0)
        pixmap = pymupdf.Pixmap(pixmap, pymupdf.Pixmap(mask))
    return pixmap.tobytes("png"), "png"


def _vector_regions(page: pymupdf.Page, drawings: list[dict]) -> list[pymupdf.Rect]:
    """Heuristic: connected multi-path regions, excluding single rules/borders."""
    if not settings.pdf_vector_fallback or len(drawings) < settings.pdf_vector_min_paths:
        return []
    page_bounds = page.rect * page.derotation_matrix
    regions = []
    for cluster in page.cluster_drawings(
        drawings=drawings, clip=page_bounds, x_tolerance=12, y_tolerance=12,
    ):
        region = pymupdf.Rect(cluster)
        # A single line can have zero area, so count contact using expanded bounds.
        count = sum(
            (pymupdf.Rect(path["rect"]) + (-1, -1, 1, 1)).intersects(region)
            for path in drawings
        )
        if count < settings.pdf_vector_min_paths or region.width < 96 or region.height < 48:
            continue
        fraction = region.get_area() / page_bounds.get_area()
        if not 0.01 <= fraction <= 0.8:
            continue
        regions.append((region + (-16, -16, 16, 16)) & page_bounds)
    return sorted(regions, key=lambda rect: (rect.y0, rect.x0, rect.y1, rect.x1))


def extract_pdf_content(pdf_path: Path, images_dir: Path | None = None) -> PdfContent:
    """Return unsplit text pages, uncaptained image candidates, and safe warnings.

    Run from the application root. Image metadata uses app-relative POSIX paths;
    image candidates carry page context separately from their eventual caption.
    Text extraction is not OCR. Rendered vector crops preserve source appearance
    but are derived evidence, identified by image_kind rather than raster originals.
    """
    pdf_path = Path(pdf_path)
    images_dir = Path(images_dir) if images_dir is not None else settings.image_dir
    try:
        source_bytes = pdf_path.read_bytes()
    except OSError:
        raise PdfProcessingError("pdf_read_failed", "Could not read the PDF document.") from None
    document_id = get_document_id(pdf_path, source_bytes)
    # Validate the output root even when this PDF has no images.
    _image_destination(images_dir, document_id, "validation.png")
    try:
        pdf = pymupdf.open(stream=source_bytes, filetype="pdf")
    except (pymupdf.FileDataError, RuntimeError, ValueError):
        raise PdfProcessingError("invalid_pdf", "Could not parse the PDF document.") from None

    result = PdfContent()
    seen_rasters: set[str] = set()

    def warn(code: str, page_number: int, message: str, record_id: str | None = None):
        warning = {"code": code, "file": pdf_path.name, "page": page_number, "message": message}
        if record_id:
            warning["record_id"] = record_id
        result.warnings.append(warning)

    def save_image(image_bytes: bytes, extension: str, page_number: int,
                   ordinal: int, kind: str, page_text: str):
        record_id = make_record_id(document_id, page_number, "image", ordinal)
        filename = f"page_{page_number}_img_{ordinal:04d}.{extension}"
        destination, relative_path = _image_destination(images_dir, document_id, filename)
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(image_bytes)
        except OSError:
            warn("image_save_failed", page_number, "Could not save extracted image.", record_id)
            return
        result.images.append({
            "id": record_id,
            "metadata": {
                "file_name": pdf_path.name, "document_id": document_id,
                "page": page_number, "modality": "image", "image_path": relative_path,
                "image_kind": kind,
            },
            "context": page_text[:2000],
        })

    with pdf:
        if pdf.needs_pass:
            raise PdfProcessingError("encrypted_pdf", "Password-protected PDFs are not supported.")
        if not pdf.page_count:
            raise PdfProcessingError("invalid_pdf", "The PDF document has no pages.")
        for page_number, page in enumerate(pdf, start=1):
            try:
                text = page.get_text("text", sort=True).strip()
                blocks = [block for block in page.get_text("dict")["blocks"] if block["type"] == 1]
                drawings = page.get_drawings() if settings.pdf_vector_fallback else []
            except (RuntimeError, ValueError):
                raise PdfProcessingError("invalid_pdf", "Could not extract PDF page content.") from None
            if text:
                result.text_documents.append(Document(
                    id_=make_record_id(document_id, page_number, "page_text", 1),
                    text=text,
                    metadata={"file_name": pdf_path.name, "document_id": document_id,
                              "page": page_number, "modality": "text"},
                    excluded_embed_metadata_keys=["document_id"],
                    excluded_llm_metadata_keys=["document_id"],
                ))
            elif blocks or drawings:
                warn("no_extractable_text", page_number,
                     "This visual page has no extractable text; scanned text is not OCR-processed.")

            page_digests: set[str] = set()
            page_bounds = page.rect * page.derotation_matrix
            for ordinal, block in enumerate(blocks, start=1):
                visible = pymupdf.Rect(block["bbox"]) & page_bounds
                if min(block["width"], block["height"]) < settings.pdf_min_image_pixels:
                    continue
                if visible.is_empty or min(visible.width, visible.height) < 24:
                    continue
                digest = hashlib.sha256(block["image"] + (block.get("mask") or b"")).hexdigest()
                small_repeat = digest in seen_rasters and visible.get_area() <= page.rect.get_area() * 0.05
                if digest in page_digests or small_repeat:
                    continue
                try:
                    image_bytes, extension = _raster_bytes(block)
                except (RuntimeError, ValueError):
                    warn("image_extraction_failed", page_number, "Could not decode an embedded image.",
                         make_record_id(document_id, page_number, "image", ordinal))
                    continue
                save_image(image_bytes, extension, page_number, ordinal, "raster", text)
                page_digests.add(digest)
                seen_rasters.add(digest)

            try:
                regions = _vector_regions(page, drawings)
            except (RuntimeError, ValueError):
                warn("vector_extraction_failed", page_number, "Could not locate vector diagram regions.")
                continue
            for index, region in enumerate(regions, start=1):
                ordinal = len(blocks) + index
                try:
                    # Cap resolution for unusually large PDF pages as well as normal CVs.
                    scale = min(settings.pdf_render_dpi / 72, 2048 / max(region.width, region.height))
                    image_bytes = page.get_pixmap(
                        matrix=pymupdf.Matrix(scale, scale), clip=region * page.rotation_matrix,
                        colorspace=pymupdf.csRGB, alpha=False, annots=False,
                    ).tobytes("png")
                except (RuntimeError, ValueError):
                    warn("vector_render_failed", page_number, "Could not render a vector diagram.")
                    continue
                save_image(image_bytes, "png", page_number, ordinal, "vector_crop", text)
    return result
