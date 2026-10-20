"""Real synthetic PDFs exercise extraction without services or private documents."""
from pathlib import Path
import uuid

import pymupdf
import pytest

from app import pdf_processor
from app.config import Settings
from app.pdf_processor import PdfProcessingError, extract_pdf_content


@pytest.fixture(autouse=True)
def extraction_settings(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(pdf_processor, "settings", Settings(
        _env_file=None, image_dir=Path("data/images"), pdf_min_image_pixels=64,
        pdf_vector_fallback=True, pdf_vector_min_paths=3, pdf_render_dpi=144,
    ))


def png(width=160, height=120, shade=160, alpha=False):
    pixmap = pymupdf.Pixmap(pymupdf.csRGB, pymupdf.IRect(0, 0, width, height), alpha)
    pixmap.clear_with(shade)
    if alpha:
        pixmap.set_alpha(bytes([128]) * (width * height))
    return pixmap.tobytes("png")


@pytest.fixture
def mixed_pdf(tmp_path):
    path = tmp_path / "resume.pdf"
    with pymupdf.open() as pdf:
        first = pdf.new_page()
        first.insert_text((72, 72), "Python libraries: FastAPI and Qdrant.")
        second = pdf.new_page()
        second.insert_text((72, 72), "Architecture: PDF feeds Qdrant.")
        second.insert_image(pymupdf.Rect(80, 180, 320, 360), stream=png())
        pdf.save(path)
    return path


def test_two_page_text_and_raster_have_correct_metadata(mixed_pdf):
    result = extract_pdf_content(mixed_pdf)
    assert [doc.metadata["page"] for doc in result.text_documents] == [1, 2]
    assert "FastAPI" in result.text_documents[0].text
    assert "Architecture" in result.text_documents[1].text
    assert all(doc.metadata["modality"] == "text" for doc in result.text_documents)
    assert len(result.images) == 1
    image = result.images[0]
    metadata = image["metadata"]
    assert metadata["file_name"] == "resume.pdf"
    assert metadata["page"] == 2
    assert metadata["modality"] == "image"
    assert metadata["image_kind"] == "raster"
    assert len(metadata["document_id"]) == 64
    assert all(doc.metadata["document_id"] == metadata["document_id"] for doc in result.text_documents)
    assert not Path(metadata["image_path"]).is_absolute()
    assert metadata["image_path"].startswith(f'data/images/{metadata["document_id"]}/page_2_')
    assert "\\" not in metadata["image_path"]
    assert pymupdf.Pixmap(metadata["image_path"]).width == 160
    assert "Architecture" in image["context"]
    assert uuid.UUID(image["id"]).version == 5
    assert result.warnings == []


def test_ids_and_paths_survive_reextraction_and_input_folder_migration(mixed_pdf, tmp_path):
    first = extract_pdf_content(mixed_pdf)
    folder = tmp_path / "documents"
    folder.mkdir()
    relocated = folder / mixed_pdf.name
    relocated.write_bytes(mixed_pdf.read_bytes())
    second = extract_pdf_content(relocated)
    assert first.images == second.images
    assert [doc.id_ for doc in first.text_documents] == [doc.id_ for doc in second.text_documents]
    assert len(list(Path("data/images").rglob("*.png"))) == 1
    renamed = folder / "different.pdf"
    renamed.write_bytes(relocated.read_bytes())
    third = extract_pdf_content(renamed)
    assert first.images[0]["id"] != third.images[0]["id"]
    with pymupdf.open(relocated) as pdf:
        pdf[0].insert_text((72, 100), "New content")
        changed = pdf.tobytes()
    relocated.write_bytes(changed)
    fourth = extract_pdf_content(relocated)
    assert first.images[0]["metadata"]["document_id"] != fourth.images[0]["metadata"]["document_id"]


def test_decorative_images_and_duplicate_logos_are_filtered(tmp_path):
    path = tmp_path / "logos.pdf"
    with pymupdf.open() as pdf:
        for number in range(2):
            page = pdf.new_page()
            page.insert_text((72, 72), f"Page {number + 1}")
            # Same-page duplicate, and a small recurring logo across pages.
            page.insert_image(pymupdf.Rect(30, 100, 78, 148), stream=png(80, 80, 100))
            page.insert_image(pymupdf.Rect(90, 100, 138, 148), stream=png(80, 80, 100))
            # Small source and high-resolution-but-tiny-on-page decoration.
            page.insert_image(pymupdf.Rect(30, 160, 60, 190), stream=png(16, 16, 20))
            page.insert_image(pymupdf.Rect(70, 160, 80, 170), stream=png(160, 160, 30))
            # Substantial repeated visual must retain each page's citation.
            page.insert_image(pymupdf.Rect(80, 250, 380, 475), stream=png())
        pdf.save(path)
    result = extract_pdf_content(path)
    assert len(result.images) == 3
    assert [img["metadata"]["page"] for img in result.images] == [1, 1, 2]


def test_transparency_is_preserved(tmp_path):
    path = tmp_path / "transparent.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_image(pymupdf.Rect(80, 180, 320, 360), stream=png(alpha=True))
        pdf.save(path)
    result = extract_pdf_content(path)
    saved = pymupdf.Pixmap(result.images[0]["metadata"]["image_path"])
    assert saved.alpha == 1
    assert saved.pixel(0, 0)[-1] == 128


@pytest.fixture
def vector_pdf(tmp_path):
    path = tmp_path / "vector_diagram.pdf"
    with pymupdf.open() as pdf:
        text_page = pdf.new_page()
        text_page.insert_text((72, 72), "This page has text only.")
        page = pdf.new_page()
        page.insert_text((72, 72), "Architecture diagram")
        page.draw_rect(pymupdf.Rect(80, 150, 180, 230))
        page.insert_text((100, 190), "PDF")
        page.draw_line((180, 190), (260, 190))
        page.draw_rect(pymupdf.Rect(260, 150, 360, 230))
        page.insert_text((275, 190), "Qdrant")
        pdf.save(path)
    return path


def test_vector_diagram_fallback_renders_only_a_cropped_visual(vector_pdf):
    result = extract_pdf_content(vector_pdf)
    assert len(result.images) == 1
    metadata = result.images[0]["metadata"]
    assert metadata["page"] == 2 and metadata["image_kind"] == "vector_crop"
    assert metadata["modality"] == "image"
    saved = pymupdf.Pixmap(metadata["image_path"])
    assert saved.width == 624 and saved.height == 224
    assert "Qdrant" in result.text_documents[1].text
    assert result.warnings == []


def test_vector_fallback_can_be_disabled(vector_pdf, monkeypatch):
    monkeypatch.setattr(pdf_processor.settings, "pdf_vector_fallback", False)
    result = extract_pdf_content(vector_pdf)
    assert result.images == []
    assert len(result.text_documents) == 2
    assert not Path("data/images").exists()


def test_rotated_page_crop_uses_the_correct_visual_coordinates(vector_pdf):
    with pymupdf.open(vector_pdf) as pdf:
        pdf[1].set_rotation(90)
        rotated = pdf.tobytes()
    vector_pdf.write_bytes(rotated)
    result = extract_pdf_content(vector_pdf)
    assert len(result.images) == 1
    saved = pymupdf.Pixmap(result.images[0]["metadata"]["image_path"])
    assert saved.width == 224 and saved.height == 624
    # The crop contains dark diagram strokes rather than a blank off-target region.
    assert min(saved.samples) < 100


def test_page_sized_vector_frame_is_not_rendered_automatically(tmp_path):
    path = tmp_path / "frame.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.draw_rect(pymupdf.Rect(10, 10, 585, 832))
        page.draw_rect(pymupdf.Rect(80, 150, 180, 230))
        page.draw_line((180, 190), (260, 190))
        page.draw_rect(pymupdf.Rect(260, 150, 360, 230))
        pdf.save(path)
    assert extract_pdf_content(path).images == []


def test_single_border_and_rule_do_not_trigger_vector_rendering(tmp_path):
    path = tmp_path / "decoration.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_text((72, 72), "Plain resume")
        page.draw_rect(pymupdf.Rect(40, 40, 550, 790))
        page.draw_line((72, 100), (450, 100))
        pdf.save(path)
    result = extract_pdf_content(path)
    assert result.images == []


def test_raster_and_vector_visuals_can_coexist_on_one_page(tmp_path):
    path = tmp_path / "mixed_visuals.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_image(pymupdf.Rect(80, 100, 320, 280), stream=png())
        page.draw_rect(pymupdf.Rect(80, 400, 180, 480))
        page.draw_line((180, 440), (260, 440))
        page.draw_rect(pymupdf.Rect(260, 400, 360, 480))
        pdf.save(path)
    result = extract_pdf_content(path)
    assert [img["metadata"]["image_kind"] for img in result.images] == ["raster", "vector_crop"]
    assert len({img["id"] for img in result.images}) == 2


def test_image_only_pdf_warns_about_missing_text_without_ocr(tmp_path):
    path = tmp_path / "scan.pdf"
    with pymupdf.open() as pdf:
        page = pdf.new_page()
        page.insert_image(page.rect, stream=png(800, 1000))
        pdf.save(path)
    result = extract_pdf_content(path)
    assert result.text_documents == []
    assert len(result.images) == 1
    assert result.warnings[0]["code"] == "no_extractable_text"
    assert result.warnings[0]["page"] == 1
    assert "OCR" in result.warnings[0]["message"]


@pytest.mark.parametrize("content", [b"", b"not a PDF", b"%PDF-1.7\ncorrupt"])
def test_malformed_pdf_has_a_clear_error(tmp_path, content):
    path = tmp_path / "broken.pdf"
    path.write_bytes(content)
    with pytest.raises(PdfProcessingError) as caught:
        extract_pdf_content(path)
    assert caught.value.code == "invalid_pdf"
    assert str(tmp_path) not in str(caught.value)


def test_password_protected_pdf_has_a_clear_error(tmp_path):
    path = tmp_path / "locked.pdf"
    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(path, encryption=pymupdf.PDF_ENCRYPT_AES_256, owner_pw="owner", user_pw="test")
    with pytest.raises(PdfProcessingError) as caught:
        extract_pdf_content(path)
    assert caught.value.code == "encrypted_pdf"


def test_missing_pdf_has_a_clear_error(tmp_path):
    with pytest.raises(PdfProcessingError) as caught:
        extract_pdf_content(tmp_path / "missing.pdf")
    assert caught.value.code == "pdf_read_failed"


def test_generated_paths_cannot_escape_application_root(mixed_pdf, tmp_path):
    outside = tmp_path.parent / "outside_images"
    with pytest.raises(PdfProcessingError) as caught:
        extract_pdf_content(mixed_pdf, outside)
    assert caught.value.code == "invalid_image_directory"
    assert not outside.exists()


def test_image_save_failure_preserves_text_and_produces_warning(mixed_pdf, monkeypatch):
    def fail_write(path, data):
        raise OSError("private filesystem detail")
    monkeypatch.setattr(Path, "write_bytes", fail_write)
    result = extract_pdf_content(mixed_pdf)
    assert len(result.text_documents) == 2
    assert result.images == []
    assert result.warnings[0]["code"] == "image_save_failed"
    assert result.warnings[0]["page"] == 2
    assert "private filesystem detail" not in result.warnings[0]["message"]
