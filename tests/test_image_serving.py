"""Real image-serving checks against a temporary, isolated image root."""
import base64
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from app import image_paths
from app.config import Settings
from app.main import app


@pytest.fixture
def image_file(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(image_paths, "settings", Settings(_env_file=None, image_dir=Path("data/images")))
    path = Path("data/images") / ("a" * 64) / "page_2_img_0001.png"
    path.parent.mkdir(parents=True)
    path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN1sAAAAASUVORK5CYII="
    ))
    return path


def test_generated_image_is_served_read_only_with_original_bytes(image_file):
    client = TestClient(app)
    url = image_paths.image_url_from_path(image_file.as_posix())
    response = client.get(url)
    assert response.status_code == 200 and response.content == image_file.read_bytes()
    assert response.headers["content-type"] == "image/png"
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["cache-control"] == "no-store"
    assert client.post(url).status_code == 405
    assert client.delete(url).status_code == 405


@pytest.mark.parametrize("path", [
    "/images/../.env", "/images/%2e%2e/%2eenv", "/images/" + "a" * 64 + "/%2e%2e%2f%2eenv",
    "/images/" + "a" * 64 + "/..%5c..%5c.env", "/images/" + "a" * 64 + "/.env",
    "/images/not-a-document/page_2_img_0001.png", "/images/" + "a" * 64 + "/page_0_img_0001.png",
    "/images/" + "a" * 64 + "/page_2_img_0001.svg",
])
def test_traversal_private_files_and_unrecognized_names_are_rejected(image_file, path):
    Path(".env").write_text("private test sentinel")
    response = TestClient(app).get(path)
    assert response.status_code == 404
    assert b"private test sentinel" not in response.content


def test_missing_image_returns_404_and_preserves_safe_url_for_ui_fallback(image_file):
    url = image_paths.image_url_from_path(image_file.as_posix())
    image_file.unlink()
    assert image_paths.image_url_from_path(image_file.as_posix()) == url
    response = TestClient(app).get(url)
    assert response.status_code == 404 and response.json()["detail"] == "Source image not found"


def test_misconfigured_external_image_root_cannot_serve_files(image_file, tmp_path, monkeypatch):
    monkeypatch.setattr(image_paths.settings, "image_dir", tmp_path.parent)
    assert image_paths.image_url_from_path(image_file.as_posix()) is None
    assert TestClient(app).get("/images/" + "a" * 64 + "/page_2_img_0001.png").status_code == 404


def test_custom_image_root_is_used_for_url_derivation_and_serving(image_file, monkeypatch):
    new_root = Path("derived/visuals")
    destination = new_root / image_file.parent.name / image_file.name
    destination.parent.mkdir(parents=True)
    image_file.rename(destination)
    monkeypatch.setattr(image_paths.settings, "image_dir", new_root)
    url = image_paths.image_url_from_path(destination.as_posix())
    assert url == "/images/" + "a" * 64 + "/page_2_img_0001.png"
    assert TestClient(app).get(url).content == destination.read_bytes()
    assert image_paths.image_url_from_path(image_file.as_posix()) is None


def test_symlink_image_is_rejected(image_file, monkeypatch):
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == image_file.resolve() or original(path))
    assert TestClient(app).get("/images/" + "a" * 64 + "/page_2_img_0001.png").status_code == 404
