"""Shared validation for generated image URLs and read-only file serving."""
from pathlib import Path, PurePosixPath
import re

from app.config import settings


def generated_image_path(document_id: str, filename: str) -> Path | None:
    """Accept only our generated names inside a dedicated application image root."""
    if not re.fullmatch(r"[a-f0-9]{64}", document_id) or not re.fullmatch(
        r"page_[1-9][0-9]*_img_[0-9]{4,}\.(png|jpeg|jpg|gif|webp)", filename,
    ):
        return None
    app_root = Path.cwd().resolve()
    configured_root = settings.image_dir
    root = configured_root.resolve()
    if root == app_root or not root.is_relative_to(app_root) or configured_root.is_symlink():
        return None
    folder = root / document_id
    candidate = folder / filename
    if folder.is_symlink() or candidate.is_symlink():
        return None
    resolved = candidate.resolve()
    return resolved if resolved.is_relative_to(root) else None


def image_url_from_path(relative_path: object) -> str | None:
    """Derive a URL from validated stored metadata, without exposing disk paths."""
    if not isinstance(relative_path, str) or "\\" in relative_path:
        return None
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != relative_path:
        return None
    if len(path.parts) < 3:
        return None
    document_id, filename = path.parts[-2:]
    candidate = generated_image_path(document_id, filename)
    if candidate is None or candidate != (Path.cwd() / relative_path).resolve():
        return None
    return f"/images/{document_id}/{filename}"
