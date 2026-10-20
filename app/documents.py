"""Read-only document inventory; inspecting sources never invokes a model."""
from pathlib import Path
import json
import re

from app.config import settings
from app.ingest import SUPPORTED_EXTS
from app.pdf_processor import get_document_id


def document_inventory(data_dir: Path, client, *, include_node_ids: bool = False) -> list[dict]:
    documents = {}
    root = Path(data_dir)
    if root.is_symlink():
        return []
    folders = [root]
    uploads = root / "documents"
    if uploads.is_dir() and not uploads.is_symlink():
        folders.append(uploads)
    for folder in folders:
        if not folder.is_dir() or folder.is_symlink():
            continue
        for path in sorted(folder.iterdir()):
            if path.is_symlink() or not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTS:
                continue
            identity = get_document_id(path, path.read_bytes())
            documents[identity] = {"id": identity, "name": path.name, "size_bytes": path.stat().st_size,
                                   "available": True, "indexed": False, "text_chunks": 0,
                                   "image_descriptions": 0}

    if client.collection_exists(settings.qdrant_collection):
        offset = None
        while True:
            points, offset = client.scroll(
                collection_name=settings.qdrant_collection, limit=256, offset=offset,
                with_payload=["document_id", "file_name", "modality", "_node_content"], with_vectors=False,
            )
            for point in points:
                metadata = point.payload or {}
                identity = metadata.get("document_id")
                # LlamaIndex reserves/overwrites the top-level document_id field.
                # The original source identity remains in serialized node metadata.
                try:
                    original = json.loads(metadata.get("_node_content", "{}"))
                    identity = original.get("metadata", {}).get("document_id", identity)
                except (ValueError, TypeError, AttributeError):
                    pass
                if not isinstance(identity, str) or not re.fullmatch(r"[a-f0-9]{64}", identity):
                    continue  # Older indexes without document IDs still work in all-files mode.
                entry = documents.setdefault(identity, {
                    "id": identity, "name": Path(str(metadata.get("file_name", "Unknown")).replace("\\", "/")).name,
                    "size_bytes": None, "available": False, "indexed": False,
                    "text_chunks": 0, "image_descriptions": 0,
                })
                entry["indexed"] = True
                if include_node_ids:
                    entry.setdefault("node_ids", []).append(point.id)
                entry["image_descriptions" if metadata.get("modality") == "image" else "text_chunks"] += 1
            if offset is None:
                break
    return sorted(documents.values(), key=lambda item: (item["name"].casefold(), item["id"]))
