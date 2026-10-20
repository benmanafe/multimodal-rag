"""Opt-in real-service acceptance check, capped at three DeepSeek requests.

Run from the app root with --live. Use --write-fixture PATH for a free demo PDF.
The live check uses a temporary input/image directory and its own Qdrant collection.
"""
import argparse
from contextlib import chdir
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from urllib.parse import urlparse
import uuid

import httpx
import pymupdf

sys.path.insert(0, str(Path.cwd()))


def make_fixture(destination: Path):
    with pymupdf.open() as diagram:
        page = diagram.new_page(width=600, height=280)
        for index, text in enumerate(["PDF", "Text + Captions", "Qdrant"]):
            left = 40 + index * 180
            page.draw_rect(pymupdf.Rect(left, 100, left + 140, 180))
            page.insert_text((left + 12, 145), text, fontsize=15)
            if index < 2:
                page.draw_line((left + 140, 140), (left + 180, 140))
        picture = page.get_pixmap(clip=pymupdf.Rect(20, 80, 580, 200), dpi=144).tobytes("png")
    with pymupdf.open() as pdf:
        page = pdf.new_page(width=600, height=600)
        page.insert_text((40, 60), "Synthetic CV - demonstration only", fontsize=16)
        page.insert_text((40, 100), "Libraries: FastAPI and Qdrant.", fontsize=14)
        page.insert_text((40, 130), "Project: document question answering.", fontsize=14)
        page = pdf.new_page(width=600, height=600)
        page.insert_text((40, 60), "Project architecture diagram", fontsize=16)
        page.insert_image(pymupdf.Rect(20, 100, 580, 220), stream=picture)
        pdf.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--write-fixture", type=Path)
    args = parser.parse_args()
    if args.write_fixture:
        make_fixture(args.write_fixture)
        print("Created non-sensitive demo PDF; zero API calls.")
        return 0
    if not args.live:
        parser.error("Specify --live to authorize at most three paid requests.")

    from fastapi.testclient import TestClient
    from app import main as api, query
    from app.config import settings
    from app.ingest import get_embed_model, get_qdrant_client

    root = Path.cwd()
    report = {"request_limit": 3, "deepseek_requests": 0, "responses": [], "checks": {}}
    destination = root / "reports/phase6_e2e_validation.json"
    destination.parent.mkdir(exist_ok=True)
    configured = urlparse(settings.vision_base_url or "https://api.deepseek.com")
    if settings.vision_provider != "deepseek" or configured.scheme != "https" or configured.hostname != "api.deepseek.com":
        parser.error("The check requires the official DeepSeek vision endpoint.")
    if not settings.deepseek_api_key.strip():
        parser.error("Configure DEEPSEEK_API_KEY in .env or the container environment.")

    original_collection, original_images = settings.qdrant_collection, settings.image_dir
    collection = "phase6_validation_" + uuid.uuid4().hex
    vector_client = None
    original_send = httpx.Client.send
    original_llm = query.DeepSeek

    def limited_llm(*positional, **keywords):
        # Real DeepSeek adapter; only bound output and disable SDK retries for this run.
        return original_llm(*positional, **keywords, max_tokens=512, max_retries=0)

    def counted_send(client, request, **keywords):
        if request.url.host != "api.deepseek.com":
            return original_send(client, request, **keywords)
        if report["deepseek_requests"] >= 3:
            raise RuntimeError("Three-request live validation limit reached")
        report["deepseek_requests"] += 1
        body = json.loads(request.content)
        kind = "caption" if any(isinstance(message.get("content"), list)
                                for message in body.get("messages", [])) else "answer"
        response = original_send(client, request, **keywords)
        response.read()
        record = {"kind": kind, "http_status": response.status_code}
        if response.status_code == 200:
            payload = response.json()
            record["model"] = payload.get("model")
            record["finish_reason"] = payload.get("choices", [{}])[0].get("finish_reason")
            usage = payload.get("usage", {})
            record["usage"] = {key: usage[key] for key in
                               ["prompt_tokens", "completion_tokens", "total_tokens"] if key in usage}
        report["responses"].append(record)
        return response

    try:
        # Free readiness checks come before any paid operation.
        vector_client = get_qdrant_client()
        vector_client.get_collections()
        vector = get_embed_model().get_text_embedding("Local readiness check.")
        report["embedding_dimension"] = len(vector)
        settings.qdrant_collection = collection
        (root / "data").mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="phase6-", dir=root / "data") as temporary:
            temporary = Path(temporary)
            settings.image_dir = temporary / "images"
            source = temporary / "synthetic_cv.pdf"
            make_fixture(source)
            data = source.read_bytes()
            with patch.object(api, "DATA_DIR", temporary / "inputs"), \
                 patch.object(query, "DeepSeek", side_effect=limited_llm), \
                 patch.object(httpx.Client, "send", new=counted_send), TestClient(api.app) as client:
                report["checks"]["health"] = client.get("/health").status_code == 200
                upload = client.post("/documents", files={"file": (source.name, data, "application/pdf")})
                report["upload_status"] = upload.status_code
                if upload.status_code != 200:
                    raise RuntimeError("Live upload failed")
                counts = upload.json()
                report["counts"] = counts
                if counts["warnings"] or counts["image_descriptions"] != 1:
                    raise RuntimeError("Live caption preparation did not produce one successful caption")
                before = vector_client.scroll(collection, limit=100, with_vectors=True)[0]
                report["checks"]["both_modalities_in_one_collection"] = {point.payload["modality"] for point in before} == {"text", "image"}
                report["checks"]["matching_vector_dimensions"] = all(
                    len(vector) == report["embedding_dimension"] for point in before
                    for vector in (point.vector.values() if isinstance(point.vector, dict) else [point.vector])
                )
                question = ("Answer these three numbered questions using only this CV: "
                            "1. Which libraries are listed? "
                            "2. What labels and flow appear in the architecture diagram? "
                            "3. Explain how the listed libraries and diagram relate to the project. "
                            "Distinguish image-description observations from extracted text and cite pages.")
                answer = client.post("/query", json={"question": question})
                report["query_status"] = answer.status_code
                if answer.status_code != 200:
                    raise RuntimeError("Live answer failed")
                result = answer.json()
                report["question"] = question
                report["answer"] = result
                text = result["answer"].lower()
                report["checks"]["text_and_diagram_answer_labels"] = all(label in text for label in ["fastapi", "qdrant", "pdf"])
                report["checks"]["mixed_source_citations"] = {item["modality"] for item in result["sources"]} == {"text", "image"}
                images = [item for item in result["sources"] if item["modality"] == "image"]
                if not images:
                    raise RuntimeError("No image source was retrieved")
                original = client.get(images[0]["image_url"])
                report["checks"]["original_image_served"] = original.status_code == 200 and original.content.startswith(b"\x89PNG")
                (destination.parent / "phase6_source.png").write_bytes(original.content)
                reupload = client.post("/documents", files={"file": (source.name, data, "application/pdf")})
                if reupload.status_code != 200:
                    raise RuntimeError("Live re-ingestion failed")
                after = vector_client.scroll(collection, limit=100)[0]
                report["checks"]["stable_rebuild_no_duplicates"] = (
                    {point.id for point in before} == {point.id for point in after}
                    and counts["total_vectors"] == reupload.json()["total_vectors"]
                    and not reupload.json()["warnings"]
                )
                report["checks"]["image_pointer_survives_rebuild"] = client.get(images[0]["image_url"]).status_code == 200
                report["checks"]["api_budget"] = report["deepseek_requests"] == 3
                report["passed"] = all(report["checks"].values())
    except Exception as exc:
        report["passed"] = False
        report["failure_type"] = type(exc).__name__
    finally:
        if vector_client is not None:
            try:
                if vector_client.collection_exists(collection):
                    vector_client.delete_collection(collection)
            except Exception as exc:
                report["cleanup_failure_type"] = type(exc).__name__
                report["passed"] = False
            finally:
                vector_client.close()
        settings.qdrant_collection, settings.image_dir = original_collection, original_images
        report["total_tokens"] = sum(item.get("usage", {}).get("total_tokens", 0) for item in report["responses"])
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
