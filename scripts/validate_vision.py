"""Opt-in, two-request caption smoke check using generated, non-sensitive evidence.

Run from the repository root: python scripts/validate_vision.py --live
Normal pytest runs never execute this script. No retry or other API calls occur.
"""
import argparse
from contextlib import chdir
import json
import os
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch
from urllib.parse import urlparse

import httpx
import pymupdf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def draw_pipeline(page, labels):
    for index, label in enumerate(labels):
        left = 40 + index * 180
        page.draw_rect(pymupdf.Rect(left, 150, left + 140, 230), color=(0.1, 0.2, 0.5))
        page.insert_text((left + 12, 194), label, fontsize=15)
        if index < len(labels) - 1:
            page.draw_line((left + 140, 190), (left + 180, 190))
            page.draw_line((left + 172, 184), (left + 180, 190))
            page.draw_line((left + 172, 196), (left + 180, 190))


def create_synthetic_cv(path):
    with pymupdf.open() as diagram:
        page = diagram.new_page(width=600, height=300)
        draw_pipeline(page, ["PDF", "Text + Captions", "Qdrant"])
        raster = page.get_pixmap(clip=pymupdf.Rect(20, 120, 580, 260), dpi=144).tobytes("png")
    with pymupdf.open() as pdf:
        first = pdf.new_page(width=600, height=600)
        first.insert_text((40, 60), "Synthetic CV: project architecture example.", fontsize=14)
        first.insert_image(pymupdf.Rect(20, 120, 580, 260), stream=raster)
        second = pdf.new_page(width=600, height=600)
        second.insert_text((40, 60), "Synthetic CV: project workflow example.", fontsize=14)
        draw_pipeline(second, ["Question", "Retrieval", "Answer"])
        pdf.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow up to two paid DeepSeek requests.")
    args = parser.parse_args()
    if not args.live:
        parser.error("Use --live to explicitly enable two caption requests.")
    os.chdir(PROJECT_ROOT)
    from app.config import settings
    from app.ingest import prepare_pdf_documents
    from app.image_processor import ImageDescriptionError

    host = urlparse(settings.vision_base_url or "https://api.deepseek.com")
    if settings.vision_provider != "deepseek" or host.scheme != "https" or host.hostname != "api.deepseek.com":
        parser.error("This smoke check requires the official HTTPS DeepSeek provider endpoint.")
    if not settings.deepseek_api_key.strip():
        parser.error("DEEPSEEK_API_KEY is missing; configure it in the ignored .env file.")

    report = {"provider": "deepseek", "requested_model": settings.vision_model,
              "request_limit": 2, "requests_attempted": 0, "responses": []}
    original_post = httpx.post

    def counted_post(url, **kwargs):
        if report["requests_attempted"] >= report["request_limit"]:
            raise ImageDescriptionError("live_request_limit", "Live validation request limit reached.")
        # If the first request failed, stop rather than repeat an invalid/expensive call.
        if report["responses"] and report["responses"][-1].get("status", 0) != 200:
            raise ImageDescriptionError("live_validation_stopped", "Stopped after the first failed live request.")
        report["requests_attempted"] += 1
        try:
            response = original_post(url, **kwargs)
        except httpx.RequestError:
            report["responses"].append({"status": None, "error": "network_or_timeout"})
            raise
        record = {"status": response.status_code}
        if response.status_code == 200:
            try:
                body = response.json()
                record["model"] = body.get("model")
                usage = body.get("usage", {})
                record["usage"] = {key: usage[key] for key in
                                   ["prompt_tokens", "completion_tokens", "total_tokens"] if key in usage}
            except ValueError:
                record["error"] = "invalid_json"
        report["responses"].append(record)
        return response

    try:
        with tempfile.TemporaryDirectory(prefix="rag-vision-check-") as temporary, chdir(temporary):
            source = Path("synthetic_cv.pdf")
            create_synthetic_cv(source)
            with patch("app.image_processor.httpx.post", side_effect=counted_post):
                prepared = prepare_pdf_documents(source)
            report["text_pages_preserved"] = len(prepared.text_documents)
            report["captions"] = [{"page": node.metadata["page"],
                                   "image_kind": node.metadata["image_kind"], "text": node.text}
                                  for node in prepared.image_nodes]
            report["warnings"] = prepared.warnings
            caption_by_page = {node.metadata["page"]: node.text.lower() for node in prepared.image_nodes}
            report["checks"] = {
                "two_text_pages": len(prepared.text_documents) == 2,
                "raster_labels": all(label in caption_by_page.get(1, "") for label in ["pdf", "qdrant"]),
                "vector_labels": all(label in caption_by_page.get(2, "") for label in ["question", "retriev", "answer"]),
                "no_warnings": not prepared.warnings,
            }
            report["passed"] = all(report["checks"].values())
    finally:
        os.chdir(PROJECT_ROOT)
        destination = PROJECT_ROOT / "reports" / "phase3_vision_validation.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        # Safe report contains only synthetic captions, HTTP statuses, usage, and counts.
        print(json.dumps(report, indent=2))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
