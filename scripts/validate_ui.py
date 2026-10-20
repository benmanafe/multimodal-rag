"""Local UI smoke check; no model calls. Requires Chromium and websocket-client.

Run from the repository root: python scripts/validate_ui.py
Uses a temporary browser profile, synthetic images, and mocked API services.
"""
import base64
from contextlib import chdir
import itertools
import json
import os
from pathlib import Path
import runpy
import socket
import subprocess
import sys
import tempfile
import threading
import time
from unittest.mock import Mock, patch

import httpx
import uvicorn
import websocket

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
from app import main, image_paths
from app.config import Settings
from app.ingest import IngestionStats
from app.pdf_processor import extract_pdf_content


def validate():
    browser_path = next((path for path in [
        Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
        Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
    ] if path.is_file()), None)
    if browser_path is None:
        raise SystemExit("No supported local Chromium browser found.")
    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    checks = {}
    document_id = "a" * 64
    filename = "page_2_img_0001.png"
    image_url = f"/images/{document_id}/{filename}"
    missing_url = f"/images/{document_id}/page_2_img_0002.png"
    with tempfile.TemporaryDirectory(prefix="rag-ui-") as temporary, chdir(temporary):
        config = Settings(_env_file=None, image_dir=Path("data/images"), deepseek_api_key="mock-only")
        fixture = runpy.run_path(str(ROOT / "scripts/validate_vision.py"))
        fixture["create_synthetic_cv"](Path("synthetic_cv.pdf"))
        extracted = extract_pdf_content(Path("synthetic_cv.pdf"))
        picture = config.image_dir / document_id / filename
        picture.parent.mkdir(parents=True)
        picture.write_bytes(Path(extracted.images[0]["metadata"]["image_path"]).read_bytes())

        inventory = [{"id": identity * 64, "name": name, "indexed": indexed, "available": True,
                      "size_bytes": 1200, "text_chunks": 2 if indexed else 0, "image_descriptions": 1 if indexed else 0}
                     for identity, name, indexed in [("a", "sample_cv.pdf", True), ("b", "notes.txt", True), ("c", "pending.pdf", False)]]

        def answer(question, history, document_ids=None):
            if document_ids == ["b" * 64]:
                sources = [{"file": "notes.txt", "score": 0.9, "modality": "text"}]
            elif "Legacy" in question:
                sources = [{"file": "legacy.txt", "score": 0.61}]
            elif "missing" in question:
                sources = [{"file": "sample_cv.pdf", "page": 2, "modality": "image", "score": 0.7, "image_url": missing_url}]
            else:
                sources = [{"file": "sample_cv.pdf", "page": 1, "modality": "text", "score": 0.81, "image_url": None},
                           {"file": "sample_cv.pdf", "page": 2, "modality": "image", "score": 0.78, "image_url": image_url}]
            return {"answer": "The diagram is described as PDF to Text + Captions to Qdrant.", "sources": sources}

        answer_mock = Mock(side_effect=answer)
        stats = IngestionStats(3, 1, 4, [{"file": "sample_cv.pdf", "page": 2,
            "code": "empty_caption", "message": "One visual could not be described."}])
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        server = uvicorn.Server(uvicorn.Config(main.app, host="127.0.0.1", port=port, log_level="error"))
        browser, connection = None, None
        with patch.object(main, "settings", config), patch.object(image_paths, "settings", config), \
             patch.object(main, "answer_question", answer_mock), \
             patch.object(main, "document_inventory", return_value=inventory), \
             patch.object(main, "get_qdrant_client", return_value=Mock()), \
             patch.object(main, "ingest_directory_with_stats", return_value=stats), \
             patch("httpx.post", side_effect=AssertionError("Model/network requests forbidden")):
            thread = threading.Thread(target=server.run, daemon=True)
            thread.start()
            try:
                deadline = time.monotonic() + 15
                while not server.started and time.monotonic() < deadline:
                    time.sleep(0.05)
                assert server.started, "Local API failed to start"
                profile = Path(temporary) / "browser-profile"
                browser = subprocess.Popen([
                    str(browser_path), "--headless=new", "--disable-gpu", "--disable-background-networking",
                    "--no-first-run", "--no-default-browser-check", "--remote-debugging-port=0",
                    f"--user-data-dir={profile}", "about:blank",
                ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                active_port = profile / "DevToolsActivePort"
                deadline = time.monotonic() + 20
                while not active_port.exists() and time.monotonic() < deadline:
                    time.sleep(0.1)
                assert active_port.exists(), "Headless browser did not start"
                debug_port = int(active_port.read_text().splitlines()[0])
                with httpx.Client(trust_env=False) as client:
                    targets = client.get(f"http://127.0.0.1:{debug_port}/json").json()
                target = next(item for item in targets if item["type"] == "page")
                connection = websocket.create_connection(target["webSocketDebuggerUrl"], timeout=15, suppress_origin=True)
                serial = itertools.count(1)

                def command(method, params=None):
                    current = next(serial)
                    connection.send(json.dumps({"id": current, "method": method, "params": params or {}}))
                    while True:
                        result = json.loads(connection.recv())
                        if result.get("id") == current:
                            assert "error" not in result, result.get("error")
                            return result.get("result", {})

                def evaluate(expression):
                    result = command("Runtime.evaluate", {"expression": expression, "awaitPromise": True, "returnByValue": True})
                    assert "exceptionDetails" not in result, "Browser JavaScript raised an exception"
                    return result["result"].get("value")

                def wait_for(expression):
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        if evaluate(expression):
                            return
                        time.sleep(0.1)
                    raise AssertionError("Browser condition timed out")

                command("Emulation.setDeviceMetricsOverride", {"width": 1100, "height": 900, "deviceScaleFactor": 1, "mobile": False})
                command("Page.navigate", {"url": f"http://127.0.0.1:{port}/"})
                wait_for("document.readyState === 'complete' && typeof ask === 'function'")
                wait_for("documentsLoaded && document.querySelectorAll('.doc-row').length === 3")
                checks["document_inventory_visible"] = evaluate("document.querySelector('#document-list').textContent.includes('sample_cv.pdf') && document.querySelector('#document-list').textContent.includes('Not indexed')")
                checks["unindexed_documents_disabled"] = evaluate("document.querySelector('#document-list input[value=\"' + 'c'.repeat(64) + '\"]').disabled")
                evaluate("ask('Explain the diagram')")
                evaluate("document.querySelector('.source-image').scrollIntoView()")
                wait_for("document.querySelector('.source-image img')?.naturalWidth > 0")
                checks["original_image_loaded"] = evaluate("document.querySelector('.source-image img').naturalWidth > 100")
                checks["filename_page_modality_citations"] = evaluate("document.querySelector('.sources').textContent.includes('sample_cv.pdf · p. 2 · image') && document.querySelector('.sources').textContent.includes('p. 1 · text')")
                screenshot = command("Page.captureScreenshot", {"format": "png"})
                (reports / "phase5_ui.png").write_bytes(base64.b64decode(screenshot["data"]))
                command("Emulation.setDeviceMetricsOverride", {"width": 390, "height": 844, "deviceScaleFactor": 1, "mobile": True})
                wait_for("document.querySelector('aside').getBoundingClientRect().right <= 1")
                checks["mobile_sidebar_collapsed"] = True
                checks["mobile_no_horizontal_overflow"] = evaluate("document.documentElement.scrollWidth <= window.innerWidth && document.querySelector('#chat').scrollWidth <= document.querySelector('#chat').clientWidth")
                screenshot = command("Page.captureScreenshot", {"format": "png"})
                (reports / "phase5_ui_mobile.png").write_bytes(base64.b64decode(screenshot["data"]))
                evaluate("document.querySelector('#scope').click()")
                wait_for("document.querySelector('aside').getBoundingClientRect().left >= -1")
                checks["mobile_document_panel_accessible"] = evaluate("document.querySelector('#documents-panel').open && document.querySelector('#apply-documents').getBoundingClientRect().bottom < innerHeight")
                screenshot = command("Page.captureScreenshot", {"format": "png"})
                (reports / "documents_mobile.png").write_bytes(base64.b64decode(screenshot["data"]))
                evaluate("document.querySelector('#scrim').click()")
                wait_for("document.querySelector('aside').getBoundingClientRect().right <= 1")
                command("Emulation.setDeviceMetricsOverride", {"width": 1100, "height": 900, "deviceScaleFactor": 1, "mobile": False})
                evaluate("ask('Show the missing image')")
                wait_for("[...document.querySelectorAll('.image-fallback')].some(el => !el.hidden && el.textContent.includes('unavailable'))")
                checks["missing_image_fallback"] = True
                evaluate("ask('Legacy source')")
                checks["legacy_text_source_rendered"] = evaluate("document.querySelector('#chat').textContent.includes('legacy.txt · text')")
                evaluate("upload(new File(['test content'], 'notes.txt', {type: 'text/plain'}))")
                checks["upload_counts_and_warnings"] = evaluate("document.querySelector('.note.warning')?.textContent.includes('3 text, 1 image descriptions') && document.querySelector('.ingest-warnings')?.textContent.includes('sample_cv.pdf · p. 2: One visual')")
                checks["external_image_url_rejected"] = evaluate("sourceImageEl({image_url:'https://example.com/image.png'}, 'untrusted').querySelector('img') === null")
                checks["filenames_rendered_as_text"] = evaluate("messageEl({role:'bot',content:'Test',sources:[{file:'<img src=x onerror=alert(1)>',score:0.5}]}).querySelector('img') === null")
                checks["history_sent_to_api"] = [len(call.args[1]) for call in answer_mock.call_args_list] == [0, 2, 4]
                command("Page.reload")
                wait_for("document.readyState === 'complete' && typeof ask === 'function' && document.querySelector('.ingest-warnings') !== null")
                checks["chat_and_warnings_survive_reload"] = evaluate("document.querySelectorAll('.msg.bot').length === 3 && document.querySelector('.note.warning') !== null")
                checks["saved_history_excludes_notes"] = evaluate("historyFor(active()).length === 6")
                previous_session = evaluate("activeId")
                evaluate("document.querySelector('#all-documents').click(); document.querySelector('#document-list input[value=\"' + 'b'.repeat(64) + '\"]').click(); document.querySelector('#apply-documents').click()")
                checks["scope_change_starts_clean_chat"] = evaluate(f"activeId !== {json.dumps(previous_session)} && historyFor(active()).length === 0 && active().documentIds[0] === 'b'.repeat(64)")
                evaluate("ask('Answer only from notes')")
                checks["selected_scope_sent_to_api"] = answer_mock.call_args.kwargs.get("document_ids") == ["b" * 64] and answer_mock.call_args.args[1] == []
                checks["selected_scope_citation"] = evaluate("document.querySelector('.sources').textContent.includes('notes.txt') && document.querySelector('#scope').textContent.includes('1 file')")
                command("Page.reload")
                wait_for("document.readyState === 'complete' && typeof ask === 'function' && documentsLoaded")
                checks["selection_survives_reload"] = evaluate("active().documentIds?.[0] === 'b'.repeat(64) && document.querySelector('#document-list input[value=\"' + 'b'.repeat(64) + '\"]').checked && !document.querySelector('#all-documents').checked")
                evaluate("document.querySelector('#document-list input[value=\"' + 'b'.repeat(64) + '\"]').click()")
                checks["empty_selection_cannot_apply"] = evaluate("document.querySelector('#apply-documents').disabled && active().documentIds[0] === 'b'.repeat(64)")
                evaluate("renderDocuments(); openSession(" + json.dumps(previous_session) + ")")
                checks["scope_restored_per_chat"] = evaluate("active().documentIds == null && document.querySelector('#all-documents').checked")
                report = {"browser": browser_path.name + " headless", "deepseek_requests": 0,
                          "checks": checks, "passed": all(checks.values())}
                (reports / "phase5_ui_validation.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(report, indent=2))
                assert report["passed"], "UI smoke checks failed"
            finally:
                if connection:
                    try:
                        command("Browser.close")
                    except Exception:
                        pass
                    connection.close()
                if browser:
                    try:
                        browser.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        browser.kill()
                        browser.wait(timeout=5)
                server.should_exit = True
                thread.join(timeout=10)


if __name__ == "__main__":
    validate()
