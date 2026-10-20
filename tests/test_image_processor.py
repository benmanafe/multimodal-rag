"""Caption boundary tests; all HTTP requests are intercepted without a model/API key."""
import base64
import json

import httpx
import pytest

from app import image_processor
from app.config import Settings
from app.image_processor import ImageDescriptionError, describe_image


@pytest.fixture
def image(tmp_path):
    # Valid one-pixel PNG; the adapter should transmit its original bytes unchanged.
    path = tmp_path / "diagram.png"
    path.write_bytes(base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aN1sAAAAASUVORK5CYII="
    ))
    return path


@pytest.fixture(autouse=True)
def mock_vision_settings(monkeypatch):
    monkeypatch.setattr(image_processor, "settings", Settings(_env_file=None,
        vision_provider="ollama", vision_model="gemma3:4b", vision_base_url="",
        ollama_base_url="http://embedding-server:11434", vision_timeout=42,
    ))


def intercept_http(monkeypatch, handler):
    client = httpx.Client(transport=httpx.MockTransport(handler))
    monkeypatch.setattr(image_processor.httpx, "post", client.post)
    return client


def use_deepseek(monkeypatch):
    monkeypatch.setattr(image_processor.settings, "vision_provider", "deepseek")
    monkeypatch.setattr(image_processor.settings, "vision_model", "deepseek-flash")
    monkeypatch.setattr(image_processor.settings, "deepseek_api_key", "test-key-not-a-secret")


@pytest.mark.parametrize("vision_url,expected_url", [
    ("", "http://embedding-server:11434/api/chat"),
    ("http://vision-server:11434/", "http://vision-server:11434/api/chat"),
])
def test_caption_sends_image_context_and_configured_model(image, monkeypatch, vision_url, expected_url):
    monkeypatch.setattr(image_processor.settings, "vision_base_url", vision_url)
    def handle(request):
        assert str(request.url) == expected_url
        body = json.loads(request.content)
        assert body["model"] == "gemma3:4b"
        assert body["stream"] is False
        assert body["options"]["temperature"] == 0
        assert request.extensions["timeout"]["read"] == 42
        system, user = body["messages"]
        assert system["role"] == "system"
        assert "unreadable or uncertain" in system["content"]
        assert user["role"] == "user"
        assert "Project Atlas" in user["content"]
        assert base64.b64decode(user["images"][0]) == image.read_bytes()
        return httpx.Response(200, json={"done": True, "message": {"content": "  PDF feeds Qdrant.\n"}})
    with intercept_http(monkeypatch, handle):
        assert describe_image(image, "Project Atlas") == "PDF feeds Qdrant."


@pytest.mark.parametrize("failure", [
    httpx.ConnectError("private server information"),
    httpx.ReadTimeout("private server information"),
    404,
    500,
    401,
    429,
])
@pytest.mark.parametrize("provider", ["ollama", "deepseek"])
def test_service_failures_are_recoverable_and_sanitized(image, monkeypatch, failure, provider):
    if provider == "deepseek":
        use_deepseek(monkeypatch)
    def handle(request):
        if isinstance(failure, Exception):
            raise failure
        return httpx.Response(failure, text="private server information")
    with intercept_http(monkeypatch, handle), pytest.raises(ImageDescriptionError) as caught:
        describe_image(image)
    assert caught.value.code == "caption_request_failed"
    assert "private server information" not in str(caught.value)
    assert "test-key-not-a-secret" not in str(caught.value)


@pytest.mark.parametrize("body,code", [
    ({"done": True, "message": {"content": " \n"}}, "empty_caption"),
    ({"done": True, "message": {"content": None}}, "invalid_caption_response"),
    ({"done": True, "message": None}, "invalid_caption_response"),
    ({"done": False, "message": {"content": "partial"}}, "invalid_caption_response"),
    ({"error": "private error"}, "invalid_caption_response"),
    ([], "invalid_caption_response"),
])
def test_empty_and_malformed_captions_raise_explicit_errors(image, monkeypatch, body, code):
    with intercept_http(monkeypatch, lambda request: httpx.Response(200, json=body)):
        with pytest.raises(ImageDescriptionError) as caught:
            describe_image(image)
    assert caught.value.code == code


@pytest.mark.parametrize("vision_url,expected_url", [
    ("", "https://api.deepseek.com/chat/completions"),
    ("https://api.deepseek.com/v1/", "https://api.deepseek.com/v1/chat/completions"),
])
def test_deepseek_caption_uses_existing_key_and_inline_image(image, monkeypatch, vision_url, expected_url):
    use_deepseek(monkeypatch)
    monkeypatch.setattr(image_processor.settings, "vision_base_url", vision_url)
    def handle(request):
        assert str(request.url) == expected_url
        assert request.headers["authorization"] == "Bearer test-key-not-a-secret"
        assert request.extensions["timeout"]["read"] == 42
        body = json.loads(request.content)
        assert body["model"] == "deepseek-flash"
        assert body["stream"] is False
        assert body["thinking"] == {"type": "disabled"}
        assert body["max_tokens"] == 512
        system, user = body["messages"]
        assert system["role"] == "system"
        assert user["role"] == "user"
        context, visual = user["content"]
        assert context["type"] == "text" and "Atlas" in context["text"]
        assert visual["type"] == "image_url"
        data_url = visual["image_url"]["url"]
        prefix, encoded = data_url.split(",", 1)
        assert prefix == "data:image/png;base64"
        assert base64.b64decode(encoded) == image.read_bytes()
        return httpx.Response(200, json={"choices": [{
            "finish_reason": "stop", "message": {"content": " PDF feeds Qdrant.\n"},
        }]})
    with intercept_http(monkeypatch, handle):
        assert describe_image(image, "Atlas") == "PDF feeds Qdrant."


@pytest.mark.parametrize("body,code", [
    ({"choices": [{"finish_reason": "stop", "message": {"content": " \n"}}]}, "empty_caption"),
    ({"choices": [{"finish_reason": "length", "message": {"content": "partial"}}]}, "invalid_caption_response"),
    ({"choices": [{"finish_reason": "content_filter", "message": {"content": "refused"}}]}, "invalid_caption_response"),
    ({"choices": [{"finish_reason": "stop", "message": {"content": None}}]}, "invalid_caption_response"),
    ({"choices": []}, "invalid_caption_response"),
    ({"error": "private error"}, "invalid_caption_response"),
    (None, "invalid_caption_response"),
])
def test_deepseek_rejects_invalid_or_incomplete_captions(image, monkeypatch, body, code):
    use_deepseek(monkeypatch)
    with intercept_http(monkeypatch, lambda request: httpx.Response(200, json=body)):
        with pytest.raises(ImageDescriptionError) as caught:
            describe_image(image)
    assert caught.value.code == code


@pytest.mark.parametrize("case,code", [
    ("key", "vision_not_configured"),
    ("format", "unsupported_image_format"),
    ("size", "image_too_large"),
])
def test_deepseek_rejects_missing_key_and_unsupported_images_before_http(image, monkeypatch, case, code):
    use_deepseek(monkeypatch)
    if case == "key":
        monkeypatch.setattr(image_processor.settings, "deepseek_api_key", " ")
    elif case == "format":
        image.write_bytes(b"not an image")
    else:
        image.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\0" * (32 * 1024 * 1024))
    def unexpected_request(request):
        pytest.fail("Invalid configuration/input must not reach DeepSeek")
    with intercept_http(monkeypatch, unexpected_request), pytest.raises(ImageDescriptionError) as caught:
        describe_image(image)
    assert caught.value.code == code


def test_non_json_response_is_a_caption_error(image, monkeypatch):
    with intercept_http(monkeypatch, lambda request: httpx.Response(200, text="not json")):
        with pytest.raises(ImageDescriptionError, match="invalid or incomplete"):
            describe_image(image)


@pytest.mark.parametrize("case,code", [
    ("missing", "image_read_failed"),
    ("empty", "image_read_failed"),
    ("provider", "unsupported_vision_provider"),
    ("model", "vision_not_configured"),
])
def test_invalid_inputs_fail_before_any_network_request(image, monkeypatch, case, code):
    if case == "missing":
        image.unlink()
    elif case == "empty":
        image.write_bytes(b"")
    elif case == "provider":
        monkeypatch.setattr(image_processor.settings, "vision_provider", "unsupported")
    else:
        monkeypatch.setattr(image_processor.settings, "vision_model", " ")
    def unexpected_request(request):
        pytest.fail("Invalid input must not reach the vision server")
    with intercept_http(monkeypatch, unexpected_request), pytest.raises(ImageDescriptionError) as caught:
        describe_image(image)
    assert caught.value.code == code
