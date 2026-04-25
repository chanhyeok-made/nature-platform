"""Tests for token estimation."""

from nature.utils.tokens import (
    estimate_tokens_for_image,
    estimate_tokens_for_text,
    estimate_tokens_for_value,
)


class TestTokenEstimation:
    def test_empty_string(self):
        assert estimate_tokens_for_text("") == 1  # minimum 1

    def test_text_estimation(self):
        text = "Hello, world!"  # 13 bytes UTF-8
        tokens = estimate_tokens_for_text(text)
        assert tokens == 13 // 4  # 3 tokens at 4 bytes/token

    def test_json_estimation(self):
        text = '{"key": "value"}'  # 16 bytes
        tokens = estimate_tokens_for_text(text, is_json=True)
        assert tokens == 16 // 2  # 8 tokens at 2 bytes/token

    def test_unicode_text(self):
        text = "한국어 텍스트"  # 3 bytes per Korean char
        tokens = estimate_tokens_for_text(text)
        assert tokens > 0

    def test_value_none(self):
        assert estimate_tokens_for_value(None) == 0

    def test_value_dict(self):
        data = {"name": "test", "count": 42}
        tokens = estimate_tokens_for_value(data)
        assert tokens > 0

    def test_image_estimate(self):
        assert estimate_tokens_for_image() == 2000
