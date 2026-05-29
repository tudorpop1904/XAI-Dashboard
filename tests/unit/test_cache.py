"""Tests for core.cache module."""

from core.cache import _make_key, clear_cache, get_cached, put_cached


class TestMakeKey:
    def test_deterministic(self):
        k1 = _make_key("model", [{"role": "user", "content": "hello"}])
        k2 = _make_key("model", [{"role": "user", "content": "hello"}])
        assert k1 == k2

    def test_different_inputs_different_keys(self):
        k1 = _make_key("model", [{"role": "user", "content": "hello"}])
        k2 = _make_key("model", [{"role": "user", "content": "world"}])
        assert k1 != k2

    def test_different_models_different_keys(self):
        k1 = _make_key("model_a", [{"role": "user", "content": "hello"}])
        k2 = _make_key("model_b", [{"role": "user", "content": "hello"}])
        assert k1 != k2


class TestCacheRoundtrip:
    def test_put_and_get(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.cache.CACHE_DIR", tmp_path / "cache")
        messages = [{"role": "user", "content": "test"}]
        response = {"message": {"content": "response"}}

        put_cached("model", messages, response)
        result = get_cached("model", messages)
        assert result is not None
        assert result["message"]["content"] == "response"

    def test_get_miss(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.cache.CACHE_DIR", tmp_path / "cache")
        result = get_cached("model", [{"role": "user", "content": "missing"}])
        assert result is None

    def test_clear_cache(self, tmp_path, monkeypatch):
        monkeypatch.setattr("core.cache.CACHE_DIR", tmp_path / "cache")
        put_cached("model", [{"content": "a"}], {"resp": 1})
        put_cached("model", [{"content": "b"}], {"resp": 2})
        count = clear_cache()
        assert count == 2
        assert get_cached("model", [{"content": "a"}]) is None
