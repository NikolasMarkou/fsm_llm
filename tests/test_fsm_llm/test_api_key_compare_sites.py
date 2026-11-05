"""Differential for the three API-key compare sites (plan 5980fbb7, step 7).

``monitor/server.py`` (``_require_api_key`` for HTTP, ``_api_key_matches`` for
the WebSocket auth message) and ``agents/remote.py`` (``AgentServer.
_require_api_key``) must accept and refuse exactly the same presented keys, and
agree with a plain oracle (utf-8 / surrogateescape byte equality). The first
class holds before and after the move of the shared logic into ``security``;
the second class pins the new helpers.
"""

from __future__ import annotations

import types

import pytest

pytest.importorskip("fastapi")

from fastapi import HTTPException
from starlette.requests import Request

import fsm_llm.monitor.server as server_module
from fsm_llm.agents.remote import AgentServer

KEYS = ["s3cr3t", "a", "k" * 300, "\xe9", "s\xe9cret"]


def _request(headers: list[tuple[bytes, bytes]]) -> Request:
    return Request(scope={"type": "http", "headers": headers})


def _oracle_token(headers: list[tuple[bytes, bytes]]) -> str | None:
    """The documented rule: Bearer wins, else X-API-Key (latin-1 header decode)."""
    named = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in headers}
    auth = named.get("authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[len("bearer ") :].strip()
    return named.get("x-api-key")


def _accepts(token: str | None, key: str) -> bool:
    return token is not None and token.encode("utf-8", "surrogateescape") == key.encode(
        "utf-8", "surrogateescape"
    )


def _outcome(call) -> str:
    try:
        call()
    except HTTPException as exc:
        assert exc.status_code == 401
        return "reject"
    except Exception as exc:  # pragma: no cover - the point of the test
        return f"error:{type(exc).__name__}"
    return "accept"


def _header_sets(key: str) -> list[list[tuple[bytes, bytes]]]:
    raws: list[bytes] = [
        key.encode("utf-8"),  # utf-8 on the wire: equal only for an ASCII key
        key[:-1].encode("utf-8"),  # shorter
        (key + "x").encode("utf-8"),  # longer
        b"",  # empty
        key.upper().encode("utf-8"),
        b" " + key.encode("utf-8") + b" ",  # padded
        b"\x80\xff\xfe",  # undecodable as utf-8
    ]
    try:
        raws.append(key.encode("latin-1"))  # equal after latin-1 decoding
    except UnicodeEncodeError:
        pass
    sets: list[list[tuple[bytes, bytes]]] = [[]]  # no header at all
    for raw in raws:
        sets.append([(b"x-api-key", raw)])
        sets.append([(b"authorization", b"Bearer " + raw)])
        sets.append([(b"authorization", b"bearer " + raw)])
        sets.append([(b"authorization", b"Basic " + raw)])
        # a wrong bearer shadows a right x-api-key
        sets.append([(b"authorization", b"Bearer nope"), (b"x-api-key", raw)])
    sets.append([(b"authorization", b"Bearer ")])
    return sets


@pytest.fixture(autouse=True)
def _restore_key():
    saved = server_module._api_key
    yield
    server_module._api_key = saved


class TestThreeSitesAgree:
    @pytest.mark.parametrize("key", KEYS)
    def test_http_sites_match_each_other_and_the_oracle(self, key):
        stub = types.SimpleNamespace(_api_key=key)
        server_module._api_key = key
        checked = 0
        accepted = 0
        for headers in _header_sets(key):
            req = _request(headers)
            expected = "accept" if _accepts(_oracle_token(headers), key) else "reject"
            monitor = _outcome(lambda req=req: server_module._require_api_key(req))
            remote = _outcome(lambda req=req: AgentServer._require_api_key(stub, req))
            assert monitor == remote == expected, (key, headers, monitor, remote)
            checked += 1
            accepted += expected == "accept"
        assert checked > 40 and accepted >= 3

    @pytest.mark.parametrize("key", KEYS)
    def test_websocket_site_matches_the_oracle(self, key):
        server_module._api_key = key
        tokens = [
            key,
            key[:-1],
            key + "x",
            "",
            key.upper(),
            " " + key,
            "\udcff\udc80",  # a surrogateescape-decoded header byte pair
            key.encode("latin-1", "replace").decode("latin-1") + "Δ",
        ]
        for token in tokens:
            assert server_module._api_key_matches(token) is _accepts(token, key), token

    def test_websocket_site_without_a_key_is_open(self):
        server_module._api_key = None
        assert server_module._api_key_matches("anything") is True
        assert server_module._api_key_matches(None) is True

    def test_no_key_configured_is_open_on_both_http_sites(self):
        server_module._api_key = None
        stub = types.SimpleNamespace(_api_key=None)
        req = _request([])
        server_module._require_api_key(req)
        AgentServer._require_api_key(stub, req)

    def test_a_non_str_token_is_never_accepted(self):
        server_module._api_key = "s3cr3t"
        for token in (b"s3cr3t", 123, ["s3cr3t"]):
            try:
                result = server_module._api_key_matches(token)  # type: ignore[arg-type]
            except Exception:
                continue
            assert result is False

    @pytest.mark.parametrize("empty", ["", " ", "\t\n"])
    def test_an_empty_key_is_refused_by_both_servers(self, empty):
        with pytest.raises(ValueError, match="api_key"):
            server_module.configure(api_key=empty)
        with pytest.raises(ValueError, match="api_key"):
            AgentServer(agent=types.SimpleNamespace(), api_key=empty)

    def test_a_blank_env_key_still_means_no_auth(self, monkeypatch):
        monkeypatch.setenv("FSM_LLM_MONITOR_API_KEY", "")
        server_module.configure(api_key=None)
        assert server_module._api_key is None

    def test_cross_product_of_wrong_lengths_never_accepts(self):
        server_module._api_key = "s3cr3t"
        stub = types.SimpleNamespace(_api_key="s3cr3t")
        for n in range(0, 12):
            for raw in ("s3cr3t"[:n], "s3cr3t" + "t" * n):
                if raw == "s3cr3t":
                    continue
                req = _request([(b"x-api-key", raw.encode())])
                assert (
                    _outcome(lambda req=req: server_module._require_api_key(req))
                    == "reject"
                )
                assert (
                    _outcome(lambda req=req: AgentServer._require_api_key(stub, req))
                    == "reject"
                )


class TestSecurityHelpers:
    def test_presented_api_key_prefers_bearer_then_x_api_key(self):
        from fsm_llm.security import presented_api_key

        assert presented_api_key({"authorization": "Bearer abc "}) == "abc"
        assert presented_api_key({"authorization": "BEARER abc"}) == "abc"
        assert presented_api_key({"x-api-key": "xyz"}) == "xyz"
        assert presented_api_key({"authorization": "Bearer a", "x-api-key": "b"}) == "a"
        assert presented_api_key({"authorization": "Basic a"}) is None
        assert presented_api_key({}) is None

    def test_api_key_matches_is_constant_time_on_bytes(self, monkeypatch):
        import fsm_llm.security as security

        calls = []
        real = security.hmac.compare_digest

        def spy(a, b):
            calls.append((a, b))
            return real(a, b)

        monkeypatch.setattr(security.hmac, "compare_digest", spy)
        assert security.api_key_matches("s3cr3t", "s3cr3t") is True
        assert security.api_key_matches("s\xe9", "s\xe9") is True
        assert security.api_key_matches("s\xe9", "se") is False
        assert security.api_key_matches(None, "s3cr3t") is False
        assert security.api_key_matches(b"s3cr3t", "s3cr3t") is False  # type: ignore[arg-type]
        assert calls[0] == (b"s3cr3t", b"s3cr3t")
        assert all(isinstance(a, bytes) and isinstance(b, bytes) for a, b in calls)

    @pytest.mark.parametrize("empty", ["", "  ", "\t\n"])
    def test_checked_api_key_refuses_a_blank_key(self, empty):
        from fsm_llm.security import checked_api_key

        with pytest.raises(ValueError, match="api_key"):
            checked_api_key(empty)

    def test_checked_api_key_passes_none_and_a_real_key_through(self):
        from fsm_llm.security import checked_api_key

        assert checked_api_key(None) is None
        assert checked_api_key(" k ") == " k "
