"""Shared prompt machinery: the history and token helpers, the base config and the
base builder (text sanitizer, security filter, context and history sections).

Layer ``context`` (plan d454fd62, step 74): split out of the old ``fsm_llm.prompts``.
"""

import html
import json
import re
import textwrap
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar

from fsm_llm.constants import (
    DEFAULT_MAX_HISTORY_SIZE,
    INTERNAL_KEY_PREFIXES,
    MAX_CONTEXT_FILTER_DEPTH,
    MAX_CONTEXT_FILTER_NODES,
    has_internal_prefix,
    is_forbidden_context_entry,
)
from fsm_llm.context.filtering import redact_non_json_leaf
from fsm_llm.core.definitions import FSMInstance
from fsm_llm.logging import logger

# Sentinel: a context container past MAX_CONTEXT_FILTER_DEPTH, dropped by the
# caller. Walker-local by design (identity is never compared across modules);
# the DEPTH BOUND itself is the shared value, and it lives in constants.py.
_TOO_DEEP = object()


# Sentinel: the walk's node budget (MAX_CONTEXT_FILTER_NODES) is spent.
_OVER_BUDGET = object()


# Per-line cap for the compact history blocks (field-extraction prompt and the
# classifier's <classification_context>), applied after sanitization.
_HISTORY_LINE_CHARS = 150


def _cap_history_line(text: str) -> str:
    """Cap one already-sanitized history line at ``_HISTORY_LINE_CHARS``.

    Contract: takes the sanitized text; returns it unchanged when short,
    else its first ``_HISTORY_LINE_CHARS`` characters plus ``"..."``. Never
    raises. Shared by the field-extraction history and
    ``build_classification_context_block`` so the two caps cannot drift.
    """
    if len(text) > _HISTORY_LINE_CHARS:
        return text[:_HISTORY_LINE_CHARS] + "..."
    return text


def _history_messages(
    exchanges: Iterable[Mapping[str, Any]],
) -> Iterator[tuple[str, Any]]:
    """The ``(role, message)`` pairs of ``exchanges``, one per message, in order.

    Contract: the one role rule of the three history renderers (Pass-2 history,
    field-extraction history, classifier context). ``role`` is ``"user"`` when
    the key, lower-cased, is ``user``, else ``"system"``; any other key than
    ``system`` is logged at WARNING and treated as ``system``. An entry with
    several keys yields one pair per key (none is lost), and a message is
    passed through untouched (an empty one included; the caller decides).
    """
    for exchange in exchanges:
        for role, message in exchange.items():
            normalised = str(role).lower()
            if normalised == "user":
                yield "user", message
                continue
            if normalised != "system":
                logger.warning(
                    f"Unknown role '{role}' in conversation history, treating as system"
                )
            yield "system", message


class HistoryManagementStrategy(str, Enum):
    """Strategy for managing conversation history."""

    MESSAGE_COUNT = "message_count"  # Limit by number of messages
    TOKEN_BUDGET = "token_budget"  # Limit by estimated token count
    HYBRID = "hybrid"  # Use both limits (whichever is hit first)


@dataclass(frozen=True)
class BasePromptConfig:
    """Base configuration shared between all prompt builders."""

    # History Management
    max_history_messages: int = DEFAULT_MAX_HISTORY_SIZE
    max_token_budget: int = 3000
    history_strategy: HistoryManagementStrategy = HistoryManagementStrategy.HYBRID

    # Context Management
    # Deliberately high: this bounds a pathological tail (runaway context
    # growth), it does not shape the happy path. Real FSMs carry tens of keys.
    max_context_keys: int = 200

    # Token Estimation
    chars_per_token: float = 2.5  # Conservative estimate
    token_estimation_factor: float = 1.3  # Safety factor for overhead
    utf8_expansion_factor: float = 1.5  # UTF-8 multi-byte expansion
    cdata_overhead_tokens: int = 50  # Overhead for CDATA wrapping

    # Content inclusion
    include_conversation_history: bool = True
    include_persona: bool = True

    # Security
    filter_internal_context: bool = True
    internal_key_prefixes: list[str] = field(
        default_factory=lambda: list(INTERNAL_KEY_PREFIXES)
    )

    # Development/Debug Options
    deterministic_output: bool = True  # Sort for consistent testing
    verbose_logging: bool = False

    def __post_init__(self):
        """Validate configuration values."""
        if self.max_history_messages < 0:
            raise ValueError("max_history_messages must be non-negative")
        if self.max_context_keys < 0:
            raise ValueError("max_context_keys must be non-negative")
        if self.max_token_budget < 100:
            raise ValueError("max_token_budget must be at least 100")
        if self.chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")


class BasePromptBuilder:
    """Base class with shared functionality for all prompt builders."""

    # DECISION plan-2026-07-18T162030-a02151fe/D-017 [STALE]: ALLOW-list, not a deny-list.
    # Do NOT reintroduce a `_CRITICAL_TAGS` enumeration of structural tags to
    # escape. That design was a live injection bypass: the denylist and the tags
    # prompts.py actually emits were two hand-maintained lists, and they drifted —
    # `previously_extracted`/`still_missing` were emitted by build_refinement_prompt
    # but were never added to the 48-entry denylist, so a user message carrying
    # them reached the built prompt unescaped and indistinguishable from the
    # framework's own wrappers. Escaping every tag EXCEPT a tiny known-inert set
    # means a tag later added to prompts.py is covered without editing a list.
    # Widening _SAFE_TAGS re-opens that hole for the tag added — each addition
    # needs a seam test in test_prompt_injection_seam.py proving it is not
    # structural.
    #
    # SCOPE CORRECTION (D-024): "covered automatically" is a claim about the
    # ALLOW-LIST, not about the regex. What is escaped is still exactly what
    # _TAG_PATTERN matches, and that is narrower than "any tag-like text" —
    # attributes are not inspected, so `<b onclick="...">` passes through on the
    # strength of its tag name alone.
    _SAFE_TAGS: ClassVar[frozenset[str]] = frozenset({"b", "i"})
    # DECISION plan-2026-07-18T162030-a02151fe/D-024 [STALE]
    # `\s*` after `</?` is load-bearing — do NOT drop it as noise. Without it the
    # pattern required a letter IMMEDIATELY after `<`, so `< task>` and
    # `<   persona>` were left unescaped while `<task>` was escaped. See D-024.
    #
    # DECISION plan-2026-07-18T162030-a02151fe/D-026 [STALE]
    # BOTH `\s*` around the optional slash are load-bearing. D-024 placed one only
    # AFTER the slash, which closed the OPENER form and left the CLOSING form
    # wide open: `< /task>` was unmatched and reached the prompt raw, and
    # `<\n/task>` was MANUFACTURED into a live `< /task>` by the newline strip
    # below — the exact defect D-024 claims to have fixed, surviving for closers.
    # Closing the framework's own `<task>`/`<response_generation>` wrapper early
    # is a stronger injection primitive than opening a new tag, so do NOT drop
    # the leading `\s*` as symmetry noise.
    #
    # The slash is grouped as `(?:\s*/)?` and NOT written as the more obvious
    # `<\s*/?\s*`. Do not "simplify" it back: the two forms accept exactly the
    # same language (verified exhaustively over every string of length <=5 over
    # `<>/ \t ab1.-`, 111110 cases, zero divergence), but `\s*/?\s*` puts two
    # ambiguous whitespace runs side by side, so a non-matching `<` followed by
    # a long run of spaces backtracks quadratically — measured 3229ms vs 1.8ms
    # on the same input. This regex runs on user-controlled message text, so
    # that is a denial-of-service shape, not just a slow path. See D-026.
    #
    # DECISION plan-2026-09-19T175721-21cd7f8e/D-047
    # The tail is an ALTERNATION with two arms, and both are load-bearing:
    #   `[^>]{0,256}/?>`  a tag closed within 256 characters (nested `<` allowed,
    #                     so `</original_input <b>` is one escaped match);
    #   `(?=[^>]{257})`   an OVERFLOWING tail seen by a ZERO-WIDTH lookahead, so
    #                     a padded closer (`</task` + 300 characters + `>`) is
    #                     still escaped, and only the `<` plus its name is
    #                     matched, never the 257 characters after it.
    # AMENDMENT (final review concern 1): the overflow arm first CONSUMED those
    # 257 characters (`[^>]{257}`), so up to 260 characters of benign prose that
    # merely held `<` and a letter, with no `>` anywhere, were HTML-escaped and
    # cut off mid-sentence. Do NOT consume the tail again. Separately, a `<` and
    # whitespace before a name with no `/` (`latency < threshold`) is a
    # comparison, and an overflow-only match of that shape is kept raw in
    # `_sanitize_text_for_prompt` ONLY when no `>` follows it anywhere in the
    # text (D-054); a closer (`< /task`) is never exempt.
    # The bound exists because an unbounded `[^>]*` scans to end-of-text for
    # every `<a` of `<a<a<a...` (quadratic on user text, 10k chars 2.5 s per
    # prompt, D-029). Do NOT go back to `[^<>]*` (iteration 3: a closer with a
    # nested `<` matched only its inner `<b>` and reached Pass 2 raw), do NOT use
    # the bounded arm alone (a padded closer regresses), and do NOT drop the
    # D-038 guard in the substitution below: it stays load-bearing for a safe
    # tag whose tail holds a second `<`. See D-047 in decisions.md.
    #
    # DECISION plan-2026-09-21T203800-8a03483a/D-040
    # Two widenings (audit D9). (1) A name may also start with `_`, `!` or `?`,
    # so `<_task>`, `<!-- -->`, `<![CDATA[` and `<?xml ?>` are escaped; digits
    # still may not (`<3`, `<1s` are prose, and no XML name starts with one).
    # (2) A THIRD, zero-width arm `(?=[^>]{0,256}\Z)` matches a `<name` with no
    # `>` after it and fewer than 257 characters left, which the two arms above
    # never matched; the substitution escapes it only when it is a CLOSER
    # (`bye </task` must not meet the builder's own `</original_input>`) and
    # keeps an opener raw exactly as before (`x<y`, `a < b`). Do NOT make this
    # arm consuming or unbounded (`[^>]*$` is quadratic on `</a</a...`), and do
    # NOT keep closers raw here. See decisions.md D-040 (differential results).
    _TAG_PATTERN = re.compile(
        r"<(?:\s*/)?\s*([A-Za-z_!?][A-Za-z0-9._:-]*)"
        r"(?:[^>]{0,256}/?>|(?=[^>]{257})|(?=[^>]{0,256}\Z))"
    )

    def __init__(self, config: BasePromptConfig | None = None):
        """Initialize with configuration."""
        self.config = config or BasePromptConfig()
        if self.config.verbose_logging:
            logger.debug(
                f"{self.__class__.__name__} initialized with config: {self.config}"
            )

    # ========================================================================
    # TEXT SANITIZATION AND SECURITY (Enhanced from old version)
    # ========================================================================

    def _sanitize_text_for_prompt(self, text: str | None) -> str:
        """
        Sanitize text to prevent XML tag confusion in prompts.

        Escapes every XML-like tag except the inert formatting tags in
        ``_SAFE_TAGS``, and strips newlines to prevent prompt injection.
        See the D-017 anchor on ``_SAFE_TAGS``: this is an allow-list by design.
        """
        if text is None:
            return ""

        # DECISION plan-2026-07-18T162030-a02151fe/D-024 [STALE]
        # ORDER IS LOAD-BEARING. Newline stripping must run BEFORE escaping, not
        # after. Reversed (as it was), `<\ntask>x` did not match the tag pattern,
        # was left unescaped, and the newline strip THEN turned it into the live
        # `< task>x`. The sanitizer manufactured the payload it exists to
        # neutralise. Do not move this below the sub. See D-024.
        flat = text.replace("\n", " ").replace("\r", " ")

        # DECISION plan-2026-09-19T175721-21cd7f8e/D-054
        # `last` is the offset of the final `>` in the text, computed ONCE (an
        # O(n) scan per call, not per match). The comparison exemption below
        # applies only to a match that ends AFTER it, i.e. with NO `>` anywhere
        # later in the text. Do NOT loosen it back to "no `>` inside the match":
        # an overflow-only match never contains one, so `< name` + 257 padding
        # characters + `>` (a padded OPENER, the same attacker-controlled shape
        # D-024 exists for) reached the prompt raw, and every earlier pattern
        # (ed6cffd, d7f1679) escaped it. Do NOT write it as a slice
        # (`">" not in flat[m.end():]`): that is O(n) per match and quadratic on
        # `"< " * n`; `m.end() > last` is output-identical to the slice form
        # (checked exhaustively over every string of length <= 6).
        last = flat.rfind(">")

        return self._TAG_PATTERN.sub(
            lambda m: (
                m.group(0)
                if (
                    # D-047/D-054: an overflow-only match holds no `>`; with a
                    # space after `<`, no `/`, and no `>` anywhere after it,
                    # it is a comparison (`latency < threshold`), not a tag.
                    # D-040: an end-arm OPENER stays raw when the old pattern
                    # could not match it at all, i.e. fewer than 257 characters
                    # follow the FIRST name character (the overflow arm used to
                    # backtrack the name to reach 257; that case still escapes).
                    ">" not in m.group(0)
                    and "/" not in m.group(0)
                    and m.end() > last
                    and (m.group(0)[1:2].isspace() or len(flat) - m.start(1) <= 257)
                )
                or (
                    m.group(1).lower() in self._SAFE_TAGS
                    # DECISION plan-2026-09-19T175721-21cd7f8e/D-038
                    # A whitelisted tag is kept raw ONLY if its text holds no
                    # second `<`. Before this guard `<b </task>` matched as ONE
                    # safe `<b>` tag (the attribute tail swallowed the nested `<`)
                    # and a hostile `</task>` reached the prompt unescaped. The
                    # pattern is bounded `[^>]{0,256}` (D-047), which still
                    # swallows a nested `<`, so this guard is load-bearing again.
                    # Do NOT replace it with a parser (rewrite of a hot filter).
                    and "<" not in m.group(0)[1:]
                )
                else html.escape(m.group(0))
            ),
            flat,
        )

    def _escape_cdata(self, text: str) -> str:
        """Escape CDATA end sequences and XML closing tags to prevent breaking CDATA/XML blocks."""
        # Escape CDATA end sequences (including doubled variants like ]]]]>)
        text = text.replace("]]>", "]]&gt;")
        # Escape XML closing tags that could break our prompt structure
        text = text.replace("</", "&lt;/")
        return text

    def _dump_json(self, obj: Any, what: str, **dumps_kwargs: Any) -> str | None:
        """``json.dumps(obj, default=str, **dumps_kwargs)``, or ``None`` with a WARNING.

        Interface contract (every prompt section that embeds a JSON dump):
        ``what`` names the section in the warning. A value that cannot be dumped
        (mixed-type keys under ``sort_keys``, an int past the digit limit) drops
        that one section instead of failing the prompt; this never raises for
        those errors.
        """
        try:
            return json.dumps(obj, default=str, **dumps_kwargs)
        except (TypeError, ValueError, OverflowError) as e:
            logger.warning(f"Failed to serialize {what}: {e}")
            return None

    def _json_cdata_block(self, tag: str, obj: Any, what: str) -> list[str]:
        """``<tag><![CDATA[ pretty JSON of obj ]]></tag>`` plus a blank line, as lines.

        Interface contract: the dump goes through ``_escape_cdata`` so it
        cannot close the block or the prompt's own tags. Returns ``[]`` when
        ``obj`` cannot be dumped (see ``_dump_json``). The caller has already
        applied the security filter and the key cap.
        """
        dumped = self._dump_json(obj, what, indent=1, separators=(",", ": "))
        if dumped is None:
            return []
        return [
            f"<{tag}><![CDATA[",
            self._escape_cdata(dumped),
            f"]]></{tag}>",
            "",
        ]

    # ========================================================================
    # HISTORY MANAGEMENT (From old version)
    # ========================================================================

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count using conservative estimates."""
        char_count = len(text)
        adjusted_count = (
            char_count
            * self.config.utf8_expansion_factor
            * self.config.token_estimation_factor
        )
        return int(adjusted_count / self.config.chars_per_token)

    def _limit_history_by_message_count(
        self, exchanges: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Limit history by maximum number of messages."""
        if self.config.max_history_messages <= 0:
            return []
        if len(exchanges) <= self.config.max_history_messages:
            return exchanges

        result = exchanges[-self.config.max_history_messages :]
        if self.config.verbose_logging:
            logger.debug(
                f"Limited history by message count: {len(exchanges)} -> {len(result)} exchanges"
            )
        return result

    def _limit_history_by_token_budget(
        self, exchanges: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Limit history by estimated token budget."""
        if not exchanges:
            return exchanges

        available_tokens = max(
            self.config.max_token_budget - self.config.cdata_overhead_tokens, 1
        )
        result: list[dict[str, str]] = []
        current_tokens = 0

        for exchange in reversed(exchanges):
            exchange_json = json.dumps(exchange, separators=(",", ": "))
            exchange_tokens = self._estimate_token_count(exchange_json)

            if current_tokens + exchange_tokens > available_tokens and result:
                break

            result.insert(0, exchange)
            current_tokens += exchange_tokens

        if len(result) < len(exchanges) and self.config.verbose_logging:
            logger.debug(
                f"Limited history by token budget: {len(exchanges)} -> {len(result)} exchanges"
            )

        return result

    def _manage_conversation_history(
        self, exchanges: list[dict[str, str]]
    ) -> list[dict[str, str]]:
        """Apply the configured history management strategy."""
        if not exchanges:
            return exchanges

        if self.config.history_strategy == HistoryManagementStrategy.MESSAGE_COUNT:
            return self._limit_history_by_message_count(exchanges)
        elif self.config.history_strategy == HistoryManagementStrategy.TOKEN_BUDGET:
            return self._limit_history_by_token_budget(exchanges)
        elif self.config.history_strategy == HistoryManagementStrategy.HYBRID:
            by_count = self._limit_history_by_message_count(exchanges)
            return self._limit_history_by_token_budget(by_count)
        else:
            logger.warning(f"Unknown history strategy: {self.config.history_strategy}")
            return exchanges

    # DECISION plan-2026-07-18T162030-a02151fe/D-020 [STALE]
    # Do NOT replace this with a token-budget cap mirroring
    # `_limit_history_by_token_budget`. A token budget makes the set of context
    # keys the model sees depend on the byte size of their VALUES, so one long
    # value silently evicts unrelated keys. History tolerates that (dropping an
    # old exchange loses recall); context does not (dropping a key the current
    # state lists in `required_context_keys` breaks transition evaluation).
    # Count-based capping keeps eviction independent of value size. See D-020.
    def _limit_context_by_key_count(
        self, context_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Limit context to the most recently written keys.

        Mirrors ``_limit_history_by_message_count``: dicts preserve insertion
        order, so the tail is the most recently written state -- the same
        "keep the newest" rule history already uses.
        """
        if self.config.max_context_keys <= 0:
            return {}
        if len(context_data) <= self.config.max_context_keys:
            return context_data

        kept = list(context_data)[-self.config.max_context_keys :]
        result = {key: context_data[key] for key in kept}
        if self.config.verbose_logging:
            logger.debug(
                f"Limited context by key count: {len(context_data)} -> {len(result)} keys"
            )
        return result

    # DECISION plan-2026-07-19T191147-4b664252/D-011 [STALE]
    # This filter is the LAST line of defense before context text reaches the
    # LLM. `clean_context_keys` does NOT cover it: that function has exactly one
    # first-party callers (`engine/turn/`, with strip_forbidden_keys=False) and is
    # not on the path for `initial_context` or `API.update_context` writes --
    # for those two entry points nothing else filters before this. So it MUST
    # recurse: while it only walked `context_data.items()`, a nested
    # `{"user": {"password": "hunter2"}}` was measured going into the prompt
    # INTACT. Do NOT "simplify" the recursion back to a flat loop, and do NOT
    # drop the depth bound (fail-CLOSED at the limit -- see D-010).
    def _filter_context_for_security(
        self, context_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter context data for security purposes, at every nesting level.

        Removes internal keys (prefixed with _, system_, etc.) and keys
        matching forbidden security patterns (password, secret, token, api_key),
        recursing into nested dicts and into dicts inside lists/tuples so
        ``{"user": {"password": "x"}}`` is filtered like its flat equivalent.
        Recursion is bounded at ``MAX_CONTEXT_FILTER_DEPTH``; a container
        deeper than that is dropped rather than passed through unfiltered.
        JSON-native scalars (including ``0``/``False``/``""``) pass through at
        any depth; any other leaf becomes ``"<redacted:TypeName>"``. A
        container already on the active recursion path (a cycle) is dropped,
        and past ``MAX_CONTEXT_FILTER_NODES`` visited values the rest is
        dropped with a WARNING.

        # DECISION plan-2026-09-21T203800-8a03483a/D-010
        # Non-JSON leaves are REDACTED via the shared
        # `context.filtering.redact_non_json_leaf`: `json.dumps(default=str)` would
        # otherwise print a pydantic model's / dataclass's fields, secrets
        # included, past every key check. Do NOT pass them through.
        # DECISION plan-2026-09-21T203800-8a03483a/D-011
        # Cycle guard (active path only) + node budget bound the work; the
        # depth bound stays. Do NOT memoise shared containers (changes output
        # for shared-but-acyclic data). See decisions.md D-010, D-011.
        """
        if not self.config.filter_internal_context:
            return context_data

        # One walk = one active-path set and one node counter (a 1-cell list,
        # decremented in place by every value visit).
        return self._filter_context_mapping(
            context_data, 0, {id(context_data)}, [MAX_CONTEXT_FILTER_NODES]
        )

    def _is_forbidden_context_key(self, key: Any, value: Any = None) -> bool:
        """Return True if this context entry must never reach the LLM prompt.

        Non-``str`` keys carry no prefix/pattern to match (and would raise on
        ``.startswith``), so they are never forbidden -- but the skip is logged
        at WARNING rather than silent. See constants.py D-017: a `bytes` key
        bypasses every name check here, and pre-fix it raised loudly instead.

        *value* feeds constants.py's layer 2 (D-019), which decides the
        ambiguous ``<qualifier>_key`` shape on the VALUE's shape because the
        NAME cannot separate `stripe_key` from `order_key`. It is optional so
        that name-only callers keep working.
        """
        if not isinstance(key, str):
            logger.warning(
                f"Prompt context key {key!r} ({type(key).__name__}) skipped the "
                "security name checks: only str keys can be matched against "
                "internal prefixes and forbidden patterns"
            )
            return False
        if has_internal_prefix(key, self.config.internal_key_prefixes):
            return True
        return is_forbidden_context_entry(key, value)

    def _filter_context_value(
        self, value: Any, depth: int, active: set[int], budget: list[int]
    ) -> Any:
        """Filter one context value; returns a drop sentinel when it must go.

        ``_TOO_DEEP`` past the depth bound or on a cycle (a container already
        in ``active``, the ids on the current recursion path);
        ``_OVER_BUDGET`` once ``budget[0]`` visits are spent. Containers are
        rebuilt; JSON-native leaves are returned unchanged, others redacted.
        """
        budget[0] -= 1
        if budget[0] < 0:
            if budget[0] == -1:  # log once per walk, at the first overrun
                logger.warning(
                    "Context values dropped from prompt: more than "
                    f"{MAX_CONTEXT_FILTER_NODES} values to security-filter"
                )
            return _OVER_BUDGET
        if not isinstance(value, (dict, list, tuple)):
            return redact_non_json_leaf(value)
        if depth > MAX_CONTEXT_FILTER_DEPTH:
            logger.warning(
                f"Context value dropped from prompt: nested deeper than "
                f"{MAX_CONTEXT_FILTER_DEPTH} levels and cannot be security-filtered"
            )
            return _TOO_DEEP
        if id(value) in active:
            logger.warning("Context value dropped from prompt: reference cycle")
            return _TOO_DEEP
        active.add(id(value))
        try:
            if isinstance(value, dict):
                return self._filter_context_mapping(value, depth, active, budget)

            # Lists/tuples are in scope: `{"users": [{"password": "x"}]}` is
            # the same leak as `{"user": {"password": "x"}}`.
            kept = []
            for item in value:
                cleaned = self._filter_context_value(item, depth + 1, active, budget)
                if cleaned is _OVER_BUDGET:
                    break
                if cleaned is not _TOO_DEEP:
                    kept.append(cleaned)
            return tuple(kept) if isinstance(value, tuple) else kept
        finally:
            active.discard(id(value))

    def _filter_context_mapping(
        self,
        source: dict[str, Any],
        depth: int,
        active: set[int],
        budget: list[int],
    ) -> dict[str, Any]:
        """Apply the key filter to one mapping level, then recurse into values."""
        filtered: dict[str, Any] = {}
        for key, value in source.items():
            if self._is_forbidden_context_key(key, value):
                continue
            cleaned = self._filter_context_value(value, depth + 1, active, budget)
            if cleaned is _OVER_BUDGET:
                break
            if cleaned is _TOO_DEEP:
                continue
            filtered[key] = cleaned
        return filtered

    def _humanize_key(self, key: str) -> str:
        """Convert technical context keys to human-readable descriptions."""
        key_mappings = {
            "email": "email address",
            "phone": "phone number",
            "name": "full name",
            "address": "address",
            "preferences": "preferences",
            "user_id": "user information",
            "date_of_birth": "date of birth",
            "payment_method": "payment information",
        }
        return key_mappings.get(key, key.replace("_", " "))

    def _build_enhanced_context_section(
        self, instance: FSMInstance, context: dict[str, Any] | None = None
    ) -> list[str]:
        """Build enhanced context section with security filtering.

        ``context`` is an optional pre-scoped view of the context (the
        pipeline passes ``context_scope.read_keys``-filtered data). ``None``
        means the full ``instance.context.data``, so callers that do not scope
        get a byte-identical section. The security filter still runs on
        whatever is passed, so a secret-named key inside ``read_keys`` is
        dropped.
        """
        data = instance.context.data if context is None else context
        if not data:
            return []

        # Filter for security FIRST, then cap. Capping first would let filtered-out
        # secret keys consume budget slots and evict legitimate ones.
        user_context = self._filter_context_for_security(data)
        user_context = self._limit_context_by_key_count(user_context)

        if not user_context:
            return []

        return self._json_cdata_block("current_context", user_context, "context")

    # ========================================================================
    # SHARED SECTION FACTORIES (used by all prompt builders)
    # ========================================================================

    @staticmethod
    def _build_task_section(task_description: str) -> list[str]:
        """Build a ``<task>`` section wrapping the given description."""
        return ["<task>", textwrap.dedent(task_description).strip(), "</task>", ""]

    @staticmethod
    def _build_response_format(
        json_schema: str,
        field_descriptions: list[str],
        notes: list[str] | None = None,
    ) -> list[str]:
        """Build a ``<response_format>`` section with schema + field docs.

        Args:
            json_schema: The JSON schema example (will be dedented).
            field_descriptions: Lines describing each field (tab-prefixed).
            notes: Optional extra notes appended after the field descriptions.
        """
        sections = [
            "<response_format>",
            "Your response must be valid JSON with the following structure:",
            textwrap.dedent(json_schema).strip(),
            "",
            "Where:",
        ]
        sections.extend(f"\t- {d}" for d in field_descriptions)
        if notes:
            sections.append("")
            heading = notes[0]
            sections.append(heading)
            sections.extend(f"\t- {n}" for n in notes[1:])
        sections.extend(["</response_format>", ""])
        return sections

    @staticmethod
    def _build_guidelines(guidelines_text: str) -> list[str]:
        """Build a ``<guidelines>`` section wrapping the given text."""
        return [
            "<guidelines>",
            textwrap.dedent(guidelines_text).strip(),
            "</guidelines>",
            "",
        ]

    def _build_summary_block(self, summary: str | None) -> str:
        """Render ``Conversation.summary`` as one sanitized line.

        Contract (shared by the base history section and the per-field
        prompt): returns ``<conversation_summary>...</conversation_summary>``
        with the text passed through ``_sanitize_text_for_prompt`` (tags
        escaped, newlines flattened), or ``""`` for a None/blank summary so
        prompts of conversations that never trimmed history stay
        byte-identical. Never raises.
        """
        # DECISION plan-2026-09-21T203800-8a03483a/D-007 (A6): the summary is
        # the only trace of trimmed exchanges, so it must reach the prompt,
        # but it is raw user/assistant text. Do NOT emit it unsanitized or
        # inside the history CDATA (its escaping covers `]]>`/`</` only), and
        # do NOT emit an empty block when there is no summary.
        if not summary or not summary.strip():
            return ""
        text = self._sanitize_text_for_prompt(summary)
        return f"<conversation_summary>{text}</conversation_summary>"

    def _build_enhanced_history_section(self, instance: FSMInstance) -> list[str]:
        """Build enhanced conversation history section.

        A non-empty ``Conversation.summary`` of trimmed exchanges is emitted
        first as a ``<conversation_summary>`` block (A6).
        """
        summary, recent_exchanges = (
            instance.context.conversation.get_summary_and_recent(
                self.config.max_history_messages
            )
        )
        summary_block = self._build_summary_block(summary)
        summary_lines = [summary_block, ""] if summary_block else []

        if not recent_exchanges:
            return summary_lines

        # Format and manage exchanges
        formatted_exchanges = [
            {role: self._sanitize_text_for_prompt(message)}
            for role, message in _history_messages(recent_exchanges)
        ]

        # Apply history management strategy
        managed_exchanges = self._manage_conversation_history(formatted_exchanges)

        if managed_exchanges:
            block = self._json_cdata_block(
                "conversation_history", managed_exchanges, "conversation history"
            )
            if block:
                return [*summary_lines, *block]

        return summary_lines
