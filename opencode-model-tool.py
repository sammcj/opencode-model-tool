#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["json5", "httpx", "textual"]
# ///
"""opencode-model-tool: Discover and configure local LLM models for OpenCode.

Scans an OpenAI-compatible API endpoint for available models, lets you
select which ones to add, and safely updates your OpenCode config.

Usage:
    uv run opencode-model-tool.py --endpoint https://llamaswap.your.domain/v1
    uv run opencode-model-tool.py --endpoint http://localhost:8080/v1 --api-key sk-...
    uv run opencode-model-tool.py --list --endpoint https://llamaswap.your.domain/v1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from urllib.parse import urlparse

import httpx
import json5
from rich.text import Text
from textual.app import App, ComposeResult
from textual.message import Message
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Footer, Header, Input, SelectionList, Static
from textual.widgets.selection_list import Selection

# Models that are not useful for OpenCode (embeddings, rerankers, etc.)
EXCLUDED_PATTERNS = ("embedding", "reranker", "embed", "minilm", "rerank")
EXCLUDED_EXACT = ("llamacpp",)

DEFAULT_OUTPUT_TOKENS = 65536
DEFAULT_CONTEXT_FALLBACK = 192000  # 192k - suggested when prompting the user
STATE_FILENAME = ".opencode-models-state.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Discover and configure local LLM models for OpenCode.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--endpoint",
        required=True,
        help="OpenAI-compatible API base URL (e.g. https://llamaswap.your.domain/v1)",
    )
    p.add_argument(
        "--provider-id",
        help="Provider key in opencode config. Auto-detected from baseURL if omitted.",
    )
    p.add_argument(
        "--config",
        help="Path to opencode config file. Auto-detected if omitted.",
    )
    p.add_argument(
        "--api-key",
        help="API key for authenticated endpoints.",
    )
    p.add_argument(
        "--api-key-env",
        help="Environment variable name containing the API key.",
    )
    p.add_argument(
        "--default-output",
        type=int,
        default=DEFAULT_OUTPUT_TOKENS,
        help=f"Default max output tokens (default: {DEFAULT_OUTPUT_TOKENS}).",
    )
    p.add_argument(
        "--include-embeddings",
        action="store_true",
        help="Include embedding/reranker models (excluded by default).",
    )
    p.add_argument(
        "--list",
        action="store_true",
        dest="list_only",
        help="List available models without interactive selection.",
    )
    p.add_argument(
        "--all",
        action="store_true",
        help="Select all models (skip interactive picker).",
    )
    p.add_argument(
        "--yes",
        action="store_true",
        help="Skip confirmation prompt before writing config.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model fetching
# ---------------------------------------------------------------------------


def normalise_endpoint(url: str) -> str:
    """Ensure endpoint URL has no trailing slash and doesn't end with /models."""
    url = url.rstrip("/")
    if url.endswith("/models"):
        url = url[: -len("/models")]
    return url


def fetch_models(endpoint: str, api_key: str | None = None) -> list[dict]:
    """Fetch model list from the /models endpoint."""
    url = f"{normalise_endpoint(endpoint)}/models"
    headers: dict[str, str] = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = httpx.get(url, headers=headers, timeout=30, follow_redirects=True)
        resp.raise_for_status()
    except httpx.HTTPStatusError as exc:
        print(f"Error: API returned {exc.response.status_code} from {url}", file=sys.stderr)
        sys.exit(1)
    except httpx.ConnectError:
        print(f"Error: Could not connect to {url}", file=sys.stderr)
        sys.exit(1)
    except httpx.RequestError as exc:
        print(f"Error: Request failed: {exc}", file=sys.stderr)
        sys.exit(1)

    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        print(f"Error: Response from {url} is not valid JSON.", file=sys.stderr)
        sys.exit(1)
    return data.get("data", [])


def is_excluded(model_id: str) -> bool:
    """Check if a model should be excluded (embeddings, rerankers, etc.)."""
    if model_id in EXCLUDED_EXACT:
        return True
    lower = model_id.lower()
    return any(pat in lower for pat in EXCLUDED_PATTERNS)


def _base_url(endpoint: str) -> str:
    """Strip /v1 suffix to get the server base URL."""
    norm = normalise_endpoint(endpoint)
    if norm.endswith("/v1"):
        return norm[:-3]
    return norm


def _auth_headers(api_key: str | None) -> dict[str, str]:
    """Build auth headers from an optional API key."""
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    return {}


def extract_context_from_model_data(model: dict) -> int | None:
    """Extract context size from model API response data.

    Checks fields used by different providers:
    - meta.n_ctx_train (llama.cpp / llamaswap)
    - context_length, max_model_len, context_window (other providers)
    """
    meta = model.get("meta")
    if isinstance(meta, dict):
        n_ctx = meta.get("n_ctx_train")
        if isinstance(n_ctx, int) and n_ctx > 0:
            return n_ctx

    for key in ("context_length", "max_model_len", "context_window"):
        val = model.get(key)
        if isinstance(val, int) and val > 0:
            return val

    return None


def extract_output_from_model_data(model: dict) -> int | None:
    """Extract max output tokens from model API response data.

    Checks fields used by different providers:
    - max_output_tokens, max_completion_tokens (OpenAI-style)
    """
    for key in ("max_output_tokens", "max_completion_tokens"):
        val = model.get(key)
        if isinstance(val, int) and val > 0:
            return val

    return None


def fetch_model_info_limits(
    endpoint: str, api_key: str | None = None
) -> tuple[dict[str, int], dict[str, int]]:
    """Fetch model context and output sizes from /model/info (litellm).

    Returns (contexts, outputs) dicts mapping model ID to token counts.
    Silently returns empty dicts on any failure.
    """
    url = f"{normalise_endpoint(endpoint)}/model/info"
    try:
        resp = httpx.get(
            url, headers=_auth_headers(api_key), timeout=10, follow_redirects=True
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return {}, {}

    contexts: dict[str, int] = {}
    outputs: dict[str, int] = {}
    for entry in data.get("data", []):
        model_name = entry.get("model_name") or entry.get("id", "")
        if not model_name:
            continue
        model_info = entry.get("model_info", {})
        if isinstance(model_info, dict):
            # litellm: model_info.max_input_tokens or model_info.max_tokens
            ctx = model_info.get("max_input_tokens") or model_info.get("max_tokens")
            if isinstance(ctx, int) and ctx > 0:
                contexts[model_name] = ctx
            # litellm: model_info.max_output_tokens
            out = model_info.get("max_output_tokens")
            if isinstance(out, int) and out > 0:
                outputs[model_name] = out
            if model_name in contexts:
                continue
        # Fallback to top-level fields (n_ctx_train, context_length, etc.)
        ctx = extract_context_from_model_data(entry)
        if ctx:
            contexts[model_name] = ctx

    return contexts, outputs


def _parse_props_response(data: dict) -> tuple[int | None, int | None, str]:
    """Parse a llama.cpp /props response.

    Returns (n_ctx, n_predict, model_alias).
    n_ctx is the per-slot runtime context (max usable per request).
    n_predict is the server's max generation length (-1 means unlimited).
    """
    gen_settings = data.get("default_generation_settings")
    if not isinstance(gen_settings, dict):
        return None, None, data.get("model_alias", "")
    n_ctx = gen_settings.get("n_ctx")
    model_alias = data.get("model_alias", "")

    # n_predict may be in params or at the top level of gen_settings
    params = gen_settings.get("params", {})
    n_predict = params.get("n_predict") if isinstance(params, dict) else None
    if n_predict is None:
        n_predict = gen_settings.get("n_predict")

    ctx = n_ctx if isinstance(n_ctx, int) and n_ctx > 0 else None
    # n_predict of -1 means unlimited (up to context size)
    pred = n_predict if isinstance(n_predict, int) and n_predict > 0 else None
    return ctx, pred, model_alias


def fetch_props_limits(
    endpoint: str, api_key: str | None = None
) -> tuple[int | None, int | None, str]:
    """Try GET /props on direct llama.cpp servers.

    Returns (n_ctx, n_predict, model_alias) or (None, None, "") on failure.
    """
    url = f"{_base_url(endpoint)}/props"
    try:
        resp = httpx.get(
            url, headers=_auth_headers(api_key), timeout=10, follow_redirects=True
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return None, None, ""

    return _parse_props_response(data)


def fetch_upstream_props_limits(
    endpoint: str,
    model_ids: list[str],
    api_key: str | None = None,
) -> tuple[dict[str, int], dict[str, int]]:
    """Fetch runtime limits via llama-swap's /upstream/{model}/props passthrough.

    Each call loads the model on the remote server, so this should only be
    called after the user has confirmed their selection.

    Returns (contexts, outputs) dicts mapping model ID to token counts.
    """
    base = _base_url(endpoint)
    headers = _auth_headers(api_key)

    contexts: dict[str, int] = {}
    outputs: dict[str, int] = {}
    for i, mid in enumerate(model_ids, 1):
        print(f"  [{i}/{len(model_ids)}] {mid} ...", end="", flush=True)
        url = f"{base}/upstream/{mid}/props"
        try:
            resp = httpx.get(url, headers=headers, timeout=60, follow_redirects=True)
            resp.raise_for_status()
            data = resp.json()
        except Exception:
            print(" failed")
            continue

        n_ctx, n_predict, _ = _parse_props_response(data)
        if n_ctx:
            contexts[mid] = n_ctx
            parts = [f"context: {n_ctx:,}"]
            if n_predict:
                outputs[mid] = n_predict
                parts.append(f"output: {n_predict:,}")
            print(f" {', '.join(parts)}")
        else:
            print(" no info")

    return contexts, outputs


# ---------------------------------------------------------------------------
# Context length parsing
# ---------------------------------------------------------------------------


def parse_context_from_id(model_id: str) -> int | None:
    """Extract context length from model ID segments like '128k', '64k', '192k'.

    Uses k=1000 (not 1024) since model names typically reflect configured
    context sizes which use decimal thousands. This is an estimate - API-detected
    context should always be preferred.

    Returns token count or None if not found.
    """
    segments = model_id.split("-")
    matches = [s for s in segments if re.match(r"^\d+k$", s, re.IGNORECASE)]
    if not matches:
        return None
    # Use the last match (most likely to be the context length)
    ctx_k = int(matches[-1].lower().rstrip("k"))
    return ctx_k * 1000


def build_model_config(
    model_id: str,
    default_output: int = DEFAULT_OUTPUT_TOKENS,
    api_context: int | None = None,
    api_output: int | None = None,
) -> dict:
    """Build an OpenCode model config entry from a model ID.

    Context priority: api_context > name parsing > DEFAULT_CONTEXT_FALLBACK.
    Output priority: api_output > default_output, capped to context.
    """
    context = api_context or parse_context_from_id(model_id) or DEFAULT_CONTEXT_FALLBACK
    output = min(api_output or default_output, context)
    return {
        "name": model_id,
        "limit": {
            "context": context,
            "output": output,
        },
    }


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


def state_path() -> Path:
    return Path.home() / ".opencode" / STATE_FILENAME


def load_state() -> dict:
    p = state_path()
    if p.exists():
        try:
            return json.loads(p.read_text())
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_state(state: dict) -> None:
    p = state_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Config file handling
# ---------------------------------------------------------------------------


def find_config() -> Path | None:
    """Auto-detect the OpenCode config file."""
    candidates = [
        Path.home() / ".opencode" / "opencode.jsonc",
        Path.home() / ".opencode" / "opencode.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def detect_provider_id(config_data: dict, endpoint: str) -> str | None:
    """Find the provider key whose baseURL matches the given endpoint."""
    norm = normalise_endpoint(endpoint)
    providers = config_data.get("provider", {})
    for pid, pconf in providers.items():
        opts = pconf.get("options", {})
        if normalise_endpoint(opts.get("baseURL", "")) == norm:
            return pid
    return None


def read_config_model_ids(config_data: dict, provider_id: str) -> set[str]:
    """Read model IDs currently configured for a provider."""
    provider = config_data.get("provider", {}).get(provider_id, {})
    return set(provider.get("models", {}).keys())


def find_provider_section_span(text: str) -> tuple[int, int] | None:
    """Find the span of the top-level 'provider' object value in the config.

    Returns (brace_start, brace_end) of the "provider": {...} value, or None.
    """
    match = re.search(r'"provider"\s*:\s*\{', text)
    if not match:
        return None
    brace_start = match.end() - 1
    brace_end = find_matching_brace(text, brace_start)
    if brace_end is None:
        return None
    return (brace_start, brace_end)


def find_models_span(text: str, provider_id: str) -> tuple[int, int] | None:
    """Find the byte span of the 'models': {...} value within a provider block.

    Returns (start, end) indices of the outermost braces of the models object,
    or None if not found. Scopes the search to within the top-level "provider"
    section to avoid false matches elsewhere in the config.
    """
    # Scope search to the "provider" section to avoid false matches in mcp/etc
    prov_section = find_provider_section_span(text)
    if prov_section is None:
        return None
    section_start, section_end = prov_section
    search_text = text[section_start : section_end + 1]

    # Find the provider key within the provider section
    provider_pattern = re.compile(
        rf'"{re.escape(provider_id)}"\s*:\s*\{{', re.DOTALL
    )
    provider_match = provider_pattern.search(search_text)
    if not provider_match:
        return None

    # Convert relative offset to absolute (within full text)
    provider_brace_start = section_start + provider_match.end() - 1
    provider_brace_end = find_matching_brace(text, provider_brace_start)
    if provider_brace_end is None:
        return None

    provider_block = text[provider_brace_start : provider_brace_end + 1]

    # Within the provider block, find "models":
    models_pattern = re.compile(r'"models"\s*:\s*\{', re.DOTALL)
    models_match = models_pattern.search(provider_block)
    if not models_match:
        return None

    # Convert relative-to-provider-block offset to absolute
    models_brace_start = provider_brace_start + models_match.end() - 1
    models_brace_end = find_matching_brace(text, models_brace_start)
    if models_brace_end is None:
        return None

    return (models_brace_start, models_brace_end)


def find_matching_brace(text: str, start: int) -> int | None:
    """Find the matching closing brace for an opening brace at position start.

    Handles nested braces, strings, and JSONC comments.
    """
    if start >= len(text) or text[start] != "{":
        return None

    depth = 0
    i = start
    in_string = False
    in_line_comment = False
    in_block_comment = False

    while i < len(text):
        ch = text[i]

        if in_line_comment:
            if ch == "\n":
                in_line_comment = False
            i += 1
            continue

        if in_block_comment:
            if ch == "*" and i + 1 < len(text) and text[i + 1] == "/":
                in_block_comment = False
                i += 2
                continue
            i += 1
            continue

        if in_string:
            if ch == "\\" and i + 1 < len(text):
                i += 2  # skip escaped char
                continue
            if ch == '"':
                in_string = False
            i += 1
            continue

        # Not in string or comment
        if ch == "/" and i + 1 < len(text):
            next_ch = text[i + 1]
            if next_ch == "/":
                in_line_comment = True
                i += 2
                continue
            if next_ch == "*":
                in_block_comment = True
                i += 2
                continue

        if ch == '"':
            in_string = True
            i += 1
            continue

        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return i

        i += 1

    return None


def format_models_json(models: dict, indent: int = 8) -> str:
    """Format a models dict as JSONC-style JSON with trailing commas."""
    if not models:
        return "{}"

    lines = ["{"]
    model_items = list(models.items())
    for model_id, config in model_items:
        mid_s = json.dumps(model_id)
        name_s = json.dumps(config["name"])
        lines.append(f'{" " * indent}{mid_s}: {{')
        lines.append(f'{" " * (indent + 2)}"name": {name_s},')
        lines.append(f'{" " * (indent + 2)}"limit": {{')
        lines.append(f'{" " * (indent + 4)}"context": {config["limit"]["context"]},')
        lines.append(f'{" " * (indent + 4)}"output": {config["limit"]["output"]}')
        lines.append(f'{" " * (indent + 2)}}}')
        lines.append(f'{" " * indent}}},')
    lines.append(f'{" " * (indent - 2)}}}')
    return "\n".join(lines)


def build_new_provider_block(
    provider_id: str,
    display_name: str,
    endpoint: str,
    models: dict,
    api_key_env: str | None = None,
) -> str:
    """Build a complete new provider block for insertion into the config."""
    indent = "    "
    pid_s = json.dumps(provider_id)
    name_s = json.dumps(display_name)
    url_s = json.dumps(endpoint)
    lines = [f'{indent}{pid_s}: {{']
    lines.append(f'{indent}  "npm": "@ai-sdk/openai-compatible",')
    lines.append(f'{indent}  "name": {name_s},')
    lines.append(f'{indent}  "options": {{')
    lines.append(f'{indent}    "baseURL": {url_s}')
    if api_key_env:
        lines[-1] = lines[-1] + ","
        key_ref = json.dumps(f"{{env:{api_key_env}}}")
        lines.append(f'{indent}    "apiKey": {key_ref}')
    lines.append(f'{indent}  }},')
    models_json = format_models_json(models, indent=8)
    lines.append(f'{indent}  "models": {models_json}')
    lines.append(f'{indent}}}')
    return "\n".join(lines)


def update_config_models(
    config_path: Path,
    provider_id: str,
    models: dict,
    endpoint: str,
    existing_model_ids: set[str] | None = None,
    api_key_env: str | None = None,
    skip_confirm: bool = False,
) -> bool:
    """Update the models block for a provider in the config file.

    Returns True if the file was updated, False if cancelled.
    """
    text = config_path.read_text()

    span = find_models_span(text, provider_id)

    if span is not None:
        # Replace existing models block
        start, end = span
        new_models_json = format_models_json(models)
        new_text = text[:start] + new_models_json + text[end + 1 :]
    else:
        # Provider doesn't exist yet - add it
        prov_span = find_provider_section_span(text)
        if prov_span is None:
            print("Error: No 'provider' section found in config.", file=sys.stderr)
            return False
        prov_brace_start, prov_brace_end = prov_span

        # Insert new provider before the closing brace
        display_name = provider_id
        new_block = build_new_provider_block(
            provider_id, display_name, endpoint, models, api_key_env
        )

        # Check if there's content before the closing brace (need a comma)
        before_close = text[prov_brace_start + 1 : prov_brace_end].rstrip()
        if before_close and not before_close.endswith(","):
            last_content_pos = prov_brace_start + 1 + len(
                text[prov_brace_start + 1 : prov_brace_end].rstrip()
            )
            text = text[:last_content_pos] + "," + text[last_content_pos:]
            prov_brace_end += 1

        new_text = text[:prov_brace_end] + "\n" + new_block + ",\n  " + text[prov_brace_end:]

    # Show diff relative to what's currently in config
    existing = existing_model_ids or set()
    selected = set(models.keys())
    adding = sorted(selected - existing)
    keeping = sorted(selected & existing)
    removing = sorted(existing - selected)

    print("\n--- Changes to apply ---")
    if span is not None:
        print(f"Updating models for provider '{provider_id}' in {config_path}")
    else:
        print(f"Adding new provider '{provider_id}' to {config_path}")

    if adding:
        print(f"\n  Adding {len(adding)} model(s):")
        for mid in adding:
            ctx = models[mid]["limit"]["context"]
            print(f"    + {mid} (context: {ctx:,})")
    if keeping:
        print(f"\n  Keeping {len(keeping)} model(s):")
        for mid in keeping:
            ctx = models[mid]["limit"]["context"]
            print(f"    = {mid} (context: {ctx:,})")
    if removing:
        print(f"\n  Removing {len(removing)} model(s):")
        for mid in removing:
            print(f"    - {mid}")
    print()

    if not skip_confirm:
        answer = input("Apply these changes? [Y/n] ").strip().lower()
        if answer and answer not in ("y", "yes"):
            print("Cancelled.")
            return False

    # Backup
    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    shutil.copy2(config_path, backup_path)
    print(f"Backup saved to {backup_path}")

    config_path.write_text(new_text)
    print(f"Config updated: {config_path}")
    return True


# ---------------------------------------------------------------------------
# Interactive selection
# ---------------------------------------------------------------------------


class ModelList(SelectionList[str]):
    """SelectionList with custom bindings that override type-ahead."""

    class Confirmed(Message):
        """Posted when the user presses Enter to confirm."""

    class SearchRequested(Message):
        """Posted when the user presses / to search."""

    class CancelRequested(Message):
        """Posted when the user presses q or Escape."""

    BINDINGS = [
        Binding("enter", "confirm", "Confirm", priority=True),
        Binding("slash", "request_search", "Search", priority=True),
        Binding("a", "toggle_all_models", "Toggle all", priority=True),
        Binding("q", "request_cancel", "Quit", priority=True),
        Binding("escape", "request_cancel", show=False, priority=True),
        Binding("ctrl+c", "request_cancel", show=False, priority=True),
    ]

    def action_confirm(self) -> None:
        self.post_message(self.Confirmed())

    def action_request_search(self) -> None:
        self.post_message(self.SearchRequested())

    def action_toggle_all_models(self) -> None:
        self.toggle_all()

    def action_request_cancel(self) -> None:
        self.post_message(self.CancelRequested())


class ModelSelectorApp(App[list[str] | None]):
    """Interactive model selector with live config preview."""

    TITLE = "opencode-model-tool"

    CSS = """
    #main { height: 1fr; }
    #left-panel { width: 1fr; }
    #model-list { height: 1fr; }
    #search { display: none; }
    #preview-scroll {
        width: 1fr;
        border-left: thick $primary;
    }
    #preview { padding: 1 2; }
    """

    # Fallback bindings for when search Input is focused
    BINDINGS = [
        Binding("escape", "escape_fallback", show=False),
        Binding("ctrl+c", "escape_fallback", show=False),
    ]

    def __init__(
        self,
        selections: list[Selection[str]],
        previously_selected: set[str],
        removed_selected: list[str],
        removed_other: list[str],
        default_output: int,
        api_contexts: dict[str, int] | None = None,
    ) -> None:
        super().__init__()
        self._selections = selections
        self._previously_selected = previously_selected
        self._removed_selected = removed_selected
        self._removed_other = removed_other
        self._default_output = default_output
        self._api_contexts = api_contexts or {}
        # Ground truth for selection state (survives filter rebuilds)
        self._selected_ids: set[str] = {
            sel.value for sel in selections if sel.initial_state
        }
        self._current_filter = ""
        self._rebuilding = False

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal(id="main"):
            with Vertical(id="left-panel"):
                yield Input(placeholder="Type to filter...", id="search")
                yield ModelList(*self._selections, id="model-list")
            with VerticalScroll(id="preview-scroll"):
                yield Static("", id="preview")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#model-list", ModelList).focus()
        self._update_preview()

    def on_selection_list_selection_toggled(
        self, event: SelectionList.SelectionToggled
    ) -> None:
        if self._rebuilding:
            return
        # Sync ground truth with visible widget state
        ml = self.query_one("#model-list", ModelList)
        visible_selected = set(ml.selected)
        visible_ids = {
            sel.value for sel in self._selections
            if not self._current_filter
            or self._current_filter in sel.value.lower()
        }
        for vid in visible_ids:
            if vid in visible_selected:
                self._selected_ids.add(vid)
            else:
                self._selected_ids.discard(vid)
        self._update_preview()

    def _rebuild_list(self, filter_query: str = "") -> None:
        """Clear and repopulate the list, showing only matching items."""
        self._rebuilding = True
        self._current_filter = filter_query
        ml = self.query_one("#model-list", ModelList)
        ml.clear_options()
        for sel in self._selections:
            if filter_query and filter_query not in sel.value.lower():
                continue
            is_selected = sel.value in self._selected_ids
            ml.add_option(Selection(sel.prompt, sel.value, is_selected))
        self._rebuilding = False
        self._update_preview()

    def _update_preview(self) -> None:
        current = self._selected_ids
        prev = self._previously_selected

        adding = sorted(current - prev)
        keeping = sorted(current & prev)
        dropping = sorted(prev - current)

        lines: list[str] = []

        # Summary
        parts = []
        if adding:
            parts.append(f"[green]+{len(adding)} new[/green]")
        if keeping:
            parts.append(f"{len(keeping)} unchanged")
        if dropping:
            parts.append(f"[red]-{len(dropping)} removed[/red]")
        summary = ", ".join(parts) if parts else "none"
        lines.append(f"[bold]{len(current)} selected[/bold] ({summary})\n")

        # Show config for newly added models only
        if adding:
            models = {
                mid: build_model_config(
                    mid, self._default_output, self._api_contexts.get(mid)
                )
                for mid in adding
            }
            lines.append("[bold green]Adding to config:[/bold green]")
            lines.append(format_models_json(models, indent=2))

        # Show models being deselected (were in config, now unchecked)
        if dropping:
            lines.append("\n[bold yellow]Dropping from config (deselected):[/bold yellow]")
            for mid in dropping:
                lines.append(f"  [yellow]- {mid}[/yellow]")

        # Models gone from the endpoint entirely
        if self._removed_selected:
            lines.append(
                "\n[bold red]Gone from endpoint "
                "(will be removed from config):[/bold red]"
            )
            for mid in self._removed_selected:
                lines.append(f"  [red]x {mid}[/red]")

        if self._removed_other:
            lines.append(
                "\n[dim]Also gone from endpoint (were not in config):[/dim]"
            )
            for mid in self._removed_other:
                lines.append(f"  [dim]- {mid}[/dim]")

        # Show unchanged as a collapsed count
        if keeping:
            lines.append(f"\n[dim]{len(keeping)} model(s) unchanged in config[/dim]")

        self.query_one("#preview", Static).update("\n".join(lines))

    def on_model_list_confirmed(self) -> None:
        self.exit(sorted(self._selected_ids))

    def on_model_list_search_requested(self) -> None:
        search = self.query_one("#search", Input)
        if search.display:
            self._close_search()
        else:
            search.display = True
            search.value = ""
            search.focus()

    def on_model_list_cancel_requested(self) -> None:
        self.exit(None)

    def on_input_changed(self, event: Input.Changed) -> None:
        self._rebuild_list(event.value.strip().lower())

    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Enter in search bar: keep filter active, return focus to list."""
        self.query_one("#model-list", ModelList).focus()

    def _close_search(self) -> None:
        search = self.query_one("#search", Input)
        search.display = False
        search.value = ""
        self._rebuild_list()  # Restore full list
        self.query_one("#model-list", ModelList).focus()

    def action_escape_fallback(self) -> None:
        """Handle escape when search Input is focused."""
        search = self.query_one("#search", Input)
        if search.display:
            self._close_search()
        else:
            self.exit(None)


def categorise_models(
    model_ids: list[str],
    state_entry: dict,
    config_model_ids: set[str],
) -> tuple[list[str], list[str], list[str], list[str], list[str]]:
    """Categorise models relative to state and current config.

    Returns (new, in_config, available, gone_configured, gone_other).
    - new: on endpoint, never seen in state before, not already in config.
    - in_config: on endpoint AND currently in the config file.
    - available: on endpoint, not new, not in config.
    - gone_configured: in config but no longer on the endpoint.
    - gone_other: was known in state, not on endpoint, not in config.
    """
    known = set(state_entry.get("known", []))
    current = set(model_ids)
    config = config_model_ids

    new = sorted(current - known - config)
    in_config = sorted(current & config)
    available = sorted(current - set(new) - config)
    gone_configured = sorted(config - current)
    gone_other = sorted((known - current) - config)

    return new, in_config, available, gone_configured, gone_other


def _make_label(
    mid: str, prefix: str = "", api_context: int | None = None
) -> Text:
    """Build a styled label for a model selection entry."""
    if api_context:
        ctx_str = f" ({api_context // 1024}k ctx)"
    else:
        ctx = parse_context_from_id(mid)
        ctx_str = f" (~{ctx // 1024}k ctx)" if ctx else ""
    label = Text(f"{prefix}{mid}{ctx_str}")
    if prefix:
        label.stylize("bold green", 0, len(prefix))
    return label


def interactive_select(
    model_ids: list[str],
    state_entry: dict,
    config_model_ids: set[str],
    default_output: int = DEFAULT_OUTPUT_TOKENS,
    api_contexts: dict[str, int] | None = None,
) -> list[str] | None:
    """Show an interactive split-pane TUI for model selection.

    Left pane: checkbox list of models (new, in config, available).
    Right pane: live diff showing adds/removes relative to current config.

    Returns selected model IDs, or None if cancelled.
    """
    contexts = api_contexts or {}
    new, in_config, available, gone_configured, gone_other = (
        categorise_models(model_ids, state_entry, config_model_ids)
    )

    selections: list[Selection[str]] = []
    for mid in new:
        selections.append(
            Selection(_make_label(mid, "[NEW] ", contexts.get(mid)), mid, False)
        )
    for mid in in_config:
        selections.append(
            Selection(_make_label(mid, api_context=contexts.get(mid)), mid, True)
        )
    for mid in available:
        selections.append(
            Selection(_make_label(mid, api_context=contexts.get(mid)), mid, False)
        )

    if not selections:
        print("No models available for selection.")
        return []

    app = ModelSelectorApp(
        selections, config_model_ids, gone_configured, gone_other, default_output,
        api_contexts=contexts,
    )
    return app.run()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def resolve_api_key(args: argparse.Namespace) -> str | None:
    """Resolve API key from args or environment."""
    if args.api_key:
        return args.api_key
    if args.api_key_env:
        key = os.environ.get(args.api_key_env)
        if not key:
            print(
                f"Warning: Environment variable {args.api_key_env} is not set.",
                file=sys.stderr,
            )
        return key
    return None


def main() -> None:
    args = parse_args()
    endpoint = normalise_endpoint(args.endpoint)
    api_key = resolve_api_key(args)

    # Fetch models from endpoint
    print(f"Fetching models from {endpoint}/models ...")
    raw_models = fetch_models(endpoint, api_key)

    if not raw_models:
        print("No models returned from the endpoint.")
        sys.exit(1)

    all_ids = sorted(m["id"] for m in raw_models if "id" in m)
    print(f"Found {len(all_ids)} models.")

    # Detect endpoint type for context size probing
    owned_by_values = {m.get("owned_by", "").lower() for m in raw_models}
    is_llama_swap = "llama-swap" in owned_by_values
    is_direct_llamacpp = "llamacpp" in owned_by_values and not is_llama_swap

    # --- Limit detection (lightweight probes, no model loading) ---
    api_contexts: dict[str, int] = {}
    api_outputs: dict[str, int] = {}

    # 1. Extract from /v1/models response data (free, already fetched)
    for m in raw_models:
        mid = m.get("id", "")
        if not mid:
            continue
        ctx = extract_context_from_model_data(m)
        if ctx:
            api_contexts[mid] = ctx
        out = extract_output_from_model_data(m)
        if out:
            api_outputs[mid] = out

    # 2. Try /props (direct llama.cpp - returns runtime limits for loaded model)
    props_ctx, props_output, props_alias = fetch_props_limits(endpoint, api_key)
    if props_ctx:
        matched = False
        for mid in all_ids:
            if mid == props_alias or mid.lower() == props_alias.lower():
                api_contexts[mid] = props_ctx
                if props_output:
                    api_outputs[mid] = props_output
                matched = True
                break
        if not matched and len(all_ids) == 1:
            api_contexts[all_ids[0]] = props_ctx
            if props_output:
                api_outputs[all_ids[0]] = props_output

    # 3. Try /model/info (litellm proxy)
    info_contexts, info_outputs = fetch_model_info_limits(endpoint, api_key)
    api_contexts.update(info_contexts)
    api_outputs.update(info_outputs)

    if api_contexts:
        print(f"Detected context size for {len(api_contexts)} model(s) from API.")

    # Filter out embeddings/rerankers unless requested
    if args.include_embeddings:
        model_ids = all_ids
    else:
        model_ids = [mid for mid in all_ids if not is_excluded(mid)]
        excluded_count = len(all_ids) - len(model_ids)
        if excluded_count > 0:
            print(
                f"Excluded {excluded_count} embedding/reranker models "
                f"(use --include-embeddings to show them)."
            )

    if not model_ids:
        print("No eligible models after filtering.")
        sys.exit(1)

    # List-only mode
    if args.list_only:
        print(f"\nAvailable models ({len(model_ids)}):\n")
        for mid in model_ids:
            api_ctx = api_contexts.get(mid)
            if api_ctx:
                print(f"  {mid}  context: {api_ctx:,}")
            else:
                name_ctx = parse_context_from_id(mid)
                ctx_str = f"  context: ~{name_ctx:,} (from name)" if name_ctx else "  context: unknown"
                print(f"  {mid}{ctx_str}")
        return

    # Parse config early so we know what's already configured
    if args.config:
        config_path: Path | None = Path(args.config).expanduser()
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
    else:
        config_path = find_config()

    config_data = None
    provider_id = None
    config_model_ids: set[str] = set()

    if config_path is not None:
        try:
            config_data = json5.loads(config_path.read_text())
        except Exception as exc:
            print(f"Error parsing config: {exc}", file=sys.stderr)
            sys.exit(1)
        provider_id = args.provider_id or detect_provider_id(config_data, endpoint)
        if provider_id:
            config_model_ids = read_config_model_ids(config_data, provider_id)
            if config_model_ids:
                print(f"Found {len(config_model_ids)} existing model(s) for provider '{provider_id}'.")

    # Load state
    state = load_state()
    state_entry = state.get(endpoint, {})

    # Model selection
    if args.all:
        selected = model_ids
        _, _, _, gone_configured, _ = categorise_models(
            model_ids, state_entry, config_model_ids
        )
        print(f"\nSelected all {len(selected)} models.")
        if gone_configured:
            print("\nRemoved from endpoint (will be removed from config):")
            for mid in gone_configured:
                print(f"  x {mid}")
    else:
        selected = interactive_select(
            model_ids, state_entry, config_model_ids, args.default_output,
            api_contexts=api_contexts,
        )
        if selected is None:
            print("Cancelled.")
            return

    if not selected:
        print("No models selected.")
        return

    # After selection: for llama-swap / direct llama.cpp endpoints, offer to
    # query upstream props to get the actual runtime context size.
    newly_added = [mid for mid in selected if mid not in config_model_ids]
    models_without_api_ctx = [mid for mid in newly_added if mid not in api_contexts]
    if (is_llama_swap or is_direct_llamacpp) and models_without_api_ctx:
        print(
            f"\n{len(models_without_api_ctx)} model(s) don't have a known context size from the API."
        )
        print(
            "The tool can query each model's runtime properties to get the actual "
            "configured context size."
        )
        print(
            "Note: this may take a minute as llama.cpp may need to partially load "
            "each model to read its properties."
        )
        if args.yes:
            fetch_upstream = True
        else:
            answer = input("Fetch context sizes from the server? [Y/n] ").strip().lower()
            fetch_upstream = not answer or answer in ("y", "yes")

        if fetch_upstream:
            if is_llama_swap:
                up_ctx, up_out = fetch_upstream_props_limits(
                    endpoint, models_without_api_ctx, api_key
                )
            else:
                # Direct llama.cpp: /props applies to the single loaded model
                up_ctx: dict[str, int] = {}
                up_out: dict[str, int] = {}
                for mid in models_without_api_ctx:
                    ctx, out, _ = fetch_props_limits(endpoint, api_key)
                    if ctx:
                        up_ctx[mid] = ctx
                    if out:
                        up_out[mid] = out
            api_contexts.update(up_ctx)
            api_outputs.update(up_out)
        else:
            print("Skipping - will use estimates from model names where available.")

    # Warn about models with completely unknown context size and prompt the user
    unknown_ctx = [
        mid for mid in selected
        if mid not in api_contexts and parse_context_from_id(mid) is None
    ]
    if unknown_ctx:
        print(
            f"\n{len(unknown_ctx)} model(s) have no known context size "
            f"(not detected from API or model name):"
        )
        for mid in unknown_ctx:
            print(f"  - {mid}")
        if args.yes:
            print(f"Using default: {DEFAULT_CONTEXT_FALLBACK:,} tokens")
            for mid in unknown_ctx:
                api_contexts[mid] = DEFAULT_CONTEXT_FALLBACK
        else:
            answer = input(
                f"Enter context size in tokens for these models [{DEFAULT_CONTEXT_FALLBACK:,}]: "
            ).strip()
            if answer:
                try:
                    ctx_val = int(answer.lower().replace("k", "000").replace(",", ""))
                    if ctx_val <= 0:
                        raise ValueError
                except ValueError:
                    print(f"Invalid value, using default: {DEFAULT_CONTEXT_FALLBACK:,}")
                    ctx_val = DEFAULT_CONTEXT_FALLBACK
            else:
                ctx_val = DEFAULT_CONTEXT_FALLBACK
            for mid in unknown_ctx:
                api_contexts[mid] = ctx_val

    # Build model configs
    models_config = {}
    for mid in selected:
        models_config[mid] = build_model_config(
            mid, args.default_output, api_contexts.get(mid), api_outputs.get(mid)
        )

    if config_path is None:
        print("\nNo OpenCode config file found.")
        print("Generated config snippet for your provider:\n")
        print(json.dumps(models_config, indent=2))
        state[endpoint] = {
            "selected": sorted(selected),
            "known": sorted(model_ids),
        }
        save_state(state)
        return

    print(f"\nUsing config: {config_path}")

    # Prompt for provider_id if not yet resolved (new provider)
    if provider_id is None:
        parsed_url = urlparse(endpoint)
        hostname = parsed_url.hostname or "local"
        default_id = hostname.split(".")[0].replace("-", "_")
        if default_id in ("localhost", "127", "192", "10"):
            default_id = "local_llm"

        if args.yes:
            provider_id = default_id
            print(f"Auto-assigned provider ID: {provider_id}")
        else:
            provider_id = input(
                f"No existing provider found. Enter a provider ID [{default_id}]: "
            ).strip()
            if not provider_id:
                provider_id = default_id

    updated = update_config_models(
        config_path,
        provider_id,
        models_config,
        endpoint,
        existing_model_ids=config_model_ids,
        api_key_env=args.api_key_env,
        skip_confirm=args.yes,
    )

    # Save state only after successful config write
    if updated:
        state[endpoint] = {
            "selected": sorted(selected),
            "known": sorted(model_ids),
        }
        save_state(state)


if __name__ == "__main__":
    main()
