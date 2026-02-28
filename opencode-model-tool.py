#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = ["json5", "httpx", "InquirerPy"]
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
from InquirerPy import inquirer
from InquirerPy.base.control import Choice
from InquirerPy.separator import Separator

# Models that are not useful for OpenCode (embeddings, rerankers, etc.)
EXCLUDED_PATTERNS = ("embedding", "reranker", "embed", "minilm", "rerank")
EXCLUDED_EXACT = ("llamacpp",)

DEFAULT_OUTPUT_TOKENS = 65536
DEFAULT_CONTEXT_FALLBACK = 8192
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


# ---------------------------------------------------------------------------
# Context length parsing
# ---------------------------------------------------------------------------


def parse_context_from_id(model_id: str) -> int | None:
    """Extract context length from model ID segments like '128k', '64k', '192k'.

    Returns token count or None if not found.
    """
    segments = model_id.split("-")
    matches = [s for s in segments if re.match(r"^\d+k$", s, re.IGNORECASE)]
    if not matches:
        return None
    # Use the last match (most likely to be the context length)
    ctx_k = int(matches[-1].lower().rstrip("k"))
    return ctx_k * 1024


def build_model_config(
    model_id: str,
    default_output: int = DEFAULT_OUTPUT_TOKENS,
) -> dict:
    """Build an OpenCode model config entry from a model ID."""
    context = parse_context_from_id(model_id) or DEFAULT_CONTEXT_FALLBACK
    output = min(default_output, context)
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
        # Derive display name from provider_id
        display_name = provider_id.replace("_", " ").replace("-", " ").title()
        new_block = build_new_provider_block(
            provider_id, display_name, endpoint, models, api_key_env
        )

        # Check if there's content before the closing brace (need a comma)
        before_close = text[prov_brace_start + 1 : prov_brace_end].rstrip()
        if before_close and not before_close.endswith(","):
            # Add a trailing comma after the last provider
            last_content_pos = prov_brace_start + 1 + len(
                text[prov_brace_start + 1 : prov_brace_end].rstrip()
            )
            text = text[:last_content_pos] + "," + text[last_content_pos:]
            # Recalculate prov_brace_end since we inserted a character
            prov_brace_end += 1

        new_text = text[:prov_brace_end] + "\n" + new_block + ",\n  " + text[prov_brace_end:]

    # Show diff
    print("\n--- Changes to apply ---")
    if span is not None:
        print(f"Updating models for provider '{provider_id}' in {config_path}")
    else:
        print(f"Adding new provider '{provider_id}' to {config_path}")
    print(f"Models: {len(models)}")
    for mid in sorted(models):
        ctx = models[mid]["limit"]["context"]
        print(f"  {mid} (context: {ctx:,})")
    print()

    if not skip_confirm:
        proceed = inquirer.confirm(
            message="Apply these changes?", default=True
        ).execute()
        if not proceed:
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


def categorise_models(
    model_ids: list[str],
    state_entry: dict,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Categorise models into new, previously_selected, available, removed.

    Returns (new, previously_selected, available, removed).
    """
    known = set(state_entry.get("known", []))
    selected = set(state_entry.get("selected", []))

    current = set(model_ids)
    new = sorted(current - known)
    previously_selected = sorted(current & selected)
    available = sorted(current - set(new) - selected)
    removed = sorted(known - current)

    return new, previously_selected, available, removed


def interactive_select(
    model_ids: list[str],
    state_entry: dict,
) -> list[str]:
    """Show an interactive multi-select checkbox for model selection."""
    new, previously_selected, available, removed = categorise_models(
        model_ids, state_entry
    )

    if removed:
        print(f"\nRemoved from endpoint since last run: {', '.join(removed)}")

    choices: list[Choice | Separator] = []

    if new:
        choices.append(Separator("--- New models ---"))
        for mid in new:
            ctx = parse_context_from_id(mid)
            ctx_str = f" ({ctx // 1024}k ctx)" if ctx else ""
            choices.append(Choice(mid, name=f"[NEW] {mid}{ctx_str}", enabled=False))

    if previously_selected:
        choices.append(Separator("--- Previously selected ---"))
        for mid in previously_selected:
            ctx = parse_context_from_id(mid)
            ctx_str = f" ({ctx // 1024}k ctx)" if ctx else ""
            choices.append(Choice(mid, name=f"{mid}{ctx_str}", enabled=True))

    if available:
        choices.append(Separator("--- Available ---"))
        for mid in available:
            ctx = parse_context_from_id(mid)
            ctx_str = f" ({ctx // 1024}k ctx)" if ctx else ""
            choices.append(Choice(mid, name=f"{mid}{ctx_str}", enabled=False))

    if not choices:
        print("No models available for selection.")
        return []

    print(f"\n{len(model_ids)} models available. Use space to toggle, enter to confirm.\n")

    selected = inquirer.checkbox(
        message="Select models to add to OpenCode config:",
        choices=choices,
        cycle=True,
        instruction="(space: toggle, a: toggle all, enter: confirm)",
    ).execute()

    return selected or []


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
            ctx = parse_context_from_id(mid)
            ctx_str = f"  context: {ctx:,}" if ctx else "  context: unknown"
            print(f"  {mid}{ctx_str}")
        return

    # Load state
    state = load_state()
    state_entry = state.get(endpoint, {})

    # Model selection
    if args.all:
        selected = model_ids
        print(f"\nSelected all {len(selected)} models.")
    else:
        selected = interactive_select(model_ids, state_entry)

    if not selected:
        print("No models selected.")
        return

    # Build model configs
    models_config = {}
    for mid in selected:
        models_config[mid] = build_model_config(mid, args.default_output)

    # Find and update config
    if args.config:
        config_path = Path(args.config).expanduser()
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}", file=sys.stderr)
            sys.exit(1)
    else:
        config_path = find_config()

    if config_path is None:
        print("\nNo OpenCode config file found.")
        print("Generated config snippet for your provider:\n")
        print(json.dumps(models_config, indent=2))
        # Save state even without config so selections persist
        state[endpoint] = {
            "selected": sorted(selected),
            "known": sorted(model_ids),
        }
        save_state(state)
        return

    print(f"\nUsing config: {config_path}")

    # Parse config to detect provider
    try:
        config_data = json5.loads(config_path.read_text())
    except Exception as exc:
        print(f"Error parsing config: {exc}", file=sys.stderr)
        sys.exit(1)

    provider_id = args.provider_id or detect_provider_id(config_data, endpoint)

    if provider_id is None:
        # Derive a sensible default provider ID from the URL
        parsed_url = urlparse(endpoint)
        hostname = parsed_url.hostname or "local"
        default_id = hostname.split(".")[0].replace("-", "_")
        if default_id in ("localhost", "127", "192", "10"):
            default_id = "local_llm"

        if args.yes:
            # Non-interactive mode: use the default
            provider_id = default_id
            print(f"Auto-assigned provider ID: {provider_id}")
        else:
            provider_id = inquirer.text(
                message="No existing provider found for this endpoint. Enter a provider ID:",
                default=default_id,
            ).execute()
            if not provider_id:
                print("No provider ID given. Aborting.")
                return

    updated = update_config_models(
        config_path,
        provider_id,
        models_config,
        endpoint,
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
