"""Oneshot (-z) mode: send a prompt, get the final content block, exit.

Bypasses cli.py entirely.  No banner, no spinner, no session_id line,
no stderr chatter.  Just the agent's final text to stdout.

Toolsets = explicit --toolsets when provided, otherwise whatever the user has
configured for "cli" in `hermes tools`.
Rules / memory / AGENTS.md / preloaded skills = same as a normal chat turn.
Approvals = auto-bypassed (HERMES_YOLO_MODE=1 is set for the call).
Working directory = the user's CWD (AGENTS.md etc. resolve from there as usual).

Model / provider selection mirrors `hermes chat`:
    - Both optional. If omitted, use the user's configured default.
    - If both given, pair them exactly as given.
    - If only --model given, auto-detect the provider that serves it.
    - If only --provider given, error out (ambiguous — caller must pick a model).

Env var fallbacks (used when the corresponding arg is not passed):
    - HERMES_INFERENCE_MODEL
"""

from __future__ import annotations

import logging
import hashlib
import os
import stat
import sys
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from hermes_cli.fallback_config import get_fallback_chain


MAX_STATIC_SKILL_BYTES = 1024 * 1024


@dataclass(frozen=True)
class _StaticSkill:
    path: str
    sha256: str
    text: str
    system_message: str


def _skill_error(message: str) -> ValueError:
    return ValueError(f"hermes -z: invalid isolated skill: {message}")


def _same_identity(left: os.stat_result, right: os.stat_result) -> bool:
    return (left.st_dev, left.st_ino) == (right.st_dev, right.st_ino)


def _read_static_skill_bytes(release_path: Path) -> bytes:
    """Read ``release_path/SKILL.md`` without following path components."""
    if not release_path.is_absolute():
        raise _skill_error("release path must be absolute")
    if any(part in {".", ".."} for part in release_path.parts):
        raise _skill_error("release path must not contain dot components")
    if release_path == Path("/"):
        raise _skill_error("release path must name a release directory")

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd: int | None = None
    try:
        directory_fd = os.open("/", flags | nofollow)
        for component in release_path.parts[1:]:
            before = os.stat(component, dir_fd=directory_fd, follow_symlinks=False)
            if stat.S_ISLNK(before.st_mode) or not stat.S_ISDIR(before.st_mode):
                raise _skill_error("release path contains a symlink or non-directory")
            next_fd = os.open(component, flags | nofollow, dir_fd=directory_fd)
            after = os.fstat(next_fd)
            if not _same_identity(before, after):
                os.close(next_fd)
                raise _skill_error("release directory changed during open")
            os.close(directory_fd)
            directory_fd = next_fd

        skill_before = os.stat("SKILL.md", dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISLNK(skill_before.st_mode) or not stat.S_ISREG(skill_before.st_mode):
            raise _skill_error("SKILL.md must be a regular non-symlink file")
        if skill_before.st_size > MAX_STATIC_SKILL_BYTES:
            raise _skill_error("SKILL.md exceeds the size limit")

        skill_fd = os.open("SKILL.md", os.O_RDONLY | nofollow, dir_fd=directory_fd)
        try:
            opened = os.fstat(skill_fd)
            if not _same_identity(skill_before, opened):
                raise _skill_error("SKILL.md changed during open")
            chunks: list[bytes] = []
            total = 0
            while True:
                chunk = os.read(skill_fd, MAX_STATIC_SKILL_BYTES + 1 - total)
                if not chunk:
                    break
                chunks.append(chunk)
                total += len(chunk)
                if total > MAX_STATIC_SKILL_BYTES:
                    raise _skill_error("SKILL.md exceeds the size limit")
            raw = b"".join(chunks)
            closed_stat = os.fstat(skill_fd)
            path_stat = os.stat("SKILL.md", dir_fd=directory_fd, follow_symlinks=False)
            if (
                not _same_identity(opened, closed_stat)
                or not _same_identity(opened, path_stat)
                or opened.st_size != closed_stat.st_size
                or opened.st_mtime_ns != closed_stat.st_mtime_ns
                or opened.st_ctime_ns != closed_stat.st_ctime_ns
                or opened.st_mtime_ns != path_stat.st_mtime_ns
                or opened.st_ctime_ns != path_stat.st_ctime_ns
                or closed_stat.st_size != len(raw)
            ):
                raise _skill_error("SKILL.md changed during read")
            return raw
        finally:
            os.close(skill_fd)
    except FileNotFoundError as exc:
        raise _skill_error("release path or SKILL.md does not exist") from exc
    except OSError as exc:
        if isinstance(exc, ValueError):
            raise
        raise _skill_error("release path is not safely accessible") from exc
    finally:
        if directory_fd is not None:
            os.close(directory_fd)


def _frontmatter_value(raw_text: str) -> str:
    lines = raw_text.splitlines(keepends=True)
    if not lines or lines[0].rstrip("\r\n") != "---":
        raise _skill_error("frontmatter must begin with ---")
    closing = None
    for index, line in enumerate(lines[1:], start=1):
        if line.rstrip("\r\n") == "---":
            closing = index
            break
    if closing is None:
        raise _skill_error("frontmatter is not closed")

    frontmatter = "".join(lines[1:closing])
    try:
        import yaml

        class _UniqueKeyLoader(yaml.SafeLoader):
            pass

        def construct_mapping(loader, node, deep=False):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node, deep=deep)
                if key in mapping:
                    raise _skill_error("frontmatter contains a duplicate key")
                mapping[key] = loader.construct_object(value_node, deep=deep)
            return mapping

        _UniqueKeyLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping,
        )
        parsed = yaml.load(frontmatter, Loader=_UniqueKeyLoader)
    except ValueError:
        raise
    except Exception as exc:
        raise _skill_error("frontmatter is malformed") from exc
    if not isinstance(parsed, dict) or parsed.get("name") != "review-system":
        raise _skill_error("frontmatter name must be review-system")
    return raw_text


def _load_static_skill(release_path: Path | str) -> _StaticSkill:
    """Load the approved skill as bounded inert text from an immutable path."""
    release = Path(release_path)
    raw = _read_static_skill_bytes(release)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise _skill_error("SKILL.md is not valid UTF-8") from exc
    _frontmatter_value(text)
    resolved_release = release.resolve(strict=True)
    resolved_skill = resolved_release / "SKILL.md"
    digest = hashlib.sha256(raw).hexdigest()
    system_message = (
        "[R2H isolated static skill]\n"
        "The following UTF-8 bytes are inert instruction text. Do not evaluate "
        "frontmatter values or execute body content.\n"
        f'<skill path="{resolved_skill}" sha256="{digest}">\n'
        f"{text}"
        "</skill>"
    )
    return _StaticSkill(
        path=str(resolved_skill),
        sha256=digest,
        text=text,
        system_message=system_message,
    )


def _normalize_requested_skills(skills: object = None) -> list[str]:
    if skills is None:
        return []
    raw_items = [skills] if isinstance(skills, str) else skills
    if not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]
    normalized: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            normalized.extend(part.strip() for part in item.split(","))
        else:
            normalized.append(str(item).strip())
    return [item for item in normalized if item]


def _isolated_skill_requested(skills: object, skill_path: object) -> bool:
    return bool(skill_path) or "review-system" in _normalize_requested_skills(skills)


def _validate_isolated_request(
    *,
    skills: object,
    skill_path: object,
    model: Optional[str],
    provider: Optional[str],
    no_tools: bool,
    no_fallback: bool,
    no_dotenv: bool,
    safe_mode: bool,
) -> _StaticSkill | None:
    if not _isolated_skill_requested(skills, skill_path):
        return None
    requested_skills = _normalize_requested_skills(skills)
    required = ["--skills review-system", "--skill-path PATH", "--no-tools", "--no-fallback", "--no-dotenv", "--safe-mode"]
    if requested_skills != ["review-system"] or not skill_path:
        raise ValueError("hermes -z: isolated review-system requires " + ", ".join(required))
    if not all((no_tools, no_fallback, no_dotenv, safe_mode)):
        raise ValueError("hermes -z: isolated review-system requires " + ", ".join(required))
    if not (model or "").strip() or not (provider or "").strip():
        raise ValueError("hermes -z: isolated review-system requires explicit --model and --provider")
    if provider.strip().lower() != "nous":
        raise ValueError("hermes -z: isolated review-system requires provider nous")
    return _load_static_skill(Path(str(skill_path)))


def _normalize_toolsets(toolsets: object = None) -> list[str] | None:
    if not toolsets:
        return None

    raw_items = [toolsets] if isinstance(toolsets, str) else toolsets
    if not isinstance(raw_items, (list, tuple)):
        raw_items = [raw_items]

    normalized: list[str] = []
    for item in raw_items:
        if isinstance(item, str):
            normalized.extend(part.strip() for part in item.split(","))
        else:
            normalized.append(str(item).strip())

    return [item for item in normalized if item] or None


def _validate_explicit_toolsets(toolsets: object = None) -> tuple[list[str] | None, str | None]:
    normalized = _normalize_toolsets(toolsets)
    if normalized is None:
        return None, None

    try:
        from toolsets import validate_toolset
    except Exception as exc:
        return None, f"hermes -z: failed to validate --toolsets: {exc}\n"

    built_in = [name for name in normalized if validate_toolset(name)]
    unresolved = [name for name in normalized if name not in built_in]

    if unresolved:
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins()
            plugin_valid = [name for name in unresolved if validate_toolset(name)]
        except Exception:
            plugin_valid = []

        if plugin_valid:
            built_in.extend(plugin_valid)
            unresolved = [name for name in unresolved if name not in plugin_valid]

    if any(name in {"all", "*"} for name in built_in):
        ignored = [name for name in normalized if name not in {"all", "*"}]
        if ignored:
            sys.stderr.write(
                "hermes -z: --toolsets all enables every toolset; "
                f"ignoring additional entries: {', '.join(ignored)}\n"
            )
        return None, None

    mcp_names: set[str] = set()
    mcp_disabled: set[str] = set()
    if unresolved:
        try:
            from hermes_cli.config import read_raw_config
            from hermes_cli.tools_config import _parse_enabled_flag

            cfg = read_raw_config()
            mcp_servers = cfg.get("mcp_servers") if isinstance(cfg.get("mcp_servers"), dict) else {}
            for name, server_cfg in mcp_servers.items():
                if not isinstance(server_cfg, dict):
                    continue
                if _parse_enabled_flag(server_cfg.get("enabled", True), default=True):
                    mcp_names.add(str(name))
                else:
                    mcp_disabled.add(str(name))
        except Exception:
            mcp_names = set()
            mcp_disabled = set()

    mcp_valid = [name for name in unresolved if name in mcp_names]
    disabled = [name for name in unresolved if name in mcp_disabled]
    unknown = [name for name in unresolved if name not in mcp_names and name not in mcp_disabled]
    valid = built_in + mcp_valid

    if unknown:
        sys.stderr.write(f"hermes -z: ignoring unknown --toolsets entries: {', '.join(unknown)}\n")
    if disabled:
        sys.stderr.write(
            "hermes -z: ignoring disabled MCP servers (set enabled: true in config.yaml to use): "
            f"{', '.join(disabled)}\n"
        )

    if not valid:
        return None, "hermes -z: --toolsets did not contain any valid toolsets.\n"

    return valid, None


def _write_usage_file(path: Optional[str], result: dict, failure: Optional[str] = None) -> None:
    """Best-effort JSON usage report for pipelines (``-z --usage-file``).

    Written even on failure so callers can always account for spend. Never
    raises — a broken usage write must not mask the run's own outcome.
    """
    if not path:
        return
    try:
        import json

        report = {
            "estimated_cost_usd": result.get("estimated_cost_usd"),
            "cost_status": result.get("cost_status"),
            "cost_source": result.get("cost_source"),
            "input_tokens": result.get("input_tokens"),
            "output_tokens": result.get("output_tokens"),
            "cache_read_tokens": result.get("cache_read_tokens"),
            "cache_write_tokens": result.get("cache_write_tokens"),
            "reasoning_tokens": result.get("reasoning_tokens"),
            "total_tokens": result.get("total_tokens"),
            "api_calls": result.get("api_calls"),
            "model": result.get("model"),
            "provider": result.get("provider"),
            "requested_model": result.get("requested_model"),
            "effective_model": result.get("effective_model"),
            "requested_provider": result.get("requested_provider"),
            "effective_provider": result.get("effective_provider"),
            "skill_path": result.get("skill_path"),
            "skill_sha256": result.get("skill_sha256"),
            "isolation_flags": result.get("isolation_flags"),
            "session_id": result.get("session_id"),
            "completed": result.get("completed"),
            "failed": bool(result.get("failed")) or failure is not None,
        }
        if failure is not None:
            report["failure"] = failure
        out = Path(path).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception:
        pass


def run_oneshot(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: object = None,
    usage_file: Optional[str] = None,
    skills: object = None,
    skill_path: Optional[str] = None,
    no_tools: bool = False,
    no_fallback: bool = False,
    no_dotenv: bool = False,
    safe_mode: bool = False,
    inference_runner: Optional[Callable[..., Any]] = None,
) -> int:
    """Execute a single prompt and print only the final content block.

    Args:
        prompt: The user message to send.
        model: Optional model override. Falls back to HERMES_INFERENCE_MODEL
            env var, then config.yaml's model.default / model.model.
        provider: Optional provider override. Falls back to config.yaml's
            model.provider, then "auto".
        toolsets: Optional comma-separated string or iterable of toolsets.
        usage_file: Optional path; when set, a JSON usage report (estimated
            cost, token counts, model, api_calls) is written there after the
            run — even when the run fails — so pipelines can account for
            spend per invocation.

    Returns the exit code.  Caller should sys.exit() with the return.
    """
    # Silence every stdlib logger for the duration.  AIAgent, tools, and
    # provider adapters all log to stderr through the root logger; file
    # handlers added by setup_logging() keep working (they're attached to
    # the root logger's handler list, not affected by level), but no
    # bytes reach the terminal.
    logging.disable(logging.CRITICAL)

    if no_dotenv:
        os.environ["HERMES_NO_DOTENV"] = "1"
    if safe_mode:
        os.environ["HERMES_SAFE_MODE"] = "1"
        os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"
        os.environ["HERMES_IGNORE_RULES"] = "1"

    try:
        static_skill = _validate_isolated_request(
            skills=skills,
            skill_path=skill_path,
            model=model,
            provider=provider,
            no_tools=no_tools,
            no_fallback=no_fallback,
            no_dotenv=no_dotenv,
            safe_mode=safe_mode,
        )
    except (ValueError, OSError) as exc:
        message = str(exc)
        _write_usage_file(usage_file, {}, failure=message)
        sys.stderr.write(f"{message}\n")
        return 2

    # --provider without --model is ambiguous: carrying the user's configured
    # model across to a different provider is usually wrong (that provider may
    # not host it), and silently picking the provider's catalog default hides
    # the mismatch.  Require the caller to be explicit.  Validate BEFORE the
    # stderr redirect so the message actually reaches the terminal.
    env_model_early = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    if provider and not ((model or "").strip() or env_model_early):
        sys.stderr.write(
            "hermes -z: --provider requires --model (or HERMES_INFERENCE_MODEL). "
            "Pass both explicitly, or neither to use your configured defaults.\n"
        )
        return 2

    if no_tools:
        if _normalize_toolsets(toolsets) is not None:
            message = "hermes -z: --no-tools cannot be combined with --toolsets"
            _write_usage_file(usage_file, {}, failure=message)
            sys.stderr.write(f"{message}\n")
            return 2
        explicit_toolsets = []
        use_config_toolsets = False
    else:
        explicit_toolsets, toolsets_error = _validate_explicit_toolsets(toolsets)
        if toolsets_error:
            sys.stderr.write(toolsets_error)
            return 2
        use_config_toolsets = _normalize_toolsets(toolsets) is None

    # Auto-approve any shell / tool approvals.  Non-interactive by
    # definition — a prompt would hang forever.
    os.environ["HERMES_YOLO_MODE"] = "1"
    os.environ["HERMES_ACCEPT_HOOKS"] = "1"

    # Redirect stderr AND stdout to devnull for the entire call tree.
    # We'll print the final response to the real stdout at the end.
    real_stdout = sys.stdout
    real_stderr = sys.stderr
    devnull = open(os.devnull, "w", encoding="utf-8")

    response: Optional[str] = None
    result: dict = {}
    failure: BaseException | None = None
    try:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            try:
                response, result = _run_agent(
                    prompt,
                    model=model,
                    provider=provider,
                    toolsets=explicit_toolsets,
                    use_config_toolsets=use_config_toolsets,
                    skills=skills,
                    skill_path=skill_path,
                    no_tools=no_tools,
                    no_fallback=no_fallback,
                    no_dotenv=no_dotenv,
                    safe_mode=safe_mode,
                    inference_runner=inference_runner,
                    _static_skill=static_skill,
                )
            except BaseException as exc:  # noqa: BLE001
                # Capture anything that escapes the agent (including OSError
                # from prompt_toolkit/Vt100 when stdout is a non-TTY pipe,
                # KeyboardInterrupt, SystemExit, etc.) so we can surface it on
                # the real stderr instead of crashing past the redirect with a
                # traceback that the caller never sees. A silent exit in a
                # cron / SSH / subprocess context is the worst failure mode.
                # See #30623.
                failure = exc
    finally:
        try:
            devnull.close()
        except Exception:
            pass

    if failure is not None:
        # Re-raise control-flow exceptions so the parent handles them as usual
        # (Ctrl-C / explicit sys.exit() inside the agent).
        if isinstance(failure, (KeyboardInterrupt, SystemExit)):
            _write_usage_file(usage_file, result, failure=repr(failure))
            raise failure
        _write_usage_file(usage_file, result, failure=str(failure))
        real_stderr.write(f"hermes -z: agent failed: {failure}\n")
        real_stderr.flush()
        return 1

    _write_usage_file(usage_file, result)

    if response:
        real_stdout.write(response)
        if not response.endswith("\n"):
            real_stdout.write("\n")
        real_stdout.flush()

    if (result.get("failed") or result.get("partial")) and not (response or "").strip():
        return 2

    if not (response or "").strip():
        real_stderr.write("hermes -z: no final response was produced; treating the run as failed.\n")
        real_stderr.flush()
        return 1

    return 0


def _create_session_db_for_oneshot():
    """Best-effort SessionDB for ``hermes -z`` / oneshot mode.

    Oneshot bypasses ``HermesCLI._init_agent()``, so it must wire the SQLite
    session store itself. Without this, the ``session_search``/recall tool is
    advertised but every call returns "Session database not available.".
    """
    try:
        from hermes_state import SessionDB

        return SessionDB()
    except Exception as exc:
        logging.debug("SQLite session store not available for oneshot mode: %s", exc)
        return None


def _run_agent(
    prompt: str,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    toolsets: object = None,
    use_config_toolsets: bool = True,
    skills: object = None,
    skill_path: Optional[str] = None,
    no_tools: bool = False,
    no_fallback: bool = False,
    no_dotenv: bool = False,
    safe_mode: bool = False,
    inference_runner: Optional[Callable[..., Any]] = None,
    _static_skill: _StaticSkill | None = None,
) -> tuple[str, dict]:
    """Build an AIAgent exactly like a normal CLI chat turn would, then
    run a single conversation.  Returns ``(final_response, run_result)``."""
    if no_dotenv:
        os.environ["HERMES_NO_DOTENV"] = "1"
    if safe_mode:
        os.environ["HERMES_SAFE_MODE"] = "1"
        os.environ["HERMES_IGNORE_USER_CONFIG"] = "1"
        os.environ["HERMES_IGNORE_RULES"] = "1"

    static_skill = _static_skill or _validate_isolated_request(
        skills=skills,
        skill_path=skill_path,
        model=model,
        provider=provider,
        no_tools=no_tools,
        no_fallback=no_fallback,
        no_dotenv=no_dotenv,
        safe_mode=safe_mode,
    )
    isolated = static_skill is not None

    # Imports are local so they don't run when hermes is invoked for
    # other commands (keeps top-level CLI startup cheap).
    from hermes_cli.config import load_config
    from hermes_cli.models import detect_provider_for_model
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.tools_config import _get_platform_tools
    from run_agent import AIAgent

    cfg = {} if (isolated or safe_mode) else load_config()

    # Resolve effective model: explicit arg → env var → config.
    model_cfg = cfg.get("model") or {}
    if isinstance(model_cfg, str):
        cfg_model = model_cfg
    else:
        cfg_model = model_cfg.get("default") or model_cfg.get("model") or ""

    env_model = os.getenv("HERMES_INFERENCE_MODEL", "").strip()
    effective_model = (model or "").strip() or env_model or cfg_model

    # Resolve effective provider: explicit arg → (auto-detect from model if
    # model was explicit) → env / config (handled inside resolve_runtime_provider).
    #
    # When --model is given without --provider, auto-detect the provider that
    # serves that model — same semantic as `/model <name>` in an interactive
    # session.  Without this, resolve_runtime_provider() would fall back to
    # the user's configured default provider, which may not host the model
    # the caller just asked for.
    effective_provider = (provider or "").strip() or None
    explicit_base_url_from_alias: Optional[str] = None
    if effective_provider is None and (model or env_model):
        # Only auto-detect when the model was explicitly requested via arg or
        # env var (not when it came from config — that's the "use my defaults"
        # path and the configured provider is already correct).
        explicit_model = (model or "").strip() or env_model
        if explicit_model:
            # First check DIRECT_ALIASES populated from config.yaml `model_aliases:`.
            # These map a user-defined alias to (model, provider, base_url) for
            # endpoints not in any catalog (local servers, custom proxies, etc.).
            try:
                from hermes_cli import model_switch as _ms
                _ms._ensure_direct_aliases()
                direct = _ms.DIRECT_ALIASES.get(explicit_model.strip().lower())
            except Exception:
                direct = None
            if direct is not None:
                effective_model = direct.model
                effective_provider = direct.provider
                if direct.base_url:
                    explicit_base_url_from_alias = direct.base_url.rstrip("/")
            else:
                cfg_provider = ""
                if isinstance(model_cfg, dict):
                    cfg_provider = str(model_cfg.get("provider") or "").strip().lower()
                current_provider = (
                    cfg_provider
                    or os.getenv("HERMES_INFERENCE_PROVIDER", "").strip().lower()
                    or "auto"
                )
                detected = detect_provider_for_model(explicit_model, current_provider)
                if detected:
                    effective_provider, effective_model = detected

    runtime = resolve_runtime_provider(
        requested=effective_provider,
        target_model=effective_model or None,
        explicit_base_url=explicit_base_url_from_alias,
    )
    if isolated:
        runtime_provider = str(runtime.get("provider") or "").strip().lower()
        if runtime_provider != (provider or "").strip().lower():
            raise RuntimeError("hermes -z: effective provider does not match requested provider")
        if effective_model != (model or "").strip():
            raise RuntimeError("hermes -z: effective model does not match requested model")

    # Pull in explicit toolsets when provided; otherwise use whatever the user
    # has enabled for "cli". sorted() gives stable ordering for config-derived
    # sets; explicit values preserve user order.
    toolsets_list = [] if (no_tools or isolated or safe_mode) else _normalize_toolsets(toolsets)
    if toolsets_list is None and use_config_toolsets and not safe_mode:
        toolsets_list = sorted(_get_platform_tools(cfg, "cli"))

    session_db = None if (isolated or safe_mode) else _create_session_db_for_oneshot()
    # Read the effective fallback chain from profile config so oneshot workers
    # honour the same merge semantics as interactive CLI and gateway sessions.
    _fb = [] if (no_fallback or isolated or safe_mode) else get_fallback_chain(cfg)

    agent = AIAgent(
        api_key=runtime.get("api_key"),
        base_url=runtime.get("base_url"),
        provider=runtime.get("provider"),
        api_mode=runtime.get("api_mode"),
        model=effective_model,
        enabled_toolsets=toolsets_list,
        quiet_mode=True,
        platform="cli",
        session_db=session_db,
        credential_pool=runtime.get("credential_pool"),
        fallback_model=_fb if (no_fallback or isolated or safe_mode) else (_fb or None),
        skip_context_files=isolated or safe_mode,
        skip_memory=isolated or safe_mode,
        # Interactive callbacks are intentionally NOT wired beyond this
        # one.  In oneshot mode there's no user sitting at a terminal:
        #   - clarify  → returns a synthetic "pick a default" instruction
        #                so the agent continues instead of stalling on
        #                the tool's built-in "not available" error
        #   - sudo password prompt → terminal_tool gates on
        #                HERMES_INTERACTIVE which we never set
        #   - shell-hook approval → auto-approved via HERMES_ACCEPT_HOOKS=1
        #                (set above); also falls back to deny on non-tty
        #   - dangerous-command approval → bypassed via HERMES_YOLO_MODE=1
        #   - skill secret capture → returns gracefully when no callback set
        clarify_callback=_oneshot_clarify_callback,
    )

    # Belt-and-braces: make sure AIAgent doesn't invoke any streaming
    # display callbacks that would bypass our stdout capture.
    agent.suppress_status_output = True
    agent.stream_delta_callback = None
    agent.tool_gen_callback = None

    if inference_runner is None:
        result = agent.run_conversation(
            prompt,
            system_message=static_skill.system_message if static_skill else None,
        )
    else:
        result = inference_runner(
            agent,
            prompt,
            static_skill.system_message if static_skill else None,
        )
    if not isinstance(result, dict):
        result = {"final_response": str(result)}
    result.setdefault("model", effective_model)
    result.setdefault("provider", runtime.get("provider"))
    result["requested_model"] = (model or "").strip() or None
    result["effective_model"] = effective_model
    result["requested_provider"] = (provider or "").strip() or None
    result["effective_provider"] = runtime.get("provider")
    if static_skill:
        result["skill_path"] = static_skill.path
        result["skill_sha256"] = static_skill.sha256
    result["isolation_flags"] = [
        flag
        for flag, enabled in (
            ("no-tools", no_tools),
            ("no-fallback", no_fallback),
            ("no-dotenv", no_dotenv),
            ("safe-mode", safe_mode),
        )
        if enabled
    ] + (["skill-path"] if static_skill else [])
    return (result.get("final_response") or "", result)


def _oneshot_clarify_callback(question: str, choices=None) -> str:
    """Clarify is disabled in oneshot mode — tell the agent to pick a
    default and proceed instead of stalling or erroring."""
    if choices:
        return (
            f"[oneshot mode: no user available. Pick the best option from "
            f"{choices} using your own judgment and continue.]"
        )
    return (
        "[oneshot mode: no user available. Make the most reasonable "
        "assumption you can and continue.]"
    )
