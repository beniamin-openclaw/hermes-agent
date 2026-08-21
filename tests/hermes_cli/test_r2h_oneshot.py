"""Focused contract tests for the isolated Hermes R2H one-shot path."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import types

import pytest


def _write_skill(path: Path, *, name: str = "review-system", body: str = "# Review\n") -> Path:
    path.mkdir(parents=True, exist_ok=True)
    skill = path / "SKILL.md"
    skill.write_text(f"---\nname: {name}\n---\n{body}", encoding="utf-8")
    return skill


def _module(name: str, **attrs: object) -> types.ModuleType:
    module = types.ModuleType(name)
    for key, value in attrs.items():
        setattr(module, key, value)
    return module


def _install_fake_agent_stack(monkeypatch: pytest.MonkeyPatch, captured: dict[str, object]) -> None:
    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            captured["agent_kwargs"] = kwargs
            self.suppress_status_output = False
            self.stream_delta_callback = object()
            self.tool_gen_callback = object()

        def run_conversation(self, prompt: str, **kwargs: object) -> dict[str, object]:
            captured["prompt"] = prompt
            captured["inference_kwargs"] = kwargs
            return {"final_response": "ok", "failed": False, "partial": False, "api_calls": 1}

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.config",
        _module("hermes_cli.config", load_config=lambda: {"model": {"default": "configured-model"}}),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_a, **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "fixture-key",
                "base_url": "https://fixture.invalid/v1",
                "provider": "nous",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module(
            "hermes_cli.tools_config",
            _get_platform_tools=lambda *_a, **_k: (_ for _ in ()).throw(
                AssertionError("isolated one-shot must not load configured tools")
            ),
        ),
    )


def _isolated_kwargs(skill_path: Path) -> dict[str, object]:
    return {
        "model": "pinned-model",
        "provider": "nous",
        "skills": ["review-system"],
        "skill_path": str(skill_path.parent),
        "no_tools": True,
        "no_fallback": True,
        "no_dotenv": True,
        "safe_mode": True,
    }


def test_root_parser_propagates_all_isolated_one_shot_flags(tmp_path: Path) -> None:
    from hermes_cli._parser import build_top_level_parser

    parser, _subparsers, _chat = build_top_level_parser()
    args = parser.parse_args(
        [
            "-z",
            "prompt",
            "--no-tools",
            "--no-fallback",
            "--no-dotenv",
            "--safe-mode",
            "--skills",
            "review-system",
            "--skill-path",
            str(tmp_path / "release"),
        ]
    )

    assert args.oneshot == "prompt"
    assert args.no_tools is True
    assert args.no_fallback is True
    assert args.no_dotenv is True
    assert args.safe_mode is True
    assert args.skills == ["review-system"]
    assert args.skill_path == str(tmp_path / "release")


def test_isolated_static_skill_is_inert_and_agent_has_no_tools_or_session_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _write_skill(tmp_path / "release", body="{{ do not evaluate }}\n")
    captured: dict[str, object] = {}
    _install_fake_agent_stack(monkeypatch, captured)

    from hermes_cli.oneshot import _run_agent

    text, result = _run_agent(
        "review prompt",
        **_isolated_kwargs(skill),
    )

    agent_kwargs = captured["agent_kwargs"]
    assert isinstance(agent_kwargs, dict)
    assert agent_kwargs["enabled_toolsets"] == []
    assert agent_kwargs["fallback_model"] == []
    assert agent_kwargs["skip_context_files"] is True
    assert agent_kwargs["skip_memory"] is True
    assert agent_kwargs["session_db"] is None
    inference_kwargs = captured["inference_kwargs"]
    assert isinstance(inference_kwargs, dict)
    system_message = inference_kwargs["system_message"]
    assert isinstance(system_message, str)
    assert "{{ do not evaluate }}" in system_message
    assert result["effective_provider"] == "nous"
    assert result["requested_model"] == "pinned-model"
    assert result["skill_path"] == str(skill.resolve())
    assert len(result["skill_sha256"]) == 64
    assert text == "ok"


def test_real_oneshot_uses_injected_inference_seam_and_isolated_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _write_skill(tmp_path / "release", body="review body\n")
    usage_file = tmp_path / "usage.json"
    captured: dict[str, object] = {}
    _install_fake_agent_stack(monkeypatch, captured)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / "hermes-home"))

    from hermes_cli import oneshot

    def inference_runner(agent: object, prompt: str, system_message: str) -> dict[str, object]:
        captured["runner_agent"] = agent
        captured["runner_prompt"] = prompt
        captured["runner_system_message"] = system_message
        return {"final_response": "fixture response", "api_calls": 0}

    exit_code = oneshot.run_oneshot(
        "fixture prompt",
        usage_file=str(usage_file),
        inference_runner=inference_runner,
        **_isolated_kwargs(skill),
    )

    assert exit_code == 0
    assert captured["runner_prompt"] == "fixture prompt"
    assert "review body" in captured["runner_system_message"]
    report = json.loads(usage_file.read_text(encoding="utf-8"))
    assert report["requested_provider"] == "nous"
    assert report["effective_provider"] == "nous"
    assert report["skill_path"] == str(skill.resolve())


def _fake_chat_response(text: str = "isolated response") -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model="pinned-model",
        choices=[
            types.SimpleNamespace(
                message=types.SimpleNamespace(
                    content=text,
                    tool_calls=None,
                    reasoning=None,
                    reasoning_content=None,
                ),
                finish_reason="stop",
            )
        ],
        usage=types.SimpleNamespace(
            prompt_tokens=7,
            completion_tokens=3,
            total_tokens=10,
        ),
    )


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[object]]:
    import run_agent

    requests: list[dict[str, object]] = []
    constructions: list[dict[str, object]] = []
    closes: list[object] = []

    class FakeCompletions:
        def create(self, **kwargs: object) -> types.SimpleNamespace:
            requests.append(kwargs)
            return _fake_chat_response()

    class FakeOpenAI:
        def __init__(self, **kwargs: object) -> None:
            constructions.append(kwargs)
            self.chat = types.SimpleNamespace(completions=FakeCompletions())

        def close(self) -> None:
            closes.append(self)
            return None

    monkeypatch.setattr(run_agent, "OpenAI", FakeOpenAI)
    return requests, constructions, closes


@pytest.mark.parametrize("oneshot_imports_first", [False, True])
def test_real_isolated_run_conversation_uses_one_fake_transport_request(
    monkeypatch: pytest.MonkeyPatch, oneshot_imports_first: bool
) -> None:
    """The real AIAgent path must isolate without the process-global env gate."""
    import importlib

    monkeypatch.delenv("HERMES_ISOLATED_ONESHOT", raising=False)
    monkeypatch.delenv("HERMES_NO_DOTENV", raising=False)
    monkeypatch.delitem(sys.modules, "model_tools", raising=False)
    monkeypatch.delitem(sys.modules, "run_agent", raising=False)
    monkeypatch.delitem(sys.modules, "hermes_cli.oneshot", raising=False)

    if oneshot_imports_first:
        importlib.import_module("hermes_cli.oneshot")
        run_agent = importlib.import_module("run_agent")
    else:
        run_agent = importlib.import_module("run_agent")
        importlib.import_module("hermes_cli.oneshot")

    requests, constructions, closes = _install_fake_provider(monkeypatch)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("isolated run reached a forbidden subsystem")

    monkeypatch.setattr(run_agent, "get_tool_definitions", forbidden)
    monkeypatch.setattr(run_agent, "get_toolset_for_tool", forbidden)
    monkeypatch.setattr(run_agent, "check_toolset_requirements", forbidden)

    from agent import agent_init, conversation_loop
    import providers

    monkeypatch.setattr(agent_init, "ContextCompressor", forbidden)
    monkeypatch.setattr(agent_init, "StreamingContextScrubber", forbidden)
    monkeypatch.setattr(conversation_loop, "build_turn_context", forbidden)
    monkeypatch.setattr(conversation_loop, "_restore_or_build_system_prompt", forbidden)
    monkeypatch.setattr(providers, "_discover_providers", forbidden)

    agent = run_agent.AIAgent(
        api_key="fixture-key",
        base_url="https://fixture.invalid/v1",
        provider="nous",
        api_mode="chat_completions",
        model="pinned-model",
        enabled_toolsets=[],
        fallback_model=[],
        session_db=None,
        skip_context_files=True,
        skip_memory=True,
        quiet_mode=True,
        isolated_oneshot=True,
    )

    result = agent.run_conversation("review prompt", system_message="static skill")

    assert result["final_response"] == "isolated response"
    assert result["completed"] is True
    assert result["api_calls"] == 1
    assert len(requests) == 1
    assert len(constructions) == 1
    assert len(closes) == 1
    assert "tools" not in requests[0] or requests[0]["tools"] in (None, [])
    assert agent.tools == []
    assert agent._session_db is None
    assert agent.context_compressor is None
    assert os.environ.get("HERMES_ISOLATED_ONESHOT") is None


def test_main_early_dotenv_gate_restores_caller_state_between_dispatches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from hermes_cli import main

    observed: list[str | None] = []

    def fake_load(**_kwargs: object) -> list[str]:
        observed.append(os.environ.get("HERMES_NO_DOTENV"))
        return []

    monkeypatch.delenv("HERMES_NO_DOTENV", raising=False)
    main._load_project_dotenv_with_early_gate(
        argv=["-z", "prompt", "--no-dotenv"],
        load_fn=fake_load,
    )
    assert observed[-1] == "1"
    assert "HERMES_NO_DOTENV" not in os.environ

    main._load_project_dotenv_with_early_gate(argv=["chat"], load_fn=fake_load)
    assert observed[-1] is None
    assert "HERMES_NO_DOTENV" not in os.environ

    monkeypatch.setenv("HERMES_NO_DOTENV", "caller-value")
    main._load_project_dotenv_with_early_gate(
        argv=["-z", "prompt", "--no-dotenv"],
        load_fn=fake_load,
    )
    assert observed[-1] == "1"
    assert os.environ["HERMES_NO_DOTENV"] == "caller-value"


def test_isolated_rejects_non_chat_runtime_before_agent_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _write_skill(tmp_path / "release")
    captured: dict[str, object] = {}
    _install_fake_agent_stack(monkeypatch, captured)

    runtime_provider = sys.modules["hermes_cli.runtime_provider"]
    monkeypatch.setattr(
        runtime_provider,
        "resolve_runtime_provider",
        lambda **_kwargs: {
            "api_key": "fixture-key",
            "base_url": "https://fixture.invalid/v1",
            "provider": "nous",
            "api_mode": "codex_responses",
            "credential_pool": None,
        },
    )

    class MustNotConstruct:
        def __init__(self, **_kwargs: object) -> None:
            raise AssertionError("non-chat isolated mode constructed an agent")

    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=MustNotConstruct))

    from hermes_cli import oneshot

    with pytest.raises(RuntimeError, match="requires api_mode chat_completions"):
        oneshot._run_agent("prompt", **_isolated_kwargs(skill))


def test_isolated_nous_request_policy_matches_profile_without_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import run_agent
    from agent import agent_init, conversation_loop
    import providers

    requests, constructions, closes = _install_fake_provider(monkeypatch)

    def forbidden(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("isolated run reached discovery or context setup")

    monkeypatch.setattr(providers, "_discover_providers", forbidden)
    monkeypatch.setattr(agent_init, "ContextCompressor", forbidden)
    monkeypatch.setattr(agent_init, "StreamingContextScrubber", forbidden)
    monkeypatch.setattr(conversation_loop, "build_turn_context", forbidden)
    monkeypatch.setattr(conversation_loop, "_restore_or_build_system_prompt", forbidden)

    agent = run_agent.AIAgent(
        api_key="fixture-key",
        base_url="https://inference.nousresearch.com/v1",
        provider="nous",
        api_mode="chat_completions",
        model="anthropic/claude-sonnet-4.6",
        fallback_model=[],
        session_db=None,
        skip_context_files=True,
        skip_memory=True,
        isolated_oneshot=True,
    )

    result = agent.run_conversation("review prompt", system_message="static skill")

    from agent.anthropic_adapter import _get_anthropic_max_output
    from agent.portal_tags import nous_portal_tags

    assert result["completed"] is True
    assert len(requests) == len(constructions) == len(closes) == 1
    assert requests[0]["extra_body"]["tags"] == nous_portal_tags()
    assert requests[0]["extra_body"]["reasoning"] == {"enabled": True, "effort": "medium"}
    assert requests[0]["max_tokens"] == _get_anthropic_max_output("anthropic/claude-sonnet-4.6")


@pytest.mark.parametrize("response_shape", ["whitespace", "missing_content"])
def test_isolated_empty_normalized_response_is_failed_with_usage_receipt(
    tmp_path: Path,
    response_shape: str,
) -> None:
    from agent import conversation_loop

    usage = types.SimpleNamespace(prompt_tokens=11, completion_tokens=5, total_tokens=16)
    if response_shape == "whitespace":
        normalized = types.SimpleNamespace(content=" \n\t ", tool_calls=None, usage=usage, reasoning=None)
    else:
        normalized = types.SimpleNamespace(tool_calls=None, usage=usage, reasoning=None)

    class FakeTransport:
        def build_kwargs(self, **kwargs: object) -> dict[str, object]:
            return kwargs

        def normalize_response(self, _response: object) -> object:
            return normalized

    class FakeAgent:
        isolated_oneshot = True
        provider = "nous"
        api_mode = "chat_completions"
        model = "pinned-model"
        base_url = "https://fixture.invalid/v1"
        reasoning_config = None
        max_tokens = None
        request_overrides: dict[str, object] = {}
        session_id = "fixture-session"
        tools: list[object] = []

        def _get_transport(self) -> FakeTransport:
            return FakeTransport()

        def _resolved_api_call_timeout(self) -> float:
            return 30.0

        def _max_tokens_param(self, value: int) -> dict[str, int]:
            return {"max_tokens": value}

        def _interruptible_api_call(self, _kwargs: dict[str, object]) -> object:
            return object()

        def _strip_think_blocks(self, content: str) -> str:
            return content

        def _touch_activity(self, _description: str) -> None:
            return None

    result = conversation_loop.run_conversation(FakeAgent(), "prompt")

    assert result["completed"] is False
    assert result["failed"] is True
    assert result["error"] == "isolated_empty_response"
    assert result["provider"] == "nous"
    assert result["model"] == "pinned-model"
    assert result["input_tokens"] == 11
    assert result["output_tokens"] == 5
    assert result["total_tokens"] == 16

    usage_file = tmp_path / "usage.json"
    from hermes_cli.oneshot import _write_usage_file

    _write_usage_file(str(usage_file), result)
    report = json.loads(usage_file.read_text(encoding="utf-8"))
    assert report["completed"] is False
    assert report["failed"] is True
    assert report["failure"] == "isolated_empty_response"
    assert report["provider"] == "nous"
    assert report["model"] == "pinned-model"
    assert report["total_tokens"] == 16


def test_isolation_is_per_call_and_environment_is_restored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _write_skill(tmp_path / "release")
    captured: list[dict[str, object]] = []

    class FakeAgent:
        def __init__(self, **kwargs: object) -> None:
            captured.append(kwargs)
            self.suppress_status_output = False
            self.stream_delta_callback = None
            self.tool_gen_callback = None

        def run_conversation(self, *_args: object, **_kwargs: object) -> dict[str, object]:
            return {"final_response": "ok", "api_calls": 1}

    _install_fake_agent_stack(monkeypatch, {})
    monkeypatch.setitem(sys.modules, "run_agent", _module("run_agent", AIAgent=FakeAgent))
    keys = (
        "HERMES_NO_DOTENV",
        "HERMES_SAFE_MODE",
        "HERMES_IGNORE_USER_CONFIG",
        "HERMES_IGNORE_RULES",
        "HERMES_YOLO_MODE",
        "HERMES_ACCEPT_HOOKS",
        "HERMES_ISOLATED_ONESHOT",
    )
    before = {key: f"before-{key}" for key in keys}
    for key, value in before.items():
        monkeypatch.setenv(key, value)

    from hermes_cli import oneshot
    monkeypatch.setattr(
        sys.modules["hermes_cli.tools_config"],
        "_get_platform_tools",
        lambda *_args, **_kwargs: [],
    )

    def inference(*_args: object, **_kwargs: object) -> dict[str, object]:
        return {"final_response": "fixture"}

    assert oneshot.run_oneshot("isolated", inference_runner=inference, **_isolated_kwargs(skill)) == 0
    assert oneshot.run_oneshot(
        "ordinary",
        model="ordinary-model",
        provider="nous",
        inference_runner=inference,
    ) == 0
    assert oneshot.run_oneshot("isolated-again", inference_runner=inference, **_isolated_kwargs(skill)) == 0

    assert [bool(call.get("isolated_oneshot")) for call in captured] == [True, False, True]
    assert {key: os.environ.get(key) for key in keys} == before


def test_environment_is_restored_when_isolated_inference_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _write_skill(tmp_path / "release")
    captured: dict[str, object] = {}
    _install_fake_agent_stack(monkeypatch, captured)
    before = {
        "HERMES_NO_DOTENV": "sentinel-no-dotenv",
        "HERMES_SAFE_MODE": "sentinel-safe",
        "HERMES_YOLO_MODE": "sentinel-yolo",
        "HERMES_ACCEPT_HOOKS": "sentinel-hooks",
    }
    for key, value in before.items():
        monkeypatch.setenv(key, value)

    from hermes_cli import oneshot

    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("fixture inference failure")

    assert oneshot.run_oneshot(
        "isolated",
        inference_runner=fail,
        **_isolated_kwargs(skill),
    ) == 1
    assert {key: os.environ.get(key) for key in before} == before


@pytest.mark.parametrize("module_name", ["hermes_cli.main", "run_agent"])
def test_no_dotenv_real_import_subprocess_skips_env_files(
    tmp_path: Path, module_name: str
) -> None:
    hermes_home = tmp_path / "hermes-home"
    hermes_home.mkdir()
    (hermes_home / ".env").write_text(
        "R2H_NO_DOTENV_PROBE=loaded-from-env-file\n", encoding="utf-8"
    )
    env = os.environ.copy()
    env.update(
        {
            "HERMES_HOME": str(hermes_home),
            "HERMES_NO_DOTENV": "1",
            "R2H_NO_DOTENV_PROBE": "shell-value",
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            f"import os; import {module_name}; print(os.environ['R2H_NO_DOTENV_PROBE'])",
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=Path.cwd(),
        check=False,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "shell-value"


def test_real_isolated_agent_skips_context_construction_and_reaches_fake_inference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    skill = _write_skill(tmp_path / "release")
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.models",
        _module("hermes_cli.models", detect_provider_for_model=lambda *_a, **_k: None),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.runtime_provider",
        _module(
            "hermes_cli.runtime_provider",
            resolve_runtime_provider=lambda **_kwargs: {
                "api_key": "fixture-key",
                "base_url": "https://fixture.invalid/v1",
                "provider": "nous",
                "api_mode": "chat_completions",
                "credential_pool": None,
            },
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.tools_config",
        _module("hermes_cli.tools_config", _get_platform_tools=lambda *_a, **_k: []),
    )
    monkeypatch.setenv("HERMES_NO_DOTENV", "1")
    monkeypatch.delitem(sys.modules, "run_agent", raising=False)

    from agent import agent_init
    from hermes_cli import oneshot

    def forbidden_context(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("isolated construction must not create a context engine")

    monkeypatch.setattr(agent_init, "ContextCompressor", forbidden_context)
    captured: dict[str, object] = {}

    def fake_inference(agent: object, prompt: str, system_message: str) -> dict[str, object]:
        captured["agent"] = agent
        captured["prompt"] = prompt
        captured["system_message"] = system_message
        return {"final_response": "fake provider response"}

    text, _result = oneshot._run_agent(
        "prompt",
        **_isolated_kwargs(skill),
        inference_runner=fake_inference,
    )

    assert text == "fake provider response"
    assert captured["prompt"] == "prompt"
    assert captured["agent"].tools == []
    assert captured["agent"].context_compressor is None


@pytest.mark.parametrize(
    ("case", "expected"),
    [
        ("relative", "absolute"),
        ("wrong-name", "review-system"),
        ("malformed", "frontmatter"),
        ("duplicate", "duplicate"),
        ("invalid-utf8", "UTF-8"),
    ],
)
def test_static_skill_validation_fails_before_inference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    case: str,
    expected: str,
) -> None:
    from hermes_cli import oneshot

    skill = _write_skill(tmp_path / "release")
    if case == "relative":
        skill_arg = "relative/SKILL.md"
    elif case == "wrong-name":
        skill = _write_skill(tmp_path / "wrong", name="other")
        skill_arg = str(skill.parent)
    elif case == "malformed":
        skill.write_text("---\nname: review-system\nbody-without-close\n", encoding="utf-8")
        skill_arg = str(skill.parent)
    elif case == "duplicate":
        skill.write_text("---\nname: review-system\nname: other\n---\nbody\n", encoding="utf-8")
        skill_arg = str(skill.parent)
    else:
        skill.write_bytes(b"---\nname: review-system\n---\n\xff\n")
        skill_arg = str(skill.parent)

    called: list[bool] = []
    monkeypatch.setattr(oneshot, "_run_agent", lambda *_a, **_k: called.append(True))

    result = oneshot.run_oneshot(
        "prompt",
        skill_path=skill_arg,
        skills=["review-system"],
        no_tools=True,
        no_fallback=True,
        no_dotenv=True,
        safe_mode=True,
        model="pinned-model",
        provider="nous",
    )

    assert result == 2
    assert called == []
    assert expected.lower() in capsys.readouterr().err.lower()


def test_static_skill_rejects_leaf_symlink_and_ancestor_symlink(tmp_path: Path) -> None:
    from hermes_cli.oneshot import _load_static_skill

    real = _write_skill(tmp_path / "real")
    leaf = tmp_path / "leaf"
    leaf.symlink_to(real.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|unsafe"):
        _load_static_skill(leaf)

    ancestor = tmp_path / "ancestor"
    (tmp_path / "ancestor-real").mkdir()
    (tmp_path / "ancestor-real" / "SKILL.md").write_bytes(real.read_bytes())
    ancestor.symlink_to(tmp_path / "ancestor-real", target_is_directory=True)
    with pytest.raises(ValueError, match="symlink|unsafe"):
        _load_static_skill(ancestor)


def test_static_skill_rejects_file_identity_change_during_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import oneshot

    skill = _write_skill(tmp_path / "release", body="x" * 128)
    original_read = oneshot.os.read
    mutated = False

    def mutate_after_read(fd: int, size: int) -> bytes:
        nonlocal mutated
        data = original_read(fd, size)
        if not mutated:
            mutated = True
            skill.write_text(skill.read_text(encoding="utf-8") + "changed\n", encoding="utf-8")
        return data

    monkeypatch.setattr(oneshot.os, "read", mutate_after_read)
    with pytest.raises(ValueError, match="changed|identity"):
        oneshot._load_static_skill(skill.parent)


def test_static_skill_receipt_does_not_resolve_after_descriptor_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import oneshot

    skill = _write_skill(tmp_path / "release")
    monkeypatch.setattr(
        Path,
        "resolve",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("receipt must not resolve after descriptor close")
        ),
    )

    result = oneshot._load_static_skill(skill.parent)

    assert result.path == str(skill)


def test_static_skill_rejects_ancestor_swap_during_descriptor_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from hermes_cli import oneshot

    skill = _write_skill(tmp_path / "release", body="original\n")
    release = skill.parent
    original_close = oneshot.os.close
    swapped = False

    def swap_after_release_close(fd: int) -> None:
        nonlocal swapped
        try:
            fd_stat = oneshot.os.fstat(fd)
            release_stat = release.stat()
        except OSError:
            fd_stat = None
            release_stat = None
        original_close(fd)
        if not swapped and fd_stat and release_stat and fd_stat.st_ino == release_stat.st_ino:
            replacement = _write_skill(tmp_path / "replacement", body="swapped\n")
            old_release = tmp_path / "old-release"
            release.rename(old_release)
            replacement.parent.rename(release)
            swapped = True

    monkeypatch.setattr(oneshot.os, "close", swap_after_release_close)

    with pytest.raises(ValueError, match="changed|identity|path"):
        oneshot._load_static_skill(release)


def test_usage_receipt_contains_isolation_and_requested_effective_identity(tmp_path: Path) -> None:
    from hermes_cli.oneshot import _write_usage_file

    path = tmp_path / "usage.json"
    _write_usage_file(
        str(path),
        {
            "model": "pinned-model",
            "provider": "nous",
            "requested_model": "pinned-model",
            "effective_model": "pinned-model",
            "requested_provider": "nous",
            "effective_provider": "nous",
            "skill_path": str(tmp_path / "release" / "SKILL.md"),
            "skill_sha256": "a" * 64,
            "isolation_flags": ["no-tools", "no-fallback", "no-dotenv", "safe-mode"],
        },
    )
    report = json.loads(path.read_text(encoding="utf-8"))
    assert report["requested_provider"] == "nous"
    assert report["effective_model"] == "pinned-model"
    assert report["skill_sha256"] == "a" * 64
    assert report["isolation_flags"] == ["no-tools", "no-fallback", "no-dotenv", "safe-mode"]


def test_termux_fast_isolated_dispatch_skips_startup_and_forwards_flags(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import hermes_cli.main as main_mod

    captured: dict[str, object] = {}
    monkeypatch.setenv("TERMUX_VERSION", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "hermes",
            "-z",
            "prompt",
            "--provider",
            "nous",
            "--model",
            "pinned-model",
            "--no-tools",
            "--no-fallback",
            "--no-dotenv",
            "--safe-mode",
            "--skills",
            "review-system",
            "--skill-path",
            str(tmp_path / "release"),
        ],
    )
    monkeypatch.setattr(main_mod, "_prepare_agent_startup", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("startup must be skipped")))
    monkeypatch.setitem(
        sys.modules,
        "hermes_cli.oneshot",
        _module("hermes_cli.oneshot", run_oneshot=lambda prompt, **kwargs: captured.update({"prompt": prompt, **kwargs}) or 0),
    )

    with pytest.raises(SystemExit) as exc:
        main_mod._try_termux_fast_cli_launch()

    assert exc.value.code == 0
    assert captured["prompt"] == "prompt"
    assert captured["no_tools"] is True
    assert captured["no_fallback"] is True
    assert captured["no_dotenv"] is True
    assert captured["safe_mode"] is True
    assert captured["skills"] == ["review-system"]
    assert captured["skill_path"] == str(tmp_path / "release")
