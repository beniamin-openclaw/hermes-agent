"""Nous Portal provider profile."""

from typing import Any

from agent.portal_tags import nous_request_policy
from providers import register_provider
from providers.base import ProviderProfile


class NousProfile(ProviderProfile):
    """Nous Portal — product tags, reasoning with Nous-specific omission."""

    def build_extra_body(
        self, *, session_id: str | None = None, **context
    ) -> dict[str, Any]:
        policy = nous_request_policy(
            model=context.get("model"),
            reasoning_config=context.get("reasoning_config"),
            supports_reasoning=context.get("supports_reasoning", True),
        )
        return {"tags": policy["tags"]}

    def build_api_kwargs_extras(
        self,
        *,
        reasoning_config: dict | None = None,
        supports_reasoning: bool = False,
        **context,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Nous: passes full reasoning_config, but OMITS when disabled."""
        policy = nous_request_policy(
            model=context.get("model"),
            reasoning_config=reasoning_config,
            supports_reasoning=supports_reasoning,
        )
        reasoning = policy["reasoning"]
        return ({"reasoning": reasoning} if reasoning is not None else {}), {}

    def get_max_tokens(self, model: str | None) -> int | None:
        return nous_request_policy(model=model)["max_tokens"]


nous = NousProfile(
    name="nous",
    aliases=("nous-portal", "nousresearch"),
    env_vars=("NOUS_API_KEY",),
    display_name="Nous Research",
    description="Nous Research — Hermes model family",
    signup_url="https://nousresearch.com/",
    fallback_models=(
        "hermes-3-405b",
        "hermes-3-70b",
    ),
    base_url="https://inference.nousresearch.com/v1",
    auth_type="oauth_device_code",
)

register_provider(nous)
