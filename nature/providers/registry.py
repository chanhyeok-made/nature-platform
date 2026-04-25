"""Provider registry — discover and instantiate LLM providers by name."""

from __future__ import annotations

from typing import Callable

from nature.protocols.provider import LLMProvider, ProviderConfig

# Factory: (config) -> LLMProvider
ProviderFactory = Callable[[ProviderConfig], LLMProvider]


class ProviderRegistry:
    """Registry mapping provider names to factory functions.

    Usage:
        registry = ProviderRegistry()
        registry.register("anthropic", AnthropicProvider)
        provider = registry.create("anthropic", config)
    """

    def __init__(self) -> None:
        self._factories: dict[str, ProviderFactory] = {}

    def register(self, name: str, factory: ProviderFactory) -> None:
        """Register a provider factory."""
        self._factories[name] = factory

    def create(self, name: str, config: ProviderConfig) -> LLMProvider:
        """Create a provider instance by name."""
        if name not in self._factories:
            available = ", ".join(sorted(self._factories)) or "(none)"
            raise KeyError(f"Unknown provider: {name!r}. Available: {available}")
        return self._factories[name](config)

    @property
    def available(self) -> list[str]:
        """List of registered provider names."""
        return sorted(self._factories)


# Global default registry
_default_registry = ProviderRegistry()


def get_provider(name: str, config: ProviderConfig) -> LLMProvider:
    """Get a provider from the default registry."""
    return _default_registry.create(name, config)


def register_provider(name: str, factory: ProviderFactory) -> None:
    """Register a provider in the default registry."""
    _default_registry.register(name, factory)
