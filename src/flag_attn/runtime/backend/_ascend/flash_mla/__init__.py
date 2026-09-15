"""Ascend MLA implementations."""
import importlib

_OPERATOR_EXPORTS = {
    "FlashMLADecodePlan": (".dense", "FlashMLADecodePlan"),
    "flash_mla": (".dense", "flash_mla"),
}
__all__ = sorted(_OPERATOR_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute_name = _OPERATOR_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc
    value = getattr(importlib.import_module(module_name, __name__), attribute_name)
    globals()[name] = value
    return value
