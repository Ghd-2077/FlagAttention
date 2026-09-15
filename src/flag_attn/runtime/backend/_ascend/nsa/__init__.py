"""Ascend NSA selected-attention forward implementation."""


def parallel_nsa_fwd(*args, **kwargs):
    from .forward import parallel_nsa_fwd as forward

    return forward(*args, **kwargs)


__all__ = ["parallel_nsa_fwd"]
