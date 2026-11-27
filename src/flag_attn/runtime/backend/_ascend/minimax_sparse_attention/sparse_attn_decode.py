# Reserved file slot (task contract: allowed_paths).
#
# Purpose: Ascend-specific MiniMax-M3 MSA decode implementation
# (minimax_m3_sparse_attn_decode and, if useful for the index stage,
# minimax_m3_index_decode helpers). Until a candidate fills this in, the
# Ascend backend re-exports the generic implementation from
# src/flag_attn/minimax_sparse_attention/sparse_attn.py.
#
# A candidate that implements decode here should update
# runtime/backend/_ascend/minimax_sparse_attention/sparse_attn.py to import
# from this module instead of the generic one, keeping the public dispatch
# (flag_attn.minimax_sparse_attention.minimax_m3_sparse_attn_decode) intact.
