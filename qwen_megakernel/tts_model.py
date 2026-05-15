"""
Weight loading and decode API for Qwen3-TTS-12Hz-0.6B-Base Talker.

The Talker backbone is identical to Qwen3-0.6B in shape:
  - 28 layers, hidden=1024, heads=16, kv_heads=8, head_dim=128
  - intermediate=3072, rope_theta=1000000

Key differences from the text model:
  - vocab_size: 3072 (speech codec tokens) vs 151936 (text tokens)
  - lm_head (talker.codec_head.weight) is NOT tied to embeddings
  - embed table is talker.model.text_embedding.weight
  - rope_theta is 1000000 not 10000
"""

import math
import struct
import torch

NUM_LAYERS = 28
NUM_KV_HEADS = 8
HEAD_DIM = 128
HIDDEN_SIZE = 1024
INTERMEDIATE_SIZE = 3072
Q_SIZE = 16 * HEAD_DIM
KV_SIZE = 8 * HEAD_DIM
MAX_SEQ_LEN = 4096
VOCAB_SIZE = 3072
ROPE_THETA = 1000000.0
LM_NUM_BLOCKS = 12

_decode = torch.ops.qwen_megakernel_C.decode


def load_tts_weights(model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base", verbose=True):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open

    if verbose:
        print(f"Loading TTS Talker weights from {model_name}...")

    sf_path = hf_hub_download(model_name, "model.safetensors")

    needed_prefixes = ("talker.model.", "talker.codec_head.")
    tensors = {}
    with safe_open(sf_path, framework="pt", device="cuda") as f:
        for key in f.keys():
            if any(key.startswith(p) for p in needed_prefixes):
                tensors[key] = f.get_tensor(key).to(torch.bfloat16).contiguous()

    if verbose:
        print(f"Loaded {len(tensors)} talker tensors onto GPU")

    inv_freq = 1.0 / (
        ROPE_THETA ** (torch.arange(0, HEAD_DIM, 2, dtype=torch.float32) / HEAD_DIM)
    )
    positions = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos_table = torch.cos(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()
    sin_table = torch.sin(freqs).repeat(1, 2).to(torch.bfloat16).cuda().contiguous()

    layer_weights = []
    for i in range(NUM_LAYERS):
        p = f"talker.model.layers.{i}."
        layer_weights.extend([
            tensors[p + "input_layernorm.weight"],
            tensors[p + "self_attn.q_proj.weight"],
            tensors[p + "self_attn.k_proj.weight"],
            tensors[p + "self_attn.v_proj.weight"],
            tensors[p + "self_attn.q_norm.weight"],
            tensors[p + "self_attn.k_norm.weight"],
            tensors[p + "self_attn.o_proj.weight"],
            tensors[p + "post_attention_layernorm.weight"],
            tensors[p + "mlp.gate_proj.weight"],
            tensors[p + "mlp.up_proj.weight"],
            tensors[p + "mlp.down_proj.weight"],
        ])

    embed_weight = tensors["talker.model.codec_embedding.weight"]
    lm_head_weight = tensors["talker.codec_head.weight"]

    weights = dict(
        embed_weight=embed_weight,
        layer_weights=layer_weights,
        final_norm_weight=tensors["talker.model.norm.weight"],
        lm_head_weight=lm_head_weight,
        cos_table=cos_table,
        sin_table=sin_table,
    )

    if verbose:
        print(f"  embed_weight:   {embed_weight.shape}")
        print(f"  lm_head_weight: {lm_head_weight.shape}")
        print(f"  cos/sin tables: {cos_table.shape}")
        print("Talker weights ready.")

    return weights


def _pack_layer_weights(layer_weights):
    ptr_size = 8
    n_ptrs = 11
    struct_bytes = n_ptrs * ptr_size
    buf = bytearray(NUM_LAYERS * struct_bytes)
    for i in range(NUM_LAYERS):
        for j in range(n_ptrs):
            ptr = layer_weights[i * n_ptrs + j].data_ptr()
            struct.pack_into("Q", buf, (i * n_ptrs + j) * ptr_size, ptr)
    return torch.frombuffer(buf, dtype=torch.uint8).cuda()


class TTSDecoder:
    """Megakernel-backed decoder for the Qwen3-TTS Talker."""

    def __init__(self, weights=None, model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base", verbose=True):
        if weights is None:
            weights = load_tts_weights(model_name, verbose=verbose)

        self._weights = weights
        self._position = 0

        self._embed_weight = weights["embed_weight"]
        self._final_norm_weight = weights["final_norm_weight"]
        self._lm_head_weight = weights["lm_head_weight"]
        self._cos_table = weights["cos_table"]
        self._sin_table = weights["sin_table"]
        self._layer_weights_packed = _pack_layer_weights(weights["layer_weights"])
        self._attn_scale = 1.0 / math.sqrt(HEAD_DIM)

        self._k_cache = torch.zeros(
            NUM_LAYERS, NUM_KV_HEADS, MAX_SEQ_LEN, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda",
        )
        self._v_cache = torch.zeros_like(self._k_cache)

        bf16 = dict(dtype=torch.bfloat16, device="cuda")
        f32  = dict(dtype=torch.float32,  device="cuda")
        self._hidden    = torch.empty(HIDDEN_SIZE,       **bf16)
        self._act       = torch.empty(HIDDEN_SIZE,       **f32)
        self._res       = torch.empty(HIDDEN_SIZE,       **f32)
        self._q         = torch.empty(Q_SIZE,            **f32)
        self._k         = torch.empty(KV_SIZE,           **f32)
        self._v         = torch.empty(KV_SIZE,           **f32)
        self._attn_out  = torch.empty(Q_SIZE,            **f32)
        self._mlp_inter = torch.empty(INTERMEDIATE_SIZE, **f32)
        self._norm_out  = torch.empty(HIDDEN_SIZE,       **f32)
        self._bmax_vals = torch.empty(LM_NUM_BLOCKS,     **f32)
        self._bmax_idxs = torch.empty(LM_NUM_BLOCKS, dtype=torch.int32, device="cuda")
        self._out_token = torch.empty(1, dtype=torch.int32, device="cuda")

    def step(self, token_id: int) -> int:
        """Run one decode step. Returns next codec token id (0-3071)."""
        _decode(
            self._out_token,
            token_id,
            self._embed_weight,
            self._layer_weights_packed,
            self._final_norm_weight,
            self._lm_head_weight,
            self._cos_table,
            self._sin_table,
            self._k_cache,
            self._v_cache,
            self._hidden,
            self._act,
            self._res,
            self._q,
            self._k,
            self._v,
            self._attn_out,
            self._mlp_inter,
            self._norm_out,
            self._bmax_vals,
            self._bmax_idxs,
            NUM_LAYERS,
            self._position,
            MAX_SEQ_LEN,
            self._attn_scale,
        )
        self._position += 1
        return self._out_token.item()

    def reset(self):
        self._position = 0
        self._k_cache.zero_()
        self._v_cache.zero_()

    @property
    def position(self):
        return self._position
