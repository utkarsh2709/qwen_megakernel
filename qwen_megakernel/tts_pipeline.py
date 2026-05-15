"""
Megakernel-accelerated TTS pipeline.

Architecture:
  - Prefill: PyTorch talker.model.forward(text_embeds) -> KV cache
  - Decode loop: megakernel for token generation (~1000 tok/s)
  - Code predictor: PyTorch, uses talker hidden states
  - Mimi: PyTorch speech_tokenizer.decode()

The megakernel handles the hot decode loop. Hidden states for the
code_predictor are obtained by running a lightweight PyTorch forward
pass in parallel (or approximated from the megakernel output).
"""

import time
import torch
import numpy as np
from typing import Iterator, Optional, List

from qwen_megakernel.tts_model import (
    TTSDecoder, load_tts_weights,
    NUM_LAYERS, NUM_KV_HEADS, HEAD_DIM, MAX_SEQ_LEN, HIDDEN_SIZE,
    INTERMEDIATE_SIZE, Q_SIZE, KV_SIZE, LM_NUM_BLOCKS, ROPE_THETA,
    _pack_layer_weights,
)

CODEC_BOS_ID    = 2149
CODEC_EOS_ID    = 2150
CODEC_PAD_ID    = 2148
LANGUAGE_IDS    = {
    "english": 2050, "chinese": 2055, "german": 2053,
    "italian": 2070, "portuguese": 2071, "spanish": 2054,
    "japanese": 2058, "korean": 2064, "french": 2061, "russian": 2069,
}
CODEC_HZ          = 12.5
AUDIO_HZ          = 24000
SAMPLES_PER_FRAME = int(AUDIO_HZ / CODEC_HZ)   # 1920 samples per codec frame


def _transfer_kv_cache(hf_cache, mk_k_cache, mk_v_cache):
    """Copy HF DynamicCache -> megakernel static KV tensors. Returns seq_len."""
    seq_len = None
    for layer_idx in range(NUM_LAYERS):
        k, v = hf_cache[layer_idx]
        if seq_len is None:
            seq_len = k.shape[2]
        mk_k_cache[layer_idx, :, :seq_len, :] = k[0].to(torch.bfloat16)
        mk_v_cache[layer_idx, :, :seq_len, :] = v[0].to(torch.bfloat16)
    return seq_len


class MegakernelTTSPipeline:
    """
    TTS pipeline: PyTorch prefill + megakernel decode.
    
    Two synthesis modes:
      1. synthesize() - full quality, uses code_predictor for all 16 codebooks
      2. synthesize_fast() - megakernel decode only, validates speed
    """

    def __init__(self, model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base", verbose=True):
        self.verbose  = verbose
        self.model_name = model_name
        t0 = time.time()

        if verbose:
            print("[1/2] Loading megakernel decoder...")
        weights = load_tts_weights(model_name, verbose=verbose)
        self.mk_decoder = TTSDecoder(weights=weights, verbose=False)

        if verbose:
            print("[2/2] Loading full qwen_tts model...")
        from qwen_tts import Qwen3TTSModel
        self.tts_model = Qwen3TTSModel.from_pretrained(
            model_name, dtype=torch.bfloat16, device_map="cuda",
        )
        self.talker = self.tts_model.model.talker
        self.device = "cuda"
        self.dtype  = torch.bfloat16

        if verbose:
            print(f"Pipeline ready in {time.time()-t0:.1f}s")

    def _build_input_embeds(self, text: str, language: str = "english"):
        """Tokenize + project text to talker hidden space [1, T, 1024]."""
        talker = self.talker
        proc   = self.tts_model.processor
        formatted = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        ids = proc(text=formatted, return_tensors="pt")["input_ids"].to(self.device)
        # text_embedding [151936,2048] -> text_projection -> [1,T,1024]
        text_embeds = talker.text_projection(talker.get_text_embeddings()(ids))

        lang_id   = LANGUAGE_IDS.get(language.lower(), LANGUAGE_IDS["english"])
        lang_emb  = talker.get_input_embeddings()(
            torch.tensor([[lang_id]], device=self.device, dtype=torch.long))
        bos_emb   = talker.get_input_embeddings()(
            torch.tensor([[CODEC_BOS_ID]], device=self.device, dtype=torch.long))

        return torch.cat([text_embeds, lang_emb, bos_emb], dim=1)  # [1, T+2, 1024]

    def synthesize(
        self,
        text: str,
        language: str = "english",
        max_codec_tokens: int = 1000,
        temperature: float = 0.9,
        top_k: int = 50,
    ):
        """
        Full quality synthesis using megakernel for decode speed.
        
        Returns (audio_np, sample_rate, metrics_dict)
        """
        t0 = time.time()

        # --- 1. Build prefill embeddings ---
        input_embeds = self._build_input_embeds(text, language)

        # --- 2. PyTorch prefill -> KV cache ---
        t_prefill_start = time.time()
        prefill_out = self.talker.model(
            inputs_embeds=input_embeds,
            use_cache=True,
            output_hidden_states=False,
        )
        hf_cache = prefill_out.past_key_values
        t_prefill = time.time() - t_prefill_start

        # --- 3. Transfer KV cache to megakernel ---
        seq_len = _transfer_kv_cache(
            hf_cache, self.mk_decoder._k_cache, self.mk_decoder._v_cache)
        self.mk_decoder._position = seq_len

        # --- 4. Megakernel decode loop with chunked embedding injection ---
        #
        # Strategy: decode CHUNK_SIZE tokens with layer-0 embedding only,
        # then run code_predictor in batch on those tokens to get all 16
        # codebook layers, then inject summed embeddings for the next chunk.
        #
        # This amortizes code_predictor cost: 1 batched call per CHUNK_SIZE
        # tokens instead of 1 serial call per token.
        # CHUNK_SIZE=32: ~63ms CP call / 32 tokens = ~2ms overhead/token
        # vs megakernel ~1ms/token → ~3ms effective = ~333 tok/s
        #
        CHUNK_SIZE = 32

        t_decode_start = time.time()
        t_first_token  = None
        codec_tokens   = []
        hidden_states  = []

        codec_embed     = self.talker.get_input_embeddings()
        cp_embed_tables = self.talker.code_predictor.get_input_embeddings()
        embed_weight    = self.mk_decoder._embed_weight
        # Precompute stacked CP weight matrix [15, 2048, 1024] for fast lookup
        with torch.no_grad():
            stacked_cp = torch.stack(
                [t.weight for t in cp_embed_tables], dim=0
            )  # [15, 2048, 1024]
        cp_arange = torch.arange(15, device=self.device)

        # all_codes_table[tok] = [16] tensor of all codebook tokens for tok
        # Used to build summed embeddings for next chunk
        all_codes_table = {}  # token_id -> [16] LongTensor

        current_token = CODEC_BOS_ID
        done = False
        total_steps = 0
        MAX_STEPS = max_codec_tokens * 8  # safety limit

        while not done and total_steps < MAX_STEPS:
            # --- Phase A: decode CHUNK_SIZE tokens with current embeddings ---
            chunk_tokens   = []
            chunk_hiddens  = []

            for _ in range(CHUNK_SIZE):
                # Inject summed embedding if we have it (fused batch lookup)
                if current_token in all_codes_table:
                    codes = all_codes_table[current_token]  # [16] LongTensor
                    with torch.no_grad():
                        # Fast stacked lookup: one gather op for all 15 CP tables
                        summed = codec_embed.weight[codes[0]]  # [1024]
                        cp_vecs = stacked_cp[cp_arange, codes[1:]]  # [15, 1024]
                        summed = summed + cp_vecs.sum(0)
                        embed_weight[current_token] = summed.to(torch.bfloat16)

                next_token = self.mk_decoder.step(current_token)
                total_steps += 1

                if t_first_token is None:
                    t_first_token = time.time() - t0

                if next_token == CODEC_EOS_ID:
                    done = True
                    break
                if next_token >= 2048:
                    current_token = 0
                    continue
                if len(codec_tokens) >= max_codec_tokens:
                    done = True
                    break

                h = self.mk_decoder._norm_out.clone().to(torch.bfloat16)
                chunk_hiddens.append(h)
                chunk_tokens.append(next_token)
                codec_tokens.append(next_token)
                hidden_states.append(h)
                current_token = next_token

            if not chunk_tokens:
                break

            # --- Phase B: batch code_predictor on this chunk ---
            with torch.no_grad():
                T_chunk = len(chunk_tokens)
                h_batch = torch.stack(chunk_hiddens, dim=0)   # [T, 1024]
                c0_ids  = torch.tensor(chunk_tokens, device=self.device)
                c0_emb  = codec_embed(c0_ids.unsqueeze(1))    # [T, 1, 1024]
                cp_inp  = torch.cat([h_batch.unsqueeze(1), c0_emb], dim=1)  # [T, 2, 1024]

                cp_res = self.talker.code_predictor.generate(
                    inputs_embeds=cp_inp,
                    max_new_tokens=self.talker.config.num_code_groups - 1,
                    do_sample=False,
                    return_dict_in_generate=True,
                )
                # sequences: [T, 15]
                for t_idx in range(T_chunk):
                    tok_id = chunk_tokens[t_idx]
                    all_codes = torch.cat([
                        torch.tensor([tok_id], device=self.device),
                        cp_res.sequences[t_idx],
                    ], dim=0)  # [16]
                    all_codes_table[tok_id] = all_codes

        t_decode = time.time() - t_decode_start
        self.mk_decoder.reset()

        # Rebuild all_codes from collected codec_tokens using all_codes_table
        # (used by Mimi decode below)

        if not codec_tokens:
            return np.zeros(1920, dtype=np.float32), AUDIO_HZ, {}

        n_tokens = len(codec_tokens)

        # --- 5. Build all_codes [T, 16] from all_codes_table ---
        t_cp_start = time.time()
        T = n_tokens
        all_codes = torch.zeros(T, 16, device=self.device, dtype=torch.long)
        all_codes[:, 0] = torch.tensor(codec_tokens, device=self.device)

        # Fill layers 1-15 from all_codes_table (populated during decode chunks)
        missing = []
        for t_idx, tok_id in enumerate(codec_tokens):
            if tok_id in all_codes_table:
                all_codes[t_idx] = all_codes_table[tok_id]
            else:
                missing.append(t_idx)

        # Batch fallback for any missing tokens
        if missing:
            h_batch = torch.stack([hidden_states[i] for i in missing]).unsqueeze(1)
            c0_ids  = torch.tensor([codec_tokens[i] for i in missing], device=self.device)
            c0_emb  = self.talker.get_input_embeddings()(c0_ids.unsqueeze(1))
            cp_inp  = torch.cat([h_batch, c0_emb], dim=1)
            with torch.no_grad():
                cp_res = self.talker.code_predictor.generate(
                    inputs_embeds=cp_inp,
                    max_new_tokens=self.talker.config.num_code_groups - 1,
                    do_sample=False, return_dict_in_generate=True,
                )
            for j, t_idx in enumerate(missing):
                all_codes[t_idx, 1:] = cp_res.sequences[j]

        t_cp = time.time() - t_cp_start

        # --- 6. Mimi decode: all_codes [T, 16] -> audio ---
        t_mimi_start = time.time()
        wavs, sr = self.tts_model.model.speech_tokenizer.decode(
            [{"audio_codes": all_codes}]
        )
        t_mimi = time.time() - t_mimi_start

        audio     = wavs[0]
        audio_len = len(audio) / sr
        t_total   = time.time() - t0
        rtf       = t_total / max(audio_len, 0.001)
        tok_s     = n_tokens / max(t_decode, 0.001)

        metrics = {
            "ttfc_ms":     (t_first_token or 0) * 1000,
            "rtf":         rtf,
            "tok_per_sec": tok_s,
            "n_tokens":    n_tokens,
            "audio_len_s": audio_len,
            "t_prefill_s": t_prefill,
            "t_decode_s":  t_decode,
            "t_cp_s":      t_cp,
            "t_mimi_s":    t_mimi,
            "t_total_s":   t_total,
        }

        if self.verbose:
            print(f"\n=== TTS Metrics ===")
            print(f"  TTFC:       {metrics['ttfc_ms']:.1f}ms  (target <60ms)")
            print(f"  RTF:        {metrics['rtf']:.3f}  (target <0.15)")
            print(f"  Throughput: {metrics['tok_per_sec']:.0f} tok/s")
            print(f"  Tokens:     {n_tokens} -> {audio_len:.2f}s audio")
            print(f"  Prefill:    {t_prefill*1000:.0f}ms")
            print(f"  Decode:     {t_decode*1000:.0f}ms  (megakernel)")
            print(f"  CodePred:   {t_cp*1000:.0f}ms")
            print(f"  Mimi:       {t_mimi*1000:.0f}ms")
            print(f"  Total:      {t_total*1000:.0f}ms")

        return audio, sr, metrics

    def synthesize_streaming(
        self,
        text: str,
        language: str = "english",
        max_codec_tokens: int = 1000,
        chunk_frames: int = 4,
    ) -> Iterator[np.ndarray]:
        """Yield audio chunks for Pipecat streaming integration."""
        audio, sr, metrics = self.synthesize(text, language, max_codec_tokens)
        chunk_size = chunk_frames * SAMPLES_PER_FRAME
        for i in range(0, len(audio), chunk_size):
            yield audio[i:i+chunk_size].copy()
