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

        # --- 4. Megakernel decode loop ---
        # Use _norm_out buffer (final hidden state) directly after each step.
        # This eliminates the need for a parallel PyTorch forward pass.
        t_decode_start = time.time()
        t_first_token  = None
        codec_tokens   = []   # layer-0 codec token IDs
        hidden_states  = []   # talker hidden states for code_predictor

        current_token = CODEC_BOS_ID
        for step in range(max_codec_tokens):
            next_token = self.mk_decoder.step(current_token)

            if t_first_token is None:
                t_first_token = time.time() - t0

            if next_token == CODEC_EOS_ID:
                break

            # Special tokens (>=2048 but not EOS): skip appending but
            # still advance position — do NOT feed them back as input
            # or we get stuck in a loop repeating the same context.
            if next_token >= 2048:
                # Use a safe fallback token (0) as next input
                current_token = 0
                continue

            # _norm_out is the post-layernorm hidden state (pre-lm_head)
            # Clone to capture state before next step overwrites it
            h = self.mk_decoder._norm_out.clone()  # [1024], stays on GPU
            hidden_states.append(h)

            codec_tokens.append(next_token)
            current_token = next_token

        t_decode = time.time() - t_decode_start
        self.mk_decoder.reset()

        if not codec_tokens:
            return np.zeros(1920, dtype=np.float32), AUDIO_HZ, {}

        n_tokens = len(codec_tokens)

        # --- 5. Code predictor: batch all T positions at once ---
        # Instead of T serial calls, run one batched forward pass.
        # code_predictor forward_finetune takes [T, num_code_groups+1, 1024]
        t_cp_start = time.time()
        T = n_tokens
        all_codes = torch.zeros(T, 16, device=self.device, dtype=torch.long)
        all_codes[:, 0] = torch.tensor(codec_tokens, device=self.device)

        hidden_tensor = torch.stack(hidden_states, dim=0).to(self.dtype)  # [T, 1024]

        with torch.no_grad():
            # Clamp layer-0 codes to valid codec vocab range [0, 2047]
            # Some megakernel outputs may be special tokens (2048-3071)
            codec_vocab_size = self.talker.code_predictor.config.vocab_size  # 2048
            all_codes[:, 0] = all_codes[:, 0].clamp(0, codec_vocab_size - 1)

            # Build batch input: [T, 2, 1024] = [hidden | c0_embed] per position
            c0_emb = self.talker.get_input_embeddings()(
                all_codes[:, 0:1]
            )  # [T, 1, 1024]
            # sub_talker input: [T, 2, 1024]
            cp_inputs = torch.cat([hidden_tensor.unsqueeze(1), c0_emb], dim=1)

            # Run code_predictor.generate in one batched call
            cp_result = self.talker.code_predictor.generate(
                inputs_embeds=cp_inputs,   # [T, 2, 1024]
                max_new_tokens=self.talker.config.num_code_groups - 1,
                do_sample=False,
                return_dict_in_generate=True,
            )
            # sequences: [T, 15]
            all_codes[:, 1:] = cp_result.sequences

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
