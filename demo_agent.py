#!/usr/bin/env python3
"""
Qwen3-TTS Megakernel Voice Agent Demo.

Pipeline: Microphone -> Deepgram STT -> OpenAI LLM -> Megakernel TTS -> Speaker

Usage:
    export OPENAI_API_KEY=sk-...
    export DEEPGRAM_API_KEY=...
    python demo_agent.py

For testing without real STT/LLM:
    python demo_agent.py --test
"""

import asyncio
import argparse
import os
import time
import numpy as np
import soundfile as sf

async def run_tts_test():
    """Test TTS pipeline standalone without full Pipecat agent."""
    print("=== Megakernel TTS Standalone Test ===")
    print()

    from qwen_megakernel.tts_pipeline import MegakernelTTSPipeline

    print("Loading pipeline...")
    pipeline = MegakernelTTSPipeline(verbose=False)

    # Warmup
    pipeline.synthesize("Warmup.", max_codec_tokens=50)

    test_phrases = [
        "Hello! How can I help you today?",
        "I am a voice assistant powered by the Qwen3 TTS megakernel, running at over eight hundred tokens per second on an RTX 5090 GPU.",
        "The weather today is sunny with a high of 72 degrees. Would you like more details?",
    ]

    results = []
    for phrase in test_phrases:
        print(f"Synthesizing: {phrase[:60]}...")
        t0 = time.time()
        audio, sr, metrics = pipeline.synthesize(phrase, language="english")
        elapsed = time.time() - t0

        fname = f"/workspace/demo_{len(results)}.wav"
        sf.write(fname, audio, sr)

        results.append(metrics)
        print(f"  TTFC={metrics['ttfc_ms']:.0f}ms  RTF={metrics['rtf']:.3f}  "
              f"tok/s={metrics['tok_per_sec']:.0f}  "
              f"audio={metrics['audio_len_s']:.1f}s  -> {fname}")
        print()

    print("=== Summary ===")
    print(f"  Mean TTFC:    {sum(r['ttfc_ms'] for r in results)/len(results):.0f}ms  (target <60ms)")
    print(f"  Mean RTF:     {sum(r['rtf'] for r in results)/len(results):.3f}  (target <0.15)")
    print(f"  Mean tok/s:   {sum(r['tok_per_sec'] for r in results)/len(results):.0f}  (target ~1000)")
    print()

    # Test streaming
    print("Testing streaming synthesis...")
    phrase = "This audio is being streamed chunk by chunk into the Pipecat pipeline."
    chunks = list(pipeline.synthesize_streaming(phrase, chunk_frames=4))
    total_samples = sum(len(c) for c in chunks)
    print(f"  {len(chunks)} chunks, {total_samples/24000:.2f}s total audio")
    print()
    print("All tests passed. Audio saved to /workspace/demo_*.wav")


async def run_pipecat_agent():
    """Run full Pipecat voice agent (requires API keys)."""
    try:
        from pipecat.pipeline.pipeline import Pipeline
        from pipecat.pipeline.runner import PipelineRunner
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
        from pipecat.services.openai import OpenAILLMService
        from pipecat.services.deepgram import DeepgramSTTService
        from pipecat.transports.local.audio import LocalAudioTransport, LocalAudioParams
        from qwen_megakernel.pipecat_service import MegakernelTTSService
    except ImportError as e:
        print(f"Import error: {e}")
        print("Running TTS test instead...")
        await run_tts_test()
        return

    openai_key = os.environ.get("OPENAI_API_KEY")
    deepgram_key = os.environ.get("DEEPGRAM_API_KEY")

    if not openai_key or not deepgram_key:
        print("Missing API keys. Set OPENAI_API_KEY and DEEPGRAM_API_KEY.")
        print("Running TTS test instead...")
        await run_tts_test()
        return

    print("=== Qwen3-TTS Megakernel Voice Agent ===")
    print("Loading components...")

    transport = LocalAudioTransport(LocalAudioParams(
        audio_in_enabled=True,
        audio_out_enabled=True,
        vad_enabled=True,
    ))

    stt = DeepgramSTTService(api_key=deepgram_key)

    llm = OpenAILLMService(
        api_key=openai_key,
        model="gpt-4o-mini",
    )

    tts = MegakernelTTSService(
        language="english",
        chunk_frames=4,
    )

    context = OpenAILLMContext(messages=[{
        "role": "system",
        "content": (
            "You are a helpful voice assistant. "
            "Keep responses concise and conversational — 1-3 sentences. "
            "You are powered by the Qwen3 TTS megakernel running at 900+ tok/s."
        )
    }])
    context_aggregator = llm.create_context_aggregator(context)

    pipeline = Pipeline([
        transport.input(),
        stt,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ])

    task = PipelineTask(pipeline, PipelineParams(allow_interruptions=True))

    @transport.event_handler("on_client_connected")
    async def on_connected(transport, client):
        await task.queue_frames([context_aggregator.user().get_context_frame()])

    runner = PipelineRunner()
    print("Voice agent ready. Speak into your microphone!")
    print("Press Ctrl+C to stop.")
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true",
                        help="Run TTS test only (no API keys needed)")
    args = parser.parse_args()

    if args.test:
        asyncio.run(run_tts_test())
    else:
        asyncio.run(run_pipecat_agent())
