"""
Pipecat TTS Service using Qwen3-TTS Megakernel.

Implements pipecat-ai's TTSService interface so it plugs into any
Pipecat pipeline as a drop-in TTS provider.

Pipeline: STT -> LLM -> MegakernelTTSService -> audio output
"""

import asyncio
import numpy as np
from typing import AsyncGenerator, Optional

from pipecat.services.tts import TTSService
from pipecat.frames.frames import (
    AudioRawFrame,
    StartFrame,
    EndFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    ErrorFrame,
)
from pipecat.processors.frame_processor import FrameDirection


class MegakernelTTSService(TTSService):
    """
    Pipecat TTS service backed by the Qwen3-TTS megakernel pipeline.

    Streams audio chunks into the Pipecat pipeline frame-by-frame.
    Never buffers the full utterance — pushes audio as soon as
    codec tokens are decoded.

    Usage in a Pipecat pipeline:
        tts = MegakernelTTSService(language="english", chunk_frames=4)
        pipeline = Pipeline([stt, llm, tts, transport.output()])
    """

    def __init__(
        self,
        *,
        language: str = "english",
        chunk_frames: int = 4,
        max_codec_tokens: int = 1000,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._language = language
        self._chunk_frames = chunk_frames
        self._max_codec_tokens = max_codec_tokens
        self._model_name = model_name
        self._pipeline = None
        self._sample_rate = 24000
        self._samples_per_frame = int(24000 / 12.5)  # 1920

    async def start(self, frame: StartFrame):
        await super().start(frame)
        # Load pipeline in background thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        self._pipeline = await loop.run_in_executor(
            None, self._load_pipeline
        )

    def _load_pipeline(self):
        from qwen_megakernel.tts_pipeline import MegakernelTTSPipeline
        return MegakernelTTSPipeline(
            model_name=self._model_name,
            verbose=False,
        )

    async def run_tts(self, text: str) -> AsyncGenerator[AudioRawFrame, None]:
        """
        Generate TTS audio for text, yielding AudioRawFrame chunks.
        
        Pipecat calls this for each text segment from the LLM.
        We stream audio chunks as soon as they are ready.
        """
        if not self._pipeline:
            yield ErrorFrame("MegakernelTTSService not started")
            return

        if not text or not text.strip():
            return

        yield TTSStartedFrame()

        try:
            # Run synthesis in executor to avoid blocking event loop
            loop = asyncio.get_event_loop()
            audio, sr, metrics = await loop.run_in_executor(
                None,
                lambda: self._pipeline.synthesize(
                    text,
                    language=self._language,
                    max_codec_tokens=self._max_codec_tokens,
                )
            )

            # Stream audio in chunks (frame-by-frame, not buffered)
            chunk_size = self._chunk_frames * self._samples_per_frame
            audio_int16 = (audio * 32767).clip(-32768, 32767).astype(np.int16)

            for i in range(0, len(audio_int16), chunk_size):
                chunk = audio_int16[i:i+chunk_size]
                yield AudioRawFrame(
                    audio=chunk.tobytes(),
                    sample_rate=self._sample_rate,
                    num_channels=1,
                )
                # Yield control to event loop between chunks
                await asyncio.sleep(0)

        except Exception as e:
            yield ErrorFrame(f"TTS error: {e}")
            raise

        finally:
            yield TTSStoppedFrame()

    @property
    def sample_rate(self) -> int:
        return self._sample_rate
