"""
Voice Activity Detection (VAD) モジュール
Silero VAD を使って発話区間を検出する
"""
from typing import Optional

import numpy as np
import torch


class VoiceActivityDetector:
    """Silero VAD を使った発話検出クラス"""

    def __init__(self, threshold: float = 0.5,
                 min_speech_duration_ms: int = 250,
                 min_silence_duration_ms: int = 700,
                 sample_rate: int = 16000):
        self.threshold = threshold
        self.min_speech_frames = int(min_speech_duration_ms * sample_rate / 1000 / 512)
        self.min_silence_frames = int(min_silence_duration_ms * sample_rate / 1000 / 512)
        self.sample_rate = sample_rate
        self._model: Optional[torch.nn.Module] = None
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        self._speech_buffer: list = []

    def load(self) -> None:
        """Silero VAD モデルをロード（silero-vad 6.x API）"""
        from silero_vad import load_silero_vad  # pylint: disable=import-outside-toplevel
        self._model = load_silero_vad()
        self._model.eval()

    def is_speech(self, audio_chunk: np.ndarray) -> float:
        """音声チャンクの発話確率を返す (0.0 〜 1.0)"""
        if self._model is None:
            raise RuntimeError("VAD モデルがロードされていません。load() を呼んでください。")
        tensor = torch.from_numpy(audio_chunk.flatten()).float()
        with torch.no_grad():
            prob = self._model(tensor, self.sample_rate).item()
        return prob

    def process_chunk(self, audio_chunk: np.ndarray) -> Optional[np.ndarray]:
        """
        チャンクを処理し、発話セグメントが完成したら返す。
        まだ完成していない場合は None を返す。
        """
        prob = self.is_speech(audio_chunk)

        if prob >= self.threshold:
            self._speech_frames += 1
            self._silence_frames = 0
            self._speech_buffer.append(audio_chunk)
            if not self._in_speech and self._speech_frames >= self.min_speech_frames:
                self._in_speech = True
        else:
            self._silence_frames += 1
            if self._in_speech:
                self._speech_buffer.append(audio_chunk)
                if self._silence_frames >= self.min_silence_frames:
                    # 発話終了 → バッファを返す
                    segment = np.concatenate(self._speech_buffer, axis=0)
                    self._reset()
                    return segment
            else:
                self._speech_frames = 0
                self._speech_buffer.clear()

        return None

    @property
    def in_speech(self) -> bool:
        """現在発話中かどうかを返す"""
        return self._in_speech

    def _reset(self) -> None:
        self._speech_frames = 0
        self._silence_frames = 0
        self._in_speech = False
        self._speech_buffer = []
