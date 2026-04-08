"""
音声認識エンジン
faster-whisper を使って音声をテキストに変換する
"""
import os
import re
import time
from typing import Optional

import numpy as np
import yaml
from faster_whisper import BatchedInferencePipeline, WhisperModel


class TranscriptionEngine:
    """faster-whisper ベースの文字起こしエンジン"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        t = cfg["transcription"]
        self.model_name: str = t["model"]
        self.language: str = t["language"]
        self.initial_prompt: str = t.get("initial_prompt", "")
        self.hotwords: str = t.get("hotwords", "")
        self.device: str = t.get("device", "auto")
        self.compute_type: str = t.get("compute_type", "int8")
        self.cpu_threads = t.get("cpu_threads", "auto")
        self.num_workers: int = t.get("num_workers", 1)
        self.beam_size: int = t.get("beam_size", 3)
        self.best_of: int = t.get("best_of", 1)
        self.temperature = t.get("temperature", 0.0)
        self.pad_duration_ms: int = t.get("pad_duration_ms", 300)
        self.condition_on_previous_text: bool = t.get(
            "condition_on_previous_text", False
        )
        self.vad_filter: bool = t.get("vad_filter", False)
        self.vad_min_silence_duration_ms: int = t.get(
            "vad_min_silence_duration_ms", 500
        )
        self.no_speech_threshold: float = t.get("no_speech_threshold", 0.6)
        self.log_prob_threshold: float = t.get("log_prob_threshold", -1.0)
        self.compression_ratio_threshold: float = t.get(
            "compression_ratio_threshold", 2.4
        )
        self.use_batched: bool = t.get("use_batched", False)
        self.batch_size: int = t.get("batch_size", 4)
        self._model: Optional[WhisperModel] = None
        self._transcriber = None

    def load(self) -> None:
        """モデルをロード（初回は自動ダウンロード）"""
        device = self.device
        if device == "auto":
            import torch  # pylint: disable=import-outside-toplevel
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # compute_type自動選択: GPU→float16, CPU→int8
        compute_type = self.compute_type
        if compute_type == "auto":
            compute_type = "float16" if device == "cuda" else "int8"

        cpu_threads = self.cpu_threads
        if cpu_threads == "auto":
            cpu_threads = min(max((os.cpu_count() or 4) - 2, 4), 12)

        print(f"[TranscriptionEngine] モデルロード中: {self.model_name} ({device}, {compute_type})")
        self._model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
            cpu_threads=cpu_threads,
            num_workers=self.num_workers,
            download_root="models",
        )
        self._transcriber = (
            BatchedInferencePipeline(model=self._model)
            if self.use_batched else self._model
        )
        print("[TranscriptionEngine] モデルロード完了")

    @staticmethod
    def _looks_repetitive(text: str) -> bool:
        """短い認識区間で出やすい同一フレーズ反復の幻覚を検出する"""
        compact = re.sub(r"\s+", "", text)
        if len(compact) < 12:
            return False

        for size in range(4, 13):
            counts = {}
            for start in range(0, max(len(compact) - size + 1, 0), size):
                phrase = compact[start:start + size]
                counts[phrase] = counts.get(phrase, 0) + 1
                if counts[phrase] >= 3:
                    return True
        return False

    def transcribe(self, audio: np.ndarray) -> str:
        """
        音声データ (float32, 16kHz, mono) をテキストに変換する

        Args:
            audio: numpy array (float32, shape: [N])

        Returns:
            認識結果テキスト
        """
        if self._model is None or self._transcriber is None:
            raise RuntimeError("モデルがロードされていません。load() を呼んでください。")

        # 短い無音を足して語頭・語尾切れを抑える。長いほど遅延も増える。
        pad_samples = int(self.pad_duration_ms * 16000 / 1000)
        pad = np.zeros(pad_samples, dtype=np.float32)
        audio = np.concatenate([pad, audio.flatten(), pad])

        start = time.perf_counter()
        kwargs = {
            "language": self.language,
            "initial_prompt": self.initial_prompt if self.initial_prompt else None,
            "beam_size": self.beam_size,
            "best_of": self.best_of,
            "temperature": self.temperature,
            "condition_on_previous_text": self.condition_on_previous_text,
            "no_speech_threshold": self.no_speech_threshold,
            "log_prob_threshold": self.log_prob_threshold,
            "compression_ratio_threshold": self.compression_ratio_threshold,
            "vad_filter": self.vad_filter,
            "hotwords": self.hotwords if self.hotwords else None,
        }
        if self.vad_filter:
            kwargs["vad_parameters"] = {
                "min_silence_duration_ms": self.vad_min_silence_duration_ms
            }
        if self.use_batched:
            kwargs["batch_size"] = self.batch_size

        segments, _info = self._transcriber.transcribe(
            audio,
            **kwargs,
        )

        result_parts = []
        for seg in segments:
            # 無音確率が高いセグメントを除外
            if seg.no_speech_prob > 0.6:
                continue
            # 平均log確率が低すぎるセグメントを除外（低品質）
            if seg.avg_logprob < -1.0:
                continue
            result_parts.append(seg.text)

        result = "".join(result_parts).strip()
        if self._looks_repetitive(result):
            print(f"[TranscriptionEngine] 反復した認識結果を破棄: {result}")
            return ""

        elapsed = time.perf_counter() - start
        print(
            f"[TranscriptionEngine] 認識完了: {elapsed:.2f}s, "
            f"{audio.size / 16000:.2f}s audio"
        )
        return result
