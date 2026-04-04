"""
音声認識エンジン
faster-whisper を使って音声をテキストに変換する
"""
from typing import Optional

import numpy as np
import yaml
from faster_whisper import WhisperModel


class TranscriptionEngine:
    """faster-whisper ベースの文字起こしエンジン"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        t = cfg["transcription"]
        self.model_name: str = t["model"]
        self.language: str = t["language"]
        self.initial_prompt: str = t.get("initial_prompt", "")
        self.device: str = t.get("device", "auto")
        self.compute_type: str = t.get("compute_type", "int8")
        self._model: Optional[WhisperModel] = None

    def load(self) -> None:
        """モデルをロード（初回は自動ダウンロード）"""
        device = self.device
        if device == "auto":
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"

        print(f"[TranscriptionEngine] モデルロード中: {self.model_name} ({device})")
        self._model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=self.compute_type,
            download_root="models",
        )
        print("[TranscriptionEngine] モデルロード完了")

    def transcribe(self, audio: np.ndarray) -> str:
        """
        音声データ (float32, 16kHz, mono) をテキストに変換する

        Args:
            audio: numpy array (float32, shape: [N])

        Returns:
            認識結果テキスト
        """
        if self._model is None:
            raise RuntimeError("モデルがロードされていません。load() を呼んでください。")

        segments, _info = self._model.transcribe(
            audio,
            language=self.language,
            initial_prompt=self.initial_prompt if self.initial_prompt else None,
            beam_size=5,
            best_of=3,
            temperature=0.0,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            log_prob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            vad_filter=False,
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

        return "".join(result_parts).strip()
