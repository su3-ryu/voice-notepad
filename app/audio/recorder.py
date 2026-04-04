"""
マイク入力の録音モジュール
sounddevice を使ったリアルタイム音声キャプチャ
"""
import queue
from typing import Optional

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """マイクから音声をリアルタイムで録音するクラス"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_duration_ms: int = 30):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self._audio_queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._is_recording = False

    def _callback(self, indata: np.ndarray, _frames: int,
                  _time_info, status) -> None:
        if status:
            print(f"[AudioRecorder] 警告: {status}")
        self._audio_queue.put(indata.copy())

    def start(self) -> None:
        """録音開始"""
        if self._is_recording:
            return
        self._is_recording = True
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_size,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        """録音停止"""
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def read_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """チャンクを1つ取得する（タイムアウトあり）"""
        try:
            return self._audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def clear_buffer(self) -> None:
        """バッファをクリア"""
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except queue.Empty:
                break

    @property
    def is_recording(self) -> bool:
        """録音中かどうかを返す"""
        return self._is_recording

    @staticmethod
    def list_devices() -> list[dict]:
        """利用可能なマイクデバイスの一覧を返す"""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]
