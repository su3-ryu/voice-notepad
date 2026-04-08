"""
音声入力の録音モジュール
sounddevice の入力録音と SoundCard の WASAPI ループバック録音を扱う
"""
import queue
from typing import Any, Optional

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """マイクまたはループバックデバイスから音声をリアルタイムで録音するクラス"""

    def __init__(self, sample_rate: int = 16000, channels: int = 1,
                 chunk_duration_ms: int = 30, device: Optional[int] = None,
                 backend: str = "sounddevice"):
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = int(sample_rate * chunk_duration_ms / 1000)
        self.device = device
        self.backend = backend
        self._audio_queue: queue.Queue = queue.Queue()
        self._stream: Optional[sd.InputStream] = None
        self._loopback_recorder_context: Optional[Any] = None
        self._loopback_recorder: Optional[Any] = None
        self._is_recording = False
        # デバイスの実際のチャンネル数（ステレオミキサー等はステレオ）
        self._device_channels = channels
        if backend == "loopback":
            # SoundCard の Windows/WASAPI は単一チャンネル録音に既知問題があるため、
            # 2chで録ってからこのクラス内でモノラル化する。
            self._device_channels = max(2, channels)
        elif device is not None:
            try:
                info = sd.query_devices(device)
                self._device_channels = min(int(info["max_input_channels"]), 2)
            except (ValueError, sd.PortAudioError):
                self._device_channels = channels

    @staticmethod
    def _to_mono(audio: np.ndarray) -> np.ndarray:
        """入力音声を downstream 用のモノラル float32 配列へ整形する"""
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim == 1:
            audio = audio.reshape(-1, 1)
        if audio.ndim == 2 and audio.shape[1] > 1:
            audio = audio.mean(axis=1, keepdims=True)
        return audio

    def _callback(self, indata: np.ndarray, _frames: int,
                  _time_info, status) -> None:
        if status:
            print(f"[AudioRecorder] 警告: {status}")
        self._audio_queue.put(self._to_mono(indata.copy()))

    def start(self) -> None:
        """録音開始"""
        if self._is_recording:
            return
        self._is_recording = True

        if self.backend == "loopback":
            try:
                self._start_loopback()
            except Exception:
                self._is_recording = False
                if self._loopback_recorder_context:
                    self._loopback_recorder_context.__exit__(None, None, None)
                    self._loopback_recorder_context = None
                    self._loopback_recorder = None
                raise
            return

        try:
            self._stream = sd.InputStream(
                device=self.device,
                samplerate=self.sample_rate,
                channels=self._device_channels,
                dtype="float32",
                blocksize=self.chunk_size,
                callback=self._callback,
            )
            self._stream.start()
        except Exception:
            self._is_recording = False
            self._stream = None
            raise

    def _start_loopback(self) -> None:
        """既定のスピーカー出力を WASAPI ループバック録音として開く"""
        try:
            import soundcard as sc
        except ImportError as e:
            raise RuntimeError(
                "スピーカー録音には soundcard パッケージが必要です。"
            ) from e

        device_id = self.device or sc.default_speaker().id
        microphone = sc.get_microphone(device_id, include_loopback=True)
        self._loopback_recorder_context = microphone.recorder(
            samplerate=self.sample_rate,
            channels=self._device_channels,
            blocksize=max(self.chunk_size * 4, 1024),
        )
        self._loopback_recorder = self._loopback_recorder_context.__enter__()

    def stop(self) -> None:
        """録音停止"""
        self._is_recording = False
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        if self._loopback_recorder_context:
            self._loopback_recorder_context.__exit__(None, None, None)
            self._loopback_recorder_context = None
            self._loopback_recorder = None

    def read_chunk(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """チャンクを1つ取得する（タイムアウトあり）"""
        if self.backend == "loopback":
            if not self._is_recording or self._loopback_recorder is None:
                return None
            audio = self._loopback_recorder.record(numframes=self.chunk_size)
            return self._to_mono(audio)

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
        """利用可能な入力デバイスの一覧を返す"""
        devices = sd.query_devices()
        return [
            {"index": i, "name": d["name"], "channels": d["max_input_channels"]}
            for i, d in enumerate(devices)
            if d["max_input_channels"] > 0
        ]

    @staticmethod
    def list_loopback_devices() -> list[dict]:
        """利用可能なスピーカーループバックデバイスの一覧を返す"""
        try:
            import soundcard as sc
        except ImportError:
            return []

        return [
            {
                "id": speaker.id,
                "name": speaker.name,
                "channels": max(int(speaker.channels), 2),
            }
            for speaker in sc.all_speakers()
        ]

    @staticmethod
    def find_loopback_device() -> Optional[Any]:
        """スピーカー録音に使えるループバックデバイスを自動検出する"""
        devices = AudioRecorder.list_loopback_devices()
        if devices:
            try:
                import soundcard as sc
                return sc.default_speaker().id
            except (ImportError, RuntimeError, OSError, IndexError):
                return devices[0]["id"]
        return None
