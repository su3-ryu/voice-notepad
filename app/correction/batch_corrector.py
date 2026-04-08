"""
バッチ校正モジュール
文字起こしテキストを蓄積し、定期的にOllamaで校正する
"""
from typing import Optional

from PyQt6.QtCore import QObject, QThread, QTimer, pyqtSignal

from app.correction.ollama_client import (
    DEFAULT_SYSTEM_PROMPT,
    REVIEW_SYSTEM_PROMPT,
    OllamaClient,
)


class CorrectionWorker(QThread):
    """バックグラウンドで複数段のOllama校正を実行するワーカー"""
    finished = pyqtSignal(str, str)  # (元テキスト, 校正後テキスト)

    def __init__(self, clients: list[OllamaClient], text: str):
        super().__init__()
        self._clients = clients
        self._text = text

    def run(self) -> None:
        """Ollama APIで校正を実行"""
        try:
            corrected = self._text
            for client in self._clients:
                corrected = client.correct_text(corrected)
            if corrected and corrected != self._text:
                self.finished.emit(self._text, corrected)
        except (RuntimeError, OSError, ValueError) as e:
            print(f"[CorrectionWorker] エラー: {e}")


class BatchCorrector(QObject):
    """文字起こしテキストを蓄積し、定期的にOllamaで校正するコントローラー"""
    correction_ready = pyqtSignal(str, str)  # (元テキスト, 校正後テキスト)
    status_changed = pyqtSignal(str)

    def __init__(self, config: Optional[dict] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        cfg = config or {}
        self._clients = [OllamaClient(
            base_url=cfg.get("ollama_url", "http://localhost:11434"),
            model=cfg.get("model", "qwen2.5:7b"),
            timeout=cfg.get("timeout_sec", 30),
            system_prompt=cfg.get("system_prompt", DEFAULT_SYSTEM_PROMPT),
        )]
        if cfg.get("review_enabled", True):
            self._clients.append(OllamaClient(
                base_url=cfg.get("ollama_url", "http://localhost:11434"),
                model=cfg.get("review_model", cfg.get("model", "qwen2.5:7b")),
                timeout=cfg.get("timeout_sec", 30),
                system_prompt=cfg.get(
                    "review_system_prompt", REVIEW_SYSTEM_PROMPT),
            ))
        self._buffer: list[str] = []
        self._active_workers: list[CorrectionWorker] = []
        self._enabled = cfg.get("enabled", True)
        self._min_segments = cfg.get("min_segments", 3)
        self._interval_sec = cfg.get("batch_interval_sec", 15)
        self._defer_while_recording = cfg.get("defer_while_recording", False)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._trigger_correction)

    def start(self) -> None:
        """校正タイマーを開始する（録音開始時に呼ぶ）"""
        if not self._enabled:
            return
        if not all(client.is_available() for client in self._clients):
            self.status_changed.emit("Ollama未接続 - 自動校正は無効です")
            self._enabled = False
            return
        if self._defer_while_recording:
            self.status_changed.emit("自動校正: 録音停止後に実行")
            return
        self.status_changed.emit("自動校正: 有効")
        self._timer.start(self._interval_sec * 1000)

    def stop(self) -> None:
        """校正タイマーを停止し、残りのバッファをフラッシュする"""
        self._timer.stop()
        if self._buffer:
            self._trigger_correction(force=True)

    def add_text(self, text: str) -> None:
        """文字起こし結果を校正バッファに追加する"""
        if not self._enabled:
            return
        self._buffer.append(text)
        if self._defer_while_recording:
            return
        # バッファが十分溜まったらタイマー待たず即発火
        if len(self._buffer) >= self._min_segments * 2:
            self._trigger_correction()

    def _trigger_correction(self, force: bool = False) -> None:
        """バッファのテキストをまとめてOllamaに送信する"""
        if not self._buffer or not self._enabled:
            return
        # 最低セグメント数に達していない場合はスキップ（stop時のフラッシュは除く）
        if (
                not force
                and len(self._buffer) < self._min_segments
                and self._timer.isActive()):
            return

        batch_text = "".join(self._buffer)
        self._buffer.clear()

        worker = CorrectionWorker(self._clients, batch_text)
        worker.finished.connect(self._on_correction_done)
        worker.finished.connect(lambda: self._cleanup_worker(worker))
        worker.start()
        self._active_workers.append(worker)

    def _on_correction_done(self, original: str, corrected: str) -> None:
        """校正完了時のハンドラ"""
        self.correction_ready.emit(original, corrected)

    def _cleanup_worker(self, worker: CorrectionWorker) -> None:
        """完了したワーカーをリストから除去"""
        if worker in self._active_workers:
            self._active_workers.remove(worker)
