"""
メインウィンドウ
PyQt6 ベースの音声メモ帳 UI
"""
import queue
import threading
from pathlib import Path
from typing import Optional

import yaml
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QMessageBox
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeySequence, QShortcut

from app.audio.recorder import AudioRecorder
from app.audio.vad import VoiceActivityDetector
from app.transcription.engine import TranscriptionEngine
from app.transcription.postprocess import postprocess
from app.storage.notes import NoteStorage


class TranscriptionWorker(QThread):
    """バックグラウンドで録音・VAD・文字起こしを行うワーカースレッド"""
    text_ready = pyqtSignal(str)   # 認識結果テキスト
    status_changed = pyqtSignal(str)  # ステータス更新

    def __init__(self, recorder: AudioRecorder, vad: VoiceActivityDetector,
                 engine: TranscriptionEngine):
        super().__init__()
        self.recorder = recorder
        self.vad = vad
        self.engine = engine
        self._running = False
        self._pending_segments: queue.Queue = queue.Queue()

    def run(self) -> None:
        """録音・VAD・文字起こしのメインループ"""
        self._running = True
        self.recorder.clear_buffer()
        self.recorder.start()
        self.status_changed.emit("録音中...")

        # 文字起こしを別スレッドで並行処理
        transcription_thread = threading.Thread(
            target=self._transcription_loop, daemon=True
        )
        transcription_thread.start()

        # VADループ（音声チャンクの処理に専念）
        while self._running:
            chunk = self.recorder.read_chunk(timeout=0.05)
            if chunk is None:
                continue

            try:
                audio_flat = chunk.flatten()
                if self.vad.in_speech:
                    self.status_changed.emit("🎙 音声検知中...")
                segment = self.vad.process_chunk(audio_flat)

                if segment is not None:
                    self._pending_segments.put(segment)
            except (RuntimeError, OSError, ValueError) as e:
                print(f"[Worker VAD] エラー: {e}")

        # 終了シグナルを送り、文字起こしスレッドの完了を待つ
        self._pending_segments.put(None)
        transcription_thread.join(timeout=5.0)
        self.recorder.stop()
        self.status_changed.emit("停止中")

    def _transcription_loop(self) -> None:
        """文字起こし専用ループ（別スレッドで実行）"""
        while True:
            segment = self._pending_segments.get()
            if segment is None:
                break
            try:
                self.status_changed.emit("⏳ 認識中...")
                text = self.engine.transcribe(segment)
                text = postprocess(text)
                if text:
                    self.text_ready.emit(text)
                self.status_changed.emit("🔴 録音中 - 話してください")
            except (RuntimeError, OSError, ValueError) as e:
                print(f"[Worker Transcription] エラー: {e}")

    def stop(self) -> None:
        """ワーカースレッドを停止する"""
        self._running = False


class MainWindow(QMainWindow):
    """Voice Notepad メインウィンドウ"""

    def __init__(self):
        super().__init__()
        self._load_config()
        self._setup_components()
        self._setup_ui()
        self._setup_shortcuts()
        self._worker: Optional[TranscriptionWorker] = None
        self._models_loaded = False

    def _load_config(self) -> None:
        with open("config.yaml", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def _setup_components(self) -> None:
        cfg = self._config
        self._recorder = AudioRecorder(
            sample_rate=cfg["audio"]["sample_rate"],
            channels=cfg["audio"]["channels"],
            chunk_duration_ms=cfg["audio"]["chunk_duration_ms"],
        )
        self._vad = VoiceActivityDetector(
            threshold=cfg["vad"]["threshold"],
            min_speech_duration_ms=cfg["vad"]["min_speech_duration_ms"],
            min_silence_duration_ms=cfg["vad"]["min_silence_duration_ms"],
            sample_rate=cfg["audio"]["sample_rate"],
        )
        self._engine = TranscriptionEngine()
        self._storage = NoteStorage(
            save_dir=cfg["storage"].get("save_directory", "notes")
        )

    def _setup_ui(self) -> None:
        self.setWindowTitle("Voice Notepad - 音声メモ帳")
        cfg_ui = self._config["ui"]
        self.resize(cfg_ui["window_width"], cfg_ui["window_height"])

        # フォント設定
        font = QFont(cfg_ui["font_family"], cfg_ui["font_size"])

        # 中央ウィジェット
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # テキストエディタ
        self._editor = QTextEdit()
        self._editor.setFont(font)
        self._editor.setPlaceholderText(
            "ここに文字起こし結果が表示されます。\n"
            "「録音開始」ボタンを押して話しかけてください。"
        )
        layout.addWidget(self._editor)

        # ボタン行
        btn_layout = QHBoxLayout()
        self._btn_record = QPushButton("● 録音開始")
        self._btn_record.setFixedHeight(40)
        self._btn_record.clicked.connect(self._toggle_recording)

        self._btn_clear = QPushButton("クリア")
        self._btn_clear.setFixedHeight(40)
        self._btn_clear.clicked.connect(self._clear_text)

        self._btn_save = QPushButton("保存")
        self._btn_save.setFixedHeight(40)
        self._btn_save.clicked.connect(self._save_note)

        self._btn_open = QPushButton("開く")
        self._btn_open.setFixedHeight(40)
        self._btn_open.clicked.connect(self._open_note)

        btn_layout.addWidget(self._btn_record)
        btn_layout.addWidget(self._btn_clear)
        btn_layout.addStretch()
        btn_layout.addWidget(self._btn_open)
        btn_layout.addWidget(self._btn_save)
        layout.addLayout(btn_layout)

        # ステータスバー
        self._status_bar = self.statusBar()
        self._status_bar.showMessage("準備中... (初回起動時はモデルのダウンロードが必要です)")

    def _setup_shortcuts(self) -> None:
        # Ctrl+S で保存
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_note)
        # Space で録音トグル
        QShortcut(QKeySequence("F2"), self).activated.connect(self._toggle_recording)

    def _load_models_async(self) -> None:
        """モデルを非同期でロード"""
        self._btn_record.setEnabled(False)
        self._btn_record.setText("モデル読み込み中...")
        self._status_bar.showMessage("モデルをロード中... しばらくお待ちください")

        def _load():
            self._vad.load()
            self._engine.load()
            self._models_loaded = True
            self._btn_record.setEnabled(True)
            self._btn_record.setText("● 録音開始")
            self._status_bar.showMessage("準備完了 - F2 または「録音開始」ボタンで開始")

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def showEvent(self, event) -> None:
        """ウィンドウ表示時にモデルの非同期ロードを開始"""
        super().showEvent(event)
        if not self._models_loaded:
            QTimer.singleShot(200, self._load_models_async)

    def _toggle_recording(self) -> None:
        if not self._models_loaded:
            return
        if self._worker is None or not self._worker.isRunning():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        self._worker = TranscriptionWorker(self._recorder, self._vad, self._engine)
        self._worker.text_ready.connect(self._append_text)
        self._worker.status_changed.connect(self._status_bar.showMessage)
        self._worker.start()
        self._btn_record.setText("■ 録音停止")
        self._btn_record.setStyleSheet("background-color: #e74c3c; color: white;")

    def _stop_recording(self) -> None:
        if self._worker:
            self._worker.stop()
            self._worker.wait()
            self._worker = None
        self._btn_record.setText("● 録音開始")
        self._btn_record.setStyleSheet("")
        self._status_bar.showMessage("停止しました")

    def _append_text(self, text: str) -> None:
        """認識結果をエディタに追記"""
        cursor = self._editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        current = self._editor.toPlainText()
        if current and not current.endswith("\n"):
            cursor.insertText(text)
        else:
            cursor.insertText(text)
        self._editor.setTextCursor(cursor)
        self._editor.ensureCursorVisible()

    def _clear_text(self) -> None:
        reply = QMessageBox.question(
            self, "確認", "テキストをクリアしますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._editor.clear()

    def _save_note(self) -> None:
        filepath, _ = QFileDialog.getSaveFileName(
            self, "メモを保存", str(self._storage.save_dir),
            "テキストファイル (*.txt);;すべてのファイル (*)"
        )
        if filepath:
            Path(filepath).write_text(self._editor.toPlainText(), encoding="utf-8")
            self._status_bar.showMessage(f"保存しました: {filepath}")

    def _open_note(self) -> None:
        filepath, _ = QFileDialog.getOpenFileName(
            self, "メモを開く", str(self._storage.save_dir),
            "テキストファイル (*.txt);;すべてのファイル (*)"
        )
        if filepath:
            text = Path(filepath).read_text(encoding="utf-8")
            self._editor.setPlainText(text)
            self._status_bar.showMessage(f"開きました: {filepath}")

    def closeEvent(self, event) -> None:
        """ウィンドウ終了時に録音を停止する"""
        self._stop_recording()
        event.accept()
