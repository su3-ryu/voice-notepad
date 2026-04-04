"""
メインウィンドウ
PyQt6 ベースの音声メモ帳 UI（マイク/スピーカー 2タブ構成）
"""
import queue
import threading
from pathlib import Path
from typing import Optional

import yaml
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QMessageBox,
    QTabWidget
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QKeySequence, QShortcut

from app.audio.recorder import AudioRecorder
from app.audio.vad import VoiceActivityDetector
from app.correction.batch_corrector import BatchCorrector
from app.transcription.engine import TranscriptionEngine
from app.transcription.postprocess import postprocess
from app.storage.notes import NoteStorage


class TranscriptionWorker(QThread):
    """バックグラウンドで録音・VAD・文字起こしを行うワーカースレッド"""
    text_ready = pyqtSignal(str)
    status_changed = pyqtSignal(str)

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

        transcription_thread = threading.Thread(
            target=self._transcription_loop, daemon=True
        )
        transcription_thread.start()

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

        self._pending_segments.put(None)
        transcription_thread.join(timeout=5.0)
        self.recorder.stop()
        self.status_changed.emit("停止中")

    def _transcription_loop(self) -> None:
        """文字起こし専用ループ"""
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


class TranscriptionTab(QWidget):
    """マイクまたはスピーカー用の文字起こしタブ"""

    def __init__(self, label: str, font: QFont, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.label = label
        self._editor = QTextEdit()
        self._editor.setFont(font)
        self._editor.setPlaceholderText(
            f"ここに{label}の文字起こし結果が表示されます。\n"
            f"「録音開始」ボタンを押してください。"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 8, 0, 0)
        layout.addWidget(self._editor)

    @property
    def editor(self) -> QTextEdit:
        """テキストエディタを返す"""
        return self._editor


class MainWindow(QMainWindow):
    """Voice Notepad メインウィンドウ（2タブ構成）"""

    def __init__(self):
        super().__init__()
        self._load_config()
        self._setup_components()
        self._setup_ui()
        self._setup_shortcuts()
        self._worker_mic: Optional[TranscriptionWorker] = None
        self._worker_spk: Optional[TranscriptionWorker] = None
        self._models_loaded = False

    def _load_config(self) -> None:
        with open("config.yaml", encoding="utf-8") as f:
            self._config = yaml.safe_load(f)

    def _setup_components(self) -> None:
        """オーディオ・VAD・エンジン・校正コンポーネントの初期化"""
        cfg = self._config
        sr = cfg["audio"]["sample_rate"]
        ch = cfg["audio"]["channels"]
        chunk_ms = cfg["audio"]["chunk_duration_ms"]

        # マイク用
        self._recorder_mic = AudioRecorder(
            sample_rate=sr, channels=ch, chunk_duration_ms=chunk_ms
        )
        self._vad_mic = VoiceActivityDetector(
            threshold=cfg["vad"]["threshold"],
            min_speech_duration_ms=cfg["vad"]["min_speech_duration_ms"],
            min_silence_duration_ms=cfg["vad"]["min_silence_duration_ms"],
            sample_rate=sr,
        )

        # スピーカー用（ステレオミキサー自動検出）
        loopback_dev = AudioRecorder.find_loopback_device()
        self._loopback_available = loopback_dev is not None
        self._recorder_spk = AudioRecorder(
            sample_rate=sr, channels=ch, chunk_duration_ms=chunk_ms,
            device=loopback_dev,
        )
        self._vad_spk = VoiceActivityDetector(
            threshold=cfg["vad"]["threshold"],
            min_speech_duration_ms=cfg["vad"]["min_speech_duration_ms"],
            min_silence_duration_ms=cfg["vad"]["min_silence_duration_ms"],
            sample_rate=sr,
        )

        # 共有コンポーネント
        self._engine = TranscriptionEngine()
        self._storage = NoteStorage(
            save_dir=cfg["storage"].get("save_directory", "notes")
        )
        self._corrector = BatchCorrector(cfg.get("correction", {}))

    def _setup_ui(self) -> None:
        """UIの構築"""
        self.setWindowTitle("Voice Notepad - 音声メモ帳")
        cfg_ui = self._config["ui"]
        self.resize(cfg_ui["window_width"], cfg_ui["window_height"])
        font = QFont(cfg_ui["font_family"], cfg_ui["font_size"])

        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)
        layout.setContentsMargins(12, 12, 12, 12)

        # タブウィジェット
        self._tabs = QTabWidget()
        self._tab_mic = TranscriptionTab("マイク", font)
        self._tab_spk = TranscriptionTab("スピーカー", font)
        self._tabs.addTab(self._tab_mic, "🎤 マイク")
        self._tabs.addTab(self._tab_spk, "🔊 スピーカー")
        layout.addWidget(self._tabs)

        # スピーカータブが使えない場合の表示
        if not self._loopback_available:
            self._tab_spk.editor.setPlaceholderText(
                "ステレオミキサーが見つかりません。\n"
                "Windowsサウンド設定で「ステレオ ミキサー」を有効にしてください。\n\n"
                "手順: サウンド設定 → 録音 → ステレオ ミキサー → 有効"
            )

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

        # 校正シグナル接続
        self._corrector.correction_ready.connect(self._apply_correction)
        self._corrector.status_changed.connect(self._status_bar.showMessage)

        # タブ切替時にボタン表示を更新
        self._tabs.currentChanged.connect(self._on_tab_changed)

    def _setup_shortcuts(self) -> None:
        """キーボードショートカットの設定"""
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_note)
        QShortcut(QKeySequence("F2"), self).activated.connect(self._toggle_recording)

    def _load_models_async(self) -> None:
        """モデルを非同期でロード"""
        self._btn_record.setEnabled(False)
        self._btn_record.setText("モデル読み込み中...")
        self._status_bar.showMessage("モデルをロード中... しばらくお待ちください")

        def _load():
            self._vad_mic.load()
            self._vad_spk.load()
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

    # --- 現在のタブに応じたヘルパー ---

    def _current_is_mic(self) -> bool:
        """現在のタブがマイクかどうか"""
        return self._tabs.currentIndex() == 0

    def _current_worker(self) -> Optional[TranscriptionWorker]:
        """現在のタブのワーカーを返す"""
        return self._worker_mic if self._current_is_mic() else self._worker_spk

    def _current_editor(self) -> QTextEdit:
        """現在のタブのエディタを返す"""
        if self._current_is_mic():
            return self._tab_mic.editor
        return self._tab_spk.editor

    # --- 録音制御 ---

    def _toggle_recording(self) -> None:
        """現在のタブの録音を開始/停止"""
        if not self._models_loaded:
            return
        worker = self._current_worker()
        if worker is None or not worker.isRunning():
            self._start_recording()
        else:
            self._stop_recording()

    def _start_recording(self) -> None:
        """現在のタブの録音を開始"""
        is_mic = self._current_is_mic()

        if not is_mic and not self._loopback_available:
            self._status_bar.showMessage("ステレオミキサーが見つかりません")
            return

        recorder = self._recorder_mic if is_mic else self._recorder_spk
        vad = self._vad_mic if is_mic else self._vad_spk
        editor = self._current_editor()

        worker = TranscriptionWorker(recorder, vad, self._engine)
        worker.text_ready.connect(lambda text, e=editor: self._append_text(text, e))
        worker.status_changed.connect(self._status_bar.showMessage)
        worker.start()

        if is_mic:
            self._worker_mic = worker
        else:
            self._worker_spk = worker

        self._corrector.start()
        self._btn_record.setText("■ 録音停止")
        self._btn_record.setStyleSheet("background-color: #e74c3c; color: white;")

    def _stop_recording(self) -> None:
        """現在のタブの録音を停止"""
        is_mic = self._current_is_mic()
        worker = self._worker_mic if is_mic else self._worker_spk

        if worker:
            worker.stop()
            worker.wait()
            if is_mic:
                self._worker_mic = None
            else:
                self._worker_spk = None

        self._corrector.stop()
        self._btn_record.setText("● 録音開始")
        self._btn_record.setStyleSheet("")
        self._status_bar.showMessage("停止しました")

    def _on_tab_changed(self) -> None:
        """タブ切替時にボタン表示を更新"""
        worker = self._current_worker()
        if worker is not None and worker.isRunning():
            self._btn_record.setText("■ 録音停止")
            self._btn_record.setStyleSheet("background-color: #e74c3c; color: white;")
        else:
            self._btn_record.setText("● 録音開始")
            self._btn_record.setStyleSheet("")

    # --- テキスト操作 ---

    def _append_text(self, text: str, editor: QTextEdit) -> None:
        """認識結果をエディタに追記"""
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        current = editor.toPlainText()
        if current and not current.endswith("\n"):
            cursor.insertText(text)
        else:
            cursor.insertText(text)
        editor.setTextCursor(cursor)
        editor.ensureCursorVisible()
        self._corrector.add_text(text)

    def _apply_correction(self, original: str, corrected: str) -> None:
        """校正結果をエディタに反映する（両タブを検索）"""
        for tab in (self._tab_mic, self._tab_spk):
            doc = tab.editor.document()
            cursor = doc.find(original)
            if not cursor.isNull():
                cursor.insertText(corrected)
                self._status_bar.showMessage("✅ 自動校正を適用しました")
                return

    def _clear_text(self) -> None:
        """現在のタブのテキストをクリア"""
        reply = QMessageBox.question(
            self, "確認", "テキストをクリアしますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._current_editor().clear()

    def _save_note(self) -> None:
        """現在のタブのテキストを保存"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "メモを保存", str(self._storage.save_dir),
            "テキストファイル (*.txt);;すべてのファイル (*)"
        )
        if filepath:
            Path(filepath).write_text(
                self._current_editor().toPlainText(), encoding="utf-8"
            )
            self._status_bar.showMessage(f"保存しました: {filepath}")

    def _open_note(self) -> None:
        """ファイルを開いて現在のタブに表示"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "メモを開く", str(self._storage.save_dir),
            "テキストファイル (*.txt);;すべてのファイル (*)"
        )
        if filepath:
            text = Path(filepath).read_text(encoding="utf-8")
            self._current_editor().setPlainText(text)
            self._status_bar.showMessage(f"開きました: {filepath}")

    def closeEvent(self, event) -> None:
        """ウィンドウ終了時に全録音を停止する"""
        for worker in (self._worker_mic, self._worker_spk):
            if worker:
                worker.stop()
                worker.wait()
        event.accept()
