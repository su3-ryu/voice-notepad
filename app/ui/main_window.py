"""
メインウィンドウ
PyQt6 ベースの音声メモ帳 UI（マイク/スピーカー 左右2パネル構成）
"""
import queue
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QFileDialog, QMessageBox,
    QSplitter, QLabel
)
from PyQt6.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt6.QtGui import QFont, QKeySequence, QShortcut

from app.audio.recorder import AudioRecorder
from app.audio.vad import VoiceActivityDetector
from app.correction.batch_corrector import BatchCorrector
from app.transcription.engine import TranscriptionEngine
from app.transcription.postprocess import postprocess
from app.storage.notes import NoteStorage


class TranscriptionWorker(QThread):
    """バックグラウンドで録音・VADを行うワーカースレッド"""
    segment_ready = pyqtSignal(object)
    status_changed = pyqtSignal(str)
    error_occurred = pyqtSignal(str)

    def __init__(self, recorder: AudioRecorder, vad: VoiceActivityDetector):
        super().__init__()
        self.recorder = recorder
        self.vad = vad
        self._running = False

    def run(self) -> None:
        """録音・VADのメインループ"""
        self._running = True
        self.recorder.clear_buffer()
        try:
            self.recorder.start()
        except Exception as e:
            self._running = False
            self.error_occurred.emit(f"録音デバイスを開けませんでした: {e}")
            return
        self.status_changed.emit("録音中...")

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
                    self.segment_ready.emit(segment)
            except (RuntimeError, OSError, ValueError) as e:
                print(f"[Worker VAD] エラー: {e}")

        self.recorder.stop()
        self.status_changed.emit("停止中")

    def stop(self) -> None:
        """ワーカースレッドを停止する"""
        self._running = False


class TranscriptionQueueWorker(QThread):
    """マイク/スピーカーの音声セグメントを1本ずつ文字起こしするワーカー"""

    text_ready = pyqtSignal(str, object)
    status_changed = pyqtSignal(str)

    def __init__(self, engine: TranscriptionEngine):
        super().__init__()
        self.engine = engine
        self._queue: queue.Queue = queue.Queue()
        self._running = False

    def enqueue(self, source: str, editor: QTextEdit, segment: np.ndarray) -> None:
        """文字起こし待ち行列に音声セグメントを追加する"""
        self._queue.put((source, editor, segment))
        waiting = self._queue.qsize()
        if waiting >= 3:
            self.status_changed.emit(f"認識待ち: {waiting}件")

    def run(self) -> None:
        """共有キューから1件ずつ取り出して文字起こしする"""
        self._running = True
        while self._running or not self._queue.empty():
            item = self._queue.get()
            if item is None:
                continue

            source, editor, segment = item
            try:
                waiting = self._queue.qsize()
                suffix = f" / 待ち {waiting}件" if waiting else ""
                self.status_changed.emit(f"⏳ {source}を認識中...{suffix}")
                start = time.perf_counter()
                text = self.engine.transcribe(segment)
                text = postprocess(text)
                if text:
                    self.text_ready.emit(text, editor)
                elapsed = time.perf_counter() - start
                self.status_changed.emit(
                    f"🔴 録音中 - 話してください（直近認識 {elapsed:.1f}秒）"
                )
            except (RuntimeError, OSError, ValueError) as e:
                print(f"[TranscriptionQueueWorker] エラー: {e}")

    def stop(self) -> None:
        """ワーカーを停止する。キューに残ったセグメントは処理してから終了する。"""
        self._running = False
        self._queue.put(None)


class TranscriptionPanel(QWidget):
    """マイクまたはスピーカー用の文字起こしパネル"""

    record_toggled = pyqtSignal()

    def __init__(self, icon: str, label: str, font: QFont,
                 parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.label = label

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ヘッダー行（ラベル＋録音ボタン）
        header = QHBoxLayout()
        title = QLabel(f"{icon} {label}")
        title.setFont(QFont(font.family(), font.pointSize() + 1))
        title.setStyleSheet("font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        self._btn_record = QPushButton("● 録音開始")
        self._btn_record.setFixedHeight(32)
        self._btn_record.setFixedWidth(120)
        self._btn_record.clicked.connect(self.record_toggled.emit)
        header.addWidget(self._btn_record)
        layout.addLayout(header)

        # テキストエディタ
        self._editor = QTextEdit()
        self._editor.setFont(font)
        self._editor.setPlaceholderText(
            f"ここに{label}の文字起こし結果が表示されます。\n"
            f"「録音開始」ボタンを押してください。"
        )
        layout.addWidget(self._editor)

    @property
    def editor(self) -> QTextEdit:
        """テキストエディタを返す"""
        return self._editor

    @property
    def record_button(self) -> QPushButton:
        """録音ボタンを返す"""
        return self._btn_record

    def set_recording(self, recording: bool) -> None:
        """録音状態に応じてボタン表示を更新"""
        if recording:
            self._btn_record.setText("■ 録音停止")
            self._btn_record.setStyleSheet(
                "background-color: #e74c3c; color: white;")
        else:
            self._btn_record.setText("● 録音開始")
            self._btn_record.setStyleSheet("")

    def set_enabled(self, enabled: bool) -> None:
        """録音ボタンの有効/無効を切り替え"""
        self._btn_record.setEnabled(enabled)

    def set_loading(self) -> None:
        """モデル読み込み中の表示"""
        self._btn_record.setEnabled(False)
        self._btn_record.setText("読み込み中...")


class MainWindow(QMainWindow):
    """Voice Notepad メインウィンドウ（左右2パネル構成）"""

    def __init__(self):
        super().__init__()
        self._load_config()
        self._setup_components()
        self._setup_ui()
        self._setup_shortcuts()
        self._worker_mic: Optional[TranscriptionWorker] = None
        self._worker_spk: Optional[TranscriptionWorker] = None
        self._models_loaded = False
        self._active_panel: Optional[TranscriptionPanel] = None

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

        # スピーカー用（既定の出力デバイスを WASAPI ループバック録音）
        loopback_dev = AudioRecorder.find_loopback_device()
        self._loopback_available = loopback_dev is not None
        self._recorder_spk = AudioRecorder(
            sample_rate=sr, channels=ch, chunk_duration_ms=chunk_ms,
            device=loopback_dev, backend="loopback",
        )
        self._vad_spk = VoiceActivityDetector(
            threshold=cfg["vad"]["threshold"],
            min_speech_duration_ms=cfg["vad"]["min_speech_duration_ms"],
            min_silence_duration_ms=cfg["vad"]["min_silence_duration_ms"],
            max_speech_duration_ms=cfg["vad"].get("speaker_max_speech_duration_ms", 2500),
            sample_rate=sr,
        )

        # 共有コンポーネント
        self._engine = TranscriptionEngine()
        self._transcription_queue_worker = TranscriptionQueueWorker(self._engine)
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

        # 左右2パネル（スプリッター）
        splitter = QSplitter(Qt.Orientation.Horizontal)

        self._panel_mic = TranscriptionPanel("🎤", "マイク", font)
        self._panel_spk = TranscriptionPanel("🔊", "スピーカー", font)

        splitter.addWidget(self._panel_mic)
        splitter.addWidget(self._panel_spk)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # スピーカーパネルが使えない場合の表示
        if not self._loopback_available:
            self._panel_spk.editor.setPlaceholderText(
                "スピーカーのループバック録音を利用できません。\n"
                "soundcard パッケージがインストールされ、Windows の既定の出力デバイスが有効か確認してください。"
            )
            self._panel_spk.set_enabled(False)

        # 各パネルの録音ボタンを接続
        self._panel_mic.record_toggled.connect(
            lambda: self._toggle_recording_for("mic"))
        self._panel_spk.record_toggled.connect(
            lambda: self._toggle_recording_for("spk"))

        # エディタクリック時にアクティブパネルを追跡
        self._panel_mic.editor.installEventFilter(self)
        self._panel_spk.editor.installEventFilter(self)
        self._active_panel = self._panel_mic

        # ボタン行（共通操作）
        btn_layout = QHBoxLayout()

        self._btn_clear = QPushButton("クリア")
        self._btn_clear.setFixedHeight(36)
        self._btn_clear.clicked.connect(self._clear_text)

        self._btn_open = QPushButton("開く")
        self._btn_open.setFixedHeight(36)
        self._btn_open.clicked.connect(self._open_note)

        self._btn_save = QPushButton("保存")
        self._btn_save.setFixedHeight(36)
        self._btn_save.clicked.connect(self._save_note)

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
        self._transcription_queue_worker.text_ready.connect(self._append_text)
        self._transcription_queue_worker.status_changed.connect(
            self._status_bar.showMessage)
        self._transcription_queue_worker.start()

    def _setup_shortcuts(self) -> None:
        """キーボードショートカットの設定"""
        QShortcut(QKeySequence("Ctrl+S"), self).activated.connect(self._save_note)
        QShortcut(QKeySequence("F2"), self).activated.connect(
            lambda: self._toggle_recording_for("mic"))
        QShortcut(QKeySequence("F3"), self).activated.connect(
            lambda: self._toggle_recording_for("spk"))

    def eventFilter(self, obj, event) -> bool:
        """エディタクリック時にアクティブパネルを追跡"""
        if event.type() == event.Type.FocusIn:
            if obj is self._panel_mic.editor:
                self._active_panel = self._panel_mic
            elif obj is self._panel_spk.editor:
                self._active_panel = self._panel_spk
        return super().eventFilter(obj, event)

    def _load_models_async(self) -> None:
        """モデルを非同期でロード"""
        self._panel_mic.set_loading()
        self._panel_spk.set_loading()
        self._status_bar.showMessage("モデルをロード中... しばらくお待ちください")

        def _load():
            self._vad_mic.load()
            self._vad_spk.load()
            self._engine.load()
            self._models_loaded = True
            self._panel_mic.set_recording(False)
            self._panel_mic.set_enabled(True)
            if self._loopback_available:
                self._panel_spk.set_recording(False)
                self._panel_spk.set_enabled(True)
            self._status_bar.showMessage(
                "準備完了 - F2(マイク) / F3(スピーカー) または各パネルのボタンで開始")

        thread = threading.Thread(target=_load, daemon=True)
        thread.start()

    def showEvent(self, event) -> None:
        """ウィンドウ表示時にモデルの非同期ロードを開始"""
        super().showEvent(event)
        if not self._models_loaded:
            QTimer.singleShot(200, self._load_models_async)

    # --- 録音制御 ---

    def _toggle_recording_for(self, source: str) -> None:
        """指定ソースの録音を開始/停止"""
        if not self._models_loaded:
            return
        is_mic = source == "mic"
        worker = self._worker_mic if is_mic else self._worker_spk

        if worker is None or not worker.isRunning():
            self._start_recording(is_mic)
        else:
            self._stop_recording(is_mic)

    def _start_recording(self, is_mic: bool) -> None:
        """指定ソースの録音を開始"""
        if not is_mic and not self._loopback_available:
            self._status_bar.showMessage("スピーカーのループバック録音を利用できません")
            return

        recorder = self._recorder_mic if is_mic else self._recorder_spk
        vad = self._vad_mic if is_mic else self._vad_spk
        panel = self._panel_mic if is_mic else self._panel_spk
        source = "マイク" if is_mic else "スピーカー"

        worker = TranscriptionWorker(recorder, vad)
        worker.segment_ready.connect(
            lambda segment, s=source, e=panel.editor:
            self._transcription_queue_worker.enqueue(s, e, segment))
        worker.status_changed.connect(self._status_bar.showMessage)
        worker.error_occurred.connect(
            lambda msg, m=is_mic: self._on_recording_error(msg, m))
        worker.start()

        if is_mic:
            self._worker_mic = worker
        else:
            self._worker_spk = worker

        panel.set_recording(True)
        self._corrector.start()

    def _stop_recording(self, is_mic: bool) -> None:
        """指定ソースの録音を停止"""
        worker = self._worker_mic if is_mic else self._worker_spk
        panel = self._panel_mic if is_mic else self._panel_spk

        if worker:
            worker.stop()
            worker.wait()
            if is_mic:
                self._worker_mic = None
            else:
                self._worker_spk = None

        panel.set_recording(False)

        # 両方停止していたら校正も停止
        if self._worker_mic is None and self._worker_spk is None:
            self._corrector.stop()

        self._status_bar.showMessage(
            f"{'マイク' if is_mic else 'スピーカー'}を停止しました")

    def _on_recording_error(self, message: str, is_mic: bool) -> None:
        """録音エラー時にUIをリセットしてユーザーに通知"""
        panel = self._panel_mic if is_mic else self._panel_spk
        panel.set_recording(False)

        if is_mic:
            self._worker_mic = None
        else:
            self._worker_spk = None

        if self._worker_mic is None and self._worker_spk is None:
            self._corrector.stop()

        source = "マイク" if is_mic else "スピーカー"
        self._status_bar.showMessage(f"❌ {source}の録音開始に失敗しました")
        QMessageBox.warning(self, f"{source}エラー", message)

    # --- テキスト操作 ---

    def _active_editor(self) -> QTextEdit:
        """アクティブなパネルのエディタを返す"""
        if self._active_panel is not None:
            return self._active_panel.editor
        return self._panel_mic.editor

    def _append_text(self, text: str, editor: QTextEdit) -> None:
        """認識結果をエディタに追記"""
        cursor = editor.textCursor()
        cursor.movePosition(cursor.MoveOperation.End)
        cursor.insertText(text)
        editor.setTextCursor(cursor)
        editor.ensureCursorVisible()
        self._corrector.add_text(text)

    def _apply_correction(self, original: str, corrected: str) -> None:
        """校正結果をエディタに反映する（両パネルを検索）"""
        for panel in (self._panel_mic, self._panel_spk):
            doc = panel.editor.document()
            cursor = doc.find(original)
            if not cursor.isNull():
                cursor.insertText(corrected)
                self._status_bar.showMessage("✅ 自動校正を適用しました")
                return

    def _clear_text(self) -> None:
        """アクティブなパネルのテキストをクリア"""
        reply = QMessageBox.question(
            self, "確認", "テキストをクリアしますか？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply == QMessageBox.StandardButton.Yes:
            self._active_editor().clear()

    def _save_note(self) -> None:
        """アクティブなパネルのテキストを保存"""
        filepath, _ = QFileDialog.getSaveFileName(
            self, "メモを保存", str(self._storage.save_dir),
            "テキストファイル (*.txt);;すべてのファイル (*)"
        )
        if filepath:
            Path(filepath).write_text(
                self._active_editor().toPlainText(), encoding="utf-8"
            )
            self._status_bar.showMessage(f"保存しました: {filepath}")

    def _open_note(self) -> None:
        """ファイルを開いてアクティブなパネルに表示"""
        filepath, _ = QFileDialog.getOpenFileName(
            self, "メモを開く", str(self._storage.save_dir),
            "テキストファイル (*.txt);;すべてのファイル (*)"
        )
        if filepath:
            text = Path(filepath).read_text(encoding="utf-8")
            self._active_editor().setPlainText(text)
            self._status_bar.showMessage(f"開きました: {filepath}")

    def closeEvent(self, event) -> None:
        """ウィンドウ終了時に全録音を停止する"""
        for worker in (self._worker_mic, self._worker_spk):
            if worker:
                worker.stop()
                worker.wait()
        self._transcription_queue_worker.stop()
        self._transcription_queue_worker.wait()
        event.accept()
