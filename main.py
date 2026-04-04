"""
Voice Notepad - エントリーポイント
音声入力をリアルタイムで文字起こしするメモ帳アプリ
"""
import sys
from PyQt6.QtWidgets import QApplication
from app.ui.main_window import MainWindow


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Voice Notepad")
    app.setOrganizationName("VoiceNotepad")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
