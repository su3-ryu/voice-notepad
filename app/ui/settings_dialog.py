"""
設定ダイアログ（将来の拡張用スケルトン）
"""
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox


class SettingsDialog(QDialog):
    """設定ダイアログ（将来の拡張用スケルトン）"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("設定")
        self.setMinimumWidth(400)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("設定機能は今後実装予定です。\nconfig.yaml を直接編集してください。"))

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)
