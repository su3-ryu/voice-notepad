"""
メモの保存・読み込みモジュール
テキストファイル形式でメモを管理する
"""
from datetime import datetime
from pathlib import Path
from typing import Optional, Union


class NoteStorage:
    """メモをテキストファイルで保存・読み込みするクラス"""

    def __init__(self, save_dir: str = "notes"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def save(self, content: str, filename: Optional[str] = None) -> Path:
        """
        メモを保存する

        Args:
            content: 保存するテキスト
            filename: ファイル名（省略時はタイムスタンプで自動生成）

        Returns:
            保存したファイルのパス
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"note_{timestamp}.txt"

        filepath = self.save_dir / filename
        filepath.write_text(content, encoding="utf-8")
        return filepath

    def load(self, filepath: Union[str, Path]) -> str:
        """メモをファイルから読み込む"""
        return Path(filepath).read_text(encoding="utf-8")

    def list_notes(self) -> list:
        """保存されているメモファイルの一覧を返す（新しい順）"""
        files = sorted(self.save_dir.glob("*.txt"), key=lambda p: p.stat().st_mtime, reverse=True)
        return files

    def delete(self, filepath: Union[str, Path]) -> None:
        """メモを削除する"""
        Path(filepath).unlink(missing_ok=True)
