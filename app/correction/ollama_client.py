"""
Ollama REST API クライアント
ローカルLLMを使った文字起こしテキストの校正
"""
import json
import urllib.request
import urllib.error


SYSTEM_PROMPT = (
    "あなたは日本語の音声認識テキストの校正専門家です。"
    "音声認識（Whisper）の誤変換を文脈に基づいて修正してください。\n"
    "ルール:\n"
    "1. 同音異義語の誤変換を修正する（例: セット管→セット間、キー被害→筋肥大）\n"
    "2. 文脈に合わない単語を正しい単語に置き換える\n"
    "3. 「カッコ○○」のような音声読み上げ表現を記号（）に変換する\n"
    "4. 文の意味や構造は変えない。追加・削除しない\n"
    "5. 修正が不要な箇所はそのまま残す\n"
    "6. 修正後のテキストのみを出力する。説明や注釈は一切不要\n"
)


class OllamaClient:
    """Ollama REST API を使ったテキスト校正クライアント"""

    def __init__(self, base_url: str = "http://localhost:11434",
                 model: str = "qwen2.5:7b", timeout: int = 30):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

    def is_available(self) -> bool:
        """Ollamaサーバーが起動しているか確認する"""
        try:
            req = urllib.request.Request(f"{self.base_url}/api/tags")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except (urllib.error.URLError, OSError):
            return False

    def correct_text(self, text: str) -> str:
        """
        テキストをOllamaに送信し、校正結果を返す。

        Args:
            text: 校正対象の文字起こしテキスト

        Returns:
            校正後のテキスト。エラー時は元のテキストをそのまま返す。
        """
        payload = json.dumps({
            "model": self.model,
            "system": SYSTEM_PROMPT,
            "prompt": text,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": len(text) * 3,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/generate",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                corrected = result.get("response", "").strip()
                if not corrected:
                    return text
                # 長さが大幅に変わった場合はLLMが余計な出力をした可能性 → 元テキストを返す
                if len(corrected) > len(text) * 1.5 or len(corrected) < len(text) * 0.5:
                    return text
                return corrected
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            print(f"[OllamaClient] 校正エラー: {e}")
            return text
