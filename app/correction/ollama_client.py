"""
Ollama REST API クライアント
ローカルLLMを使った文字起こしテキストの校正
"""
import json
import urllib.request
import urllib.error


SYSTEM_PROMPT = (
    "あなたは日本語の音声認識(Whisper)の誤変換を校正する専門家です。\n"
    "音が似ているが文脈に合わない単語を、正しい単語に置き換えてください。\n\n"
    "ルール:\n"
    "1. 文脈に合わない語は音が近い正しい語に直す\n"
    "2. 文の構造・語順は変えない。文を追加・削除しない\n"
    "3. 修正後のテキストのみ出力。説明不要\n\n"
    "校正例:\n"
    "入力: キー被害が目的ならば\n"
    "出力: 筋肥大が目的ならば\n\n"
    "入力: 禁止給力向上がもう的ならば後半復回数に取り組む\n"
    "出力: 筋持久力向上が目的ならば高反復回数に取り組む\n\n"
    "入力: セット管に十分な休息をはさむ\n"
    "出力: セット間に十分な休息をはさむ\n\n"
    "入力: サルコペニアカッコカレーや疾患\n"
    "出力: サルコペニア（加齢や疾患\n"
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
