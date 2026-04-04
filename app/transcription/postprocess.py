"""
文字起こし結果の後処理
日本語テキストの正規化・ハルシネーション除去・句読点補完など
"""
import re

# Whisperが無音時に生成する既知のハルシネーションフレーズ
HALLUCINATION_PATTERNS = [
    "ご視聴ありがとうございました",
    "ご視聴ありがとうございます",
    "チャンネル登録よろしくお願いします",
    "チャンネル登録お願いします",
    "いい動画だと思ったら",
    "高評価お願いします",
    "お疲れ様でした",
    "ありがとうございました",
    "おやすみなさい",
    "Thank you for watching",
    "Thanks for watching",
    "Please subscribe",
    "See you next time",
    "Bye bye",
    "字幕作成",
    "字幕提供",
    "MBS",
    "ではまた",
]


def is_hallucination(text: str) -> bool:
    """ハルシネーション（幻聴）かどうか判定する"""
    cleaned = text.strip().rstrip("。、！？.!?")
    for pattern in HALLUCINATION_PATTERNS:
        if cleaned == pattern or cleaned.endswith(pattern):
            return True
    # 同じ文字の繰り返し（例: "ああああ"）
    if len(cleaned) > 2 and len(set(cleaned)) <= 2:  # pylint: disable=chained-comparison
        return True
    return False


def normalize_japanese(text: str) -> str:
    """日本語テキストの基本正規化"""
    text = text.replace("\u3000", " ")
    text = re.sub(r" {2,}", " ", text)
    return text.strip()


def ensure_punctuation(text: str) -> str:
    """文末句読点がない場合に付与する"""
    if not text:
        return text
    last_char = text[-1]
    punctuation = {"。", "、", "！", "？", "…", ".", "!", "?"}
    if last_char not in punctuation:
        text += "。"
    return text


def postprocess(text: str, add_punctuation: bool = False) -> str:
    """文字起こし結果に後処理を適用する"""
    text = normalize_japanese(text)
    if not text:
        return ""
    if is_hallucination(text):
        return ""
    if add_punctuation:
        text = ensure_punctuation(text)
    return text
