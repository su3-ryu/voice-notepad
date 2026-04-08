# Voice Notepad 引継ぎ資料

## 概要

PyQt6ベースの日本語音声メモ帳アプリ。マイク入力とスピーカー出力をリアルタイムで文字起こしし、OllamaローカルLLMで自動校正する。

## 技術スタック

| カテゴリ | 技術 |
|---------|------|
| UI | PyQt6 |
| 音声認識 | faster-whisper (large-v3モデル) |
| 音声検知 | Silero VAD |
| 音声入力 | sounddevice (WASAPI) |
| 自動校正 | Ollama + qwen2.5:7b (ローカルLLM) |
| 言語 | Python 3.10+ |
| OS | Windows 11 |

## ファイル構成 (1,065行)

```
voice-notepad/
├── main.py                          # エントリーポイント (23行)
├── config.yaml                      # アプリ設定
├── requirements.txt                 # 依存パッケージ
├── launch.bat                       # バッチ起動 (エラーはerror.logに記録)
├── 起動.vbs                         # コンソール非表示で起動
├── .pylintrc                        # pylint設定
├── app/
│   ├── audio/
│   │   ├── recorder.py              # 音声録音 (106行) — マイク/ループバック対応
│   │   └── vad.py                   # 音声区間検出 (90行) — Silero VAD
│   ├── transcription/
│   │   ├── engine.py                # Whisper文字起こし (91行)
│   │   └── postprocess.py           # 後処理・ハルシネーション除去 (88行)
│   ├── correction/
│   │   ├── ollama_client.py         # Ollama APIクライアント (86行)
│   │   └── batch_corrector.py       # バッチ校正コントローラー (103行)
│   ├── storage/
│   │   └── notes.py                 # メモ保存・読込 (47行)
│   └── ui/
│       ├── main_window.py           # メインウィンドウ (411行)
│       └── settings_dialog.py       # 設定ダイアログ (20行・スケルトン)
├── models/                          # Whisperモデル (gitignore)
│   ├── models--Systran--faster-whisper-large-v3/
│   ├── models--Systran--faster-whisper-medium/
│   └── models--Systran--faster-whisper-small/
├── tests/                           # テスト (未実装)
└── bench_score.py                   # 速度・精度最適化スコア計測
```

## データフロー

```
[マイク/スピーカー]
       │
       ▼
AudioRecorder (sounddevice)
  ├── デバイス指定可能 (device パラメータ)
  ├── ステレオ→モノラル自動変換
  └── チャンク単位でキュー送信
       │
       ▼
VoiceActivityDetector (Silero VAD)
  ├── 発話区間の検出 (threshold: 0.4)
  ├── 先頭切れ防止バッファ (直前3チャンク保持)
  └── 発話セグメント完成時にキューへ
       │
       ▼
TranscriptionWorker (QThread)
  ├── VADループ (メインスレッド)
  └── 文字起こしループ (別スレッド・非同期)
       │
       ▼
TranscriptionEngine (faster-whisper)
  ├── large-v3モデル
  ├── beam_size=3, best_of=1 (速度最適化)
  ├── vad_filter=True (内蔵VAD併用)
  ├── 音声パディング 300ms (精度向上)
  └── condition_on_previous_text=True (文脈利用)
       │
       ▼
postprocess()
  ├── 日本語正規化・半角→全角カタカナ
  ├── ハルシネーション除去 (27パターン)
  ├── 短テキストフィルタ (2文字未満除外)
  └── 誤認識パターン補正 (カッコ→括弧 等)
       │
       ▼
QTextEdit に即時表示
       │
       ▼ (15秒ごと or 6セグメント蓄積時)
BatchCorrector → OllamaClient
  ├── Ollama REST API (localhost:11434)
  ├── qwen2.5:7b で文脈ベース校正
  ├── few-shot例付きプロンプト
  └── QTextDocument.find() で元テキストを置換
```

## UI構成

```
┌─────────────────────────────────────┐
│  Voice Notepad - 音声メモ帳         │
├─────────────────────────────────────┤
│ [🎤 マイク] [🔊 スピーカー]          │  ← QTabWidget
├─────────────────────────────────────┤
│                                     │
│  文字起こし結果がここに表示される      │  ← QTextEdit (タブごと独立)
│                                     │
├─────────────────────────────────────┤
│ [● 録音開始] [クリア]    [開く][保存] │  ← 現在のタブに対して動作
├─────────────────────────────────────┤
│ ステータスバー                       │
└─────────────────────────────────────┘
```

- **マイクタブ**: デフォルトマイクデバイスで録音
- **スピーカータブ**: 「ステレオ ミキサー」(WASAPI loopback) で録音
- 録音ボタン (F2): 現在表示中のタブの録音を開始/停止
- Ctrl+S: 保存

## 設定 (config.yaml)

| セクション | 主要設定 | 説明 |
|-----------|---------|------|
| `transcription.model` | `"large-v3"` | Whisperモデル (small/medium/large-v3) |
| `transcription.device` | `"auto"` | GPU/CPU自動選択 |
| `transcription.compute_type` | `"auto"` | GPU→float16, CPU→int8 自動 |
| `audio.chunk_duration_ms` | `32` | VADチャンク長 (16kHz×32ms=512サンプル) |
| `vad.threshold` | `0.4` | 発話検出閾値 |
| `vad.min_silence_duration_ms` | `300` | 発話終了判定の無音時間 |
| `correction.model` | `"qwen2.5:7b"` | Ollama校正モデル |
| `correction.batch_interval_sec` | `15` | 校正バッチ間隔 |

## 外部依存

### Pythonパッケージ (requirements.txt)
```
faster-whisper>=1.0.0
sounddevice>=0.4.6
silero-vad>=5.1
PyQt6>=6.6.0
pydub>=0.25.1
ffmpeg-python>=0.2.0
transformers>=4.40.0
torch>=2.2.0
numpy>=1.26.0
pyyaml>=6.0
```

### 外部ソフトウェア
- **Ollama**: `C:\Users\ryury\AppData\Local\Programs\Ollama\ollama.exe`
  - モデル: qwen2.5:7b (4.7GB)
  - API: http://localhost:11434
  - Ollamaが未起動でもアプリは正常動作（校正機能のみ無効）

## スピーカータブの前提条件

Windowsの「ステレオ ミキサー」が有効である必要がある:
1. サウンド設定 → 録音タブ
2. 空白部分を右クリック →「無効なデバイスの表示」
3. 「ステレオ ミキサー」を右クリック →「有効」

無効な場合、アプリはスピーカータブに手順を表示し、マイクタブは通常通り動作する。

## 開発履歴の要約

| フェーズ | 内容 |
|---------|------|
| 初期状態 | 基本的な音声メモ帳 (pylintエラー55件) |
| autoresearch #1 | pylintエラー 55→0 (コード品質改善) |
| autoresearch #2 | 速度・精度最適化スコア 16→100 (beam_size最適化、非同期化、パディング等) |
| Ollama連携 | 文脈ベース自動校正機能 (バッチ方式、few-shotプロンプト) |
| 2タブ化 | マイク/スピーカーの独立タブ、ループバック自動検出 |

## 既知の制限事項・未実装

- **テスト**: `tests/` ディレクトリは空。ユニットテスト未実装
- **設定ダイアログ**: `settings_dialog.py` はスケルトンのみ (config.yaml直接編集)
- **自動保存**: config.yamlに設定あるが未実装
- **ダークテーマ**: config.yamlに設定あるが未実装
- **Ollama校正の精度**: 「後半復→高反復」など一部の専門用語は修正できない場合がある
- **GPU VRAM**: Whisper large-v3 + Ollama qwen2.5:7b の同時実行には8GB以上のVRAM推奨
- **マイクとスピーカーの同時録音**: 現在の設計では同時録音には対応していない（タブごとに独立して録音開始/停止）
