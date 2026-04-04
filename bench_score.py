"""
速度・精度最適化スコアの計測スクリプト
各最適化項目をチェックし、スコアを出力する（高いほど良い）
"""
import ast
import re
import sys
import yaml


def check_engine_optimizations(source: str) -> dict:
    """engine.pyの最適化項目をチェック"""
    checks = {}

    # 1. beam_size最適化 (1-3が速い、5は遅い)
    m = re.search(r'beam_size\s*=\s*(\d+)', source)
    if m:
        val = int(m.group(1))
        checks['beam_size_optimized'] = val <= 3
    else:
        checks['beam_size_optimized'] = False

    # 2. best_of at temp=0 は無意味 → best_of=1であるべき
    m_temp = re.search(r'temperature\s*=\s*([\d.]+)', source)
    m_best = re.search(r'best_of\s*=\s*(\d+)', source)
    if m_temp and m_best:
        temp = float(m_temp.group(1))
        best = int(m_best.group(1))
        checks['best_of_efficient'] = not (temp == 0.0 and best > 1)
    else:
        checks['best_of_efficient'] = False

    # 3. vad_filter有効化 (faster-whisper内蔵VAD)
    checks['vad_filter_enabled'] = 'vad_filter=True' in source

    # 4. cpu_threads設定
    checks['cpu_threads_set'] = 'cpu_threads' in source

    # 5. condition_on_previous_text (精度向上)
    checks['context_conditioning'] = 'condition_on_previous_text=True' in source

    # 6. 音声パディング (精度向上)
    checks['audio_padding'] = 'pad' in source.lower() and 'np.zeros' in source

    return checks


def check_worker_optimizations(source: str) -> dict:
    """main_window.pyのワーカー最適化をチェック"""
    checks = {}

    # 7. 文字起こしの非同期化 (別スレッドまたはキュー)
    checks['async_transcription'] = ('_transcription_queue' in source
                                      or 'transcription_thread' in source
                                      or '_pending_segments' in source)

    # 8. 録音バッファクリア (録音開始時)
    checks['buffer_clear_on_start'] = 'clear_buffer' in source

    return checks


def check_vad_optimizations(source: str) -> dict:
    """vad.pyの最適化をチェック"""
    checks = {}

    # 9. 発話開始前のバッファ保持 (先頭切れ防止)
    checks['pre_speech_buffer'] = 'pre_buffer' in source or '_pre_speech' in source

    return checks


def check_postprocess_optimizations(source: str) -> dict:
    """postprocess.pyの最適化をチェック"""
    checks = {}

    # 10. 繰り返しフレーズ検出 (精度向上)
    checks['repetition_detection'] = 'repetit' in source.lower() or '繰り返し' in source

    # 11. 短すぎるテキストのフィルタ
    checks['short_text_filter'] = 'len(' in source and ('< 2' in source or '<= 1' in source or '< 3' in source)

    return checks


def check_config_optimizations(config_path: str) -> dict:
    """config.yamlの最適化をチェック"""
    checks = {}
    with open(config_path, encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    vad = cfg.get('vad', {})
    # 12. VADの沈黙閾値 (短いほど応答が速い、200-500msが適切)
    silence_ms = vad.get('min_silence_duration_ms', 700)
    checks['silence_threshold_tuned'] = 200 <= silence_ms <= 500

    return checks


def main():
    files = {
        'engine': 'app/transcription/engine.py',
        'worker': 'app/ui/main_window.py',
        'vad': 'app/audio/vad.py',
        'postprocess': 'app/transcription/postprocess.py',
        'config': 'config.yaml',
    }

    sources = {}
    for key, path in files.items():
        if key != 'config':
            with open(path, encoding='utf-8') as f:
                sources[key] = f.read()

    all_checks = {}
    all_checks.update(check_engine_optimizations(sources['engine']))
    all_checks.update(check_worker_optimizations(sources['worker']))
    all_checks.update(check_vad_optimizations(sources['vad']))
    all_checks.update(check_postprocess_optimizations(sources['postprocess']))
    all_checks.update(check_config_optimizations(files['config']))

    passed = sum(1 for v in all_checks.values() if v)
    total = len(all_checks)
    score = int(passed / total * 100)

    for name, result in all_checks.items():
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")

    print(f"\nScore: {score} ({passed}/{total})")


if __name__ == '__main__':
    main()
