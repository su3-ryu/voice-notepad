voice-notepad について
======================

このフォルダは、音声メモアプリ「voice-notepad」本体のフォルダです。
GitHub で共有する場合は、基本的にこの voice-notepad フォルダを 1 つの
リポジトリとして扱います。


GitHub 共有時の注意
------------------

voice-notepad フォルダは Git リポジトリとして扱います。
GitHub に共有するときは、voice-notepad フォルダ内でリポジトリを作成・管理します。

models/ フォルダは .gitignore で除外されています。
Whisper などのモデルファイルはサイズが大きいため、GitHub には上げない方がよいです。

他の環境で使う場合は、requirements.txt で必要な Python パッケージを入れたうえで、
必要なモデルを別途ダウンロードするか、初回起動時に取得される形にします。


起動に関係する主なファイル
--------------------------

voice-notepad の起動や実行に関係する主なファイルは次の通りです。

- main.py
- app\
- config.yaml
- requirements.txt
- launch.bat
- 起動.bat
- 起動.vbs
