# 概要

本研究は，ポートフォリオ最適化における共分散行列の推定手法の比較を行う．

## プロジェクト構成

```
Research/
├── config/             # 設定ファイル
│   └── config.yaml     # 実験パラメータ、パス設定
├── data/               # データファイル
│   ├── input/          # 株価データのCSVファイル
│   └── output/         # バックテスト結果とプロット
├── src/                # ソースコード
│   ├── config.py       # 設定読み込みモジュール
│   ├── main.py         # バックテスト実行のメインスクリプト
│   ├── method.py       # アルゴリズム実装
│   ├── backtest_engine.py # バックテストのロジック
│   ├── data_handler.py    # データ読み込みと前処理
│   └── tests/          # ユニットテスト
│       └── test_method.py
├── Thesis/             # 論文 (LaTeX)
├── requirements.txt    # Python 依存ライブラリ一覧
└── README.md           # 本ファイル（説明書）
```

### セットアップと使用方法

1. **ライブラリのインストール:**
   ```bash
   pip install -r requirements.txt
   ```

2. **テストの実行:**
   ```bash
   python -m unittest discover src/tests
   ```

3. **バックテストの実行:**
   ```bash
   python src/main.py
   ```
