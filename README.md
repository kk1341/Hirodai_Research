# ポートフォリオ最適化 研究プロジェクト

本プロジェクトは、共分散行列の推定手法（独自の線形収縮推定器を含む）を用いたポートフォリオ最適化の実装です。

## プロジェクト構成

```
Research/
├── data/               # データ
│   ├── input/          # 入力
│   └── output/         # 出力
├── program/            # ソースコード
│   ├── config.yaml     # 設定ファイル (パス、期間、手法など)
│   ├── main.py         # メインスクリプト
│   ├── method.py       # 実装 (Shrinkage, POET, PCA など)
│   ├── backtest_engine.py # バックテストのロジック
│   ├── data_handler.py    # データ読み込みと前処理
│   └── tests/          # テスト
│       └── test_method.py
├── Thesis/             # 論文
├── requirements.txt    
└── README.md            
```

## セットアップと使用方法

1. **ライブラリのインストール:**
   ```bash
   pip install -r requirements.txt
   ```

2. **テストの実行:**
   ```bash
   python -m unittest discover program/tests
   ```

3. **バックテストの実行:**
   ```bash
   cd program
   python main.py
   ```
