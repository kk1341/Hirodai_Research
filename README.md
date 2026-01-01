# 概要

本研究は，ポートフォリオ最適化における共分散行列の推定手法の比較を行う．

## 手法一覧
本プロジェクトで比較・検証を行っている共分散行列の推定手法は以下の通りである．

1.  **Sample Covariance (標本共分散)**
    *   最も標準的な推定手法だが，次元数($N$)がサンプルサイズ($T$)に近い場合，推定精度が著しく低下する（"The Ugly"）．
2.  **PCA (Principal Component Analysis)**
    *   主成分分析を用いた統計的ファクターモデル．主要な共通要因（ファクター）のみを抽出してノイズを除去する（"The Good"）．
3.  **POET (Principal Orthogonal complement Thresholding)**
    *   Fan et al. (2013) による手法．PCAに加え，残差行列に対してスパース推定（閾値処理）を行うことで，よりロバストな推定を行う（"The Better"）．
4.  **Ledoit-Wolf Linear Shrinkage**
    *   標本共分散行列とターゲット行列（単位行列など）の線形結合を用いることで，推定誤差（Bias-Variance Tradeoff）を最適化する手法．
5.  **Nonlinear Shrinkage**
    *   固有値分布に対して非線形な変換を適用し，大次元固有値問題を補正する手法．

## 実験結果の概要
$N$ (銘柄数) を変化させた際の，各手法を用いた最小分散ポートフォリオ(MVP)の Sharpe Ratio (年率換算) の平均値は以下の通りである（試行回数: 各10回）．
※ $N$ が大きくなるにつれて，標本共分散行列(Sample)の性能が劣化し，PCAやPOETなどの構造化された推定手法が優位になる傾向が確認されている．

*(ここに `data/output/experiment_results_summary.csv` の内容を要約して記載予定)*


```
Research/
├── config/             # 設定ファイル
│   └── config.yaml     # 実験パラメータ，パス設定
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
