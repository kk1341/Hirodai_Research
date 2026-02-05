# Portfolio Optimization in High-Dimensional Settings

本プロジェクトは，高次元・小標本 ($N \gg T$) という現代的な投資環境におけるポートフォリオ最適化問題に対し，**縮小推定 (Shrinkage Estimation)** の有効性を検証するための研究である．

特に，以下の3つのアプローチの比較に焦点を当てている．

1.  **Equal Weight**: 情報を「完全に捨てる（推定しない）」アプローチ
2.  **Factor Models**: 情報を「構造化して圧縮する」アプローチ
3.  **Shrinkage**: 情報を「適切に正則化して活用する」アプローチ

## 手法一覧 (Methods)

本プロジェクトで実装・検証を行っている手法は以下の通りである．

### 1. Benchmark & Baseline
*   **Equal Weight (等配分戦略)**
    *   共分散行列の推定を一切行わず，$1/N$ ずつのウェイトを配分する．
    *   推定誤差が極めて大きい環境における強力なベンチマークとなる．
*   **Sample Covariance (標本共分散)**
    *   最も標準的な推定手法だが，高次元設定では推定誤差により最適化が破綻する．

### 2. Factor Models (Structure-based)
*   **Market Factor (市場ファクター)**
    *   単一の市場インデックス (S&P 500) を共通要因とし，残差は無相関と仮定する．
*   **PCA (Principal Component Analysis)**
    *   主成分分析を用いて統計的な共通要因を抽出する．
*   **POET (Principal Orthogonal complement Thresholding)**
    *   Fan et al. (2013) による手法．PCAの残差行列に対してスパース推定を適用する．

### 3. Shrinkage Estimation (Regularization-based)
*   **Ledoit-Wolf Linear Shrinkage**
    *   標本共分散行列とターゲット行列（単位行列）の線形結合を用いることで，バイアスと分散のトレードオフを最適化する．
*   **Nonlinear Shrinkage**
    *   ランダム行列理論 (RMT) に基づき，固有値分布そのものを非線形に補正する最先端の手法．

## プロジェクト構成 (Directory Structure)

```
Research/
├── config/             # 設定ファイル
│   └── config.yaml     # 実験パラメータ (N, T, Simulation回数など)
├── data/               # データディレクトリ
│   ├── input/          # 株価データ (CSV)
│   └── output/         # 実験結果
│       ├── outputx/    # N=30~500 のシミュレーション結果
│       ├── output1k/   # N=1000 の大規模実験結果
│       └── figures/    # 生成されたプロット画像
├── src/                # ソースコード
│   ├── main.py         # モンテカルロ・シミュレーション実行スクリプト
│   ├── backtest_engine.py # バックテストのコアロジック
│   ├── method.py       # 各推定手法の実装 (Shrinkage, POET等)
│   ├── data_handler.py # データ読み込み・前処理
│   ├── plot_results.py # 結果の可視化・プロット作成
│   └── tests/          # ユニットテスト
├── Thesis/             # 論文関連 (LaTeXソース等)
└── requirements.txt    # 依存ライブラリ
```

## 使用方法 (Usage)

### 1. 環境構築
```bash
pip install -r requirements.txt
```

### 2. 設定の変更
`config/config.yaml` を編集し，シミュレーションのパラメータを設定する．
- `n_values`: 検証する銘柄数 ($N$) のリスト
- `num_simulations`: 各設定ごとの試行回数
- `pca_rank`: ファクターモデルのランク数

### 3. 実験の実行
バックテスト・シミュレーションを実行する．結果は `data/output/` に保存される．
```bash
python src/main.py
```

### 4. 結果の可視化
シミュレーション結果のCSVを読み込み，論文用のグラフを描画する．
```bash
python src/plot_results.py
```

### 5. テストの実行
実装の正当性を検証するためのユニットテストを実行する．
```bash
python -m unittest discover src/tests
```
