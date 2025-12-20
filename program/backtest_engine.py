import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import method
from portfolio_utils import calculate_mvp_weights, calculate_sharpe_ratio

def run_backtest(retx_data, train_duration, retx_cols, output_dir=None, pca_rank=3, silent=False):
    """
    ロール・オーバー・ウィンドウによるバックテストを実行し、シャープ・レシオを比較する。
    
    Parameters
    ----------
    silent : bool
        Trueの場合、進捗バーや詳細なprint出力を抑制する。
    output_dir : str or None
        Noneの場合、ファイル保存を行わない（実験用）。
    
    Returns
    -------
    performance_summary : pd.DataFrame
        各手法のパフォーマンス指標を含むDataFrame。
    """
    T_total, N = retx_data.shape
    num_test_steps = T_total - train_duration

    # 各手法のリターンを格納する辞書
    returns_storage = {
        "Sample": [],
        "MarketFactor": [], # 全銘柄平均をファクターとする
        "PCA": [],
        "POET": [],
        "LinearShrinkage": [], # method.py実装の独自収縮
        "NonlinearShrinkage": [],
    }

    if not silent:
        print(f"\n--- バックテスト実行 (T_train={train_duration} / N={N}) ---")
        print(f"総ステップ数: {num_test_steps} 回のテストを実行")
        print("比較対象: Sample, MarketFactor, PCA, POET, LinearShrinkage, NonlinearShrinkage")

    iterator = range(num_test_steps)
    if not silent:
        iterator = tqdm(iterator, desc="Backtest Progress")

    for i in iterator:
        
        # 1. 訓練期間の抽出
        train_retx = retx_data[i : i + train_duration, :]
        
        # テスト期間のリターン (次の日のリターン)
        test_return = retx_data[i + train_duration, :]

        # --- 各手法で共分散行列を推定し、ウェイトを計算 ---
        estimators = {}
        
        # (1) Sample Covariance
        try:
            estimators["Sample"] = method.sample_covariance(train_retx)
        except:
            estimators["Sample"] = None

        # (2) Market Factor (Known Factor Proxy)
        try:
            F_market = np.mean(train_retx, axis=1, keepdims=True)
            estimators["MarketFactor"] = method.factor_covariance_known(train_retx, F_market)
        except:
            estimators["MarketFactor"] = None

        # (3) PCA Factor
        try:
            estimators["PCA"] = method.pca_factor_covariance(train_retx, K=pca_rank)
        except:
            estimators["PCA"] = None

        # (4) POET
        try:
            estimators["POET"] = method.poet_covariance(train_retx, K=pca_rank)
        except:
            estimators["POET"] = None
            
        # (5) Linear Shrinkage (独自実装)
        try:
            # method.py に移動したためインポート元注意
            # method.linear_shrinkage_identity returns (S, Sigma_hat)
            _, sigma_sh = method.linear_shrinkage_identity(train_retx)
            estimators["LinearShrinkage"] = sigma_sh
        except:
            estimators["LinearShrinkage"] = None
            
        # (6) Nonlinear Shrinkage
        try:
            estimators["NonlinearShrinkage"] = method.nonlinear_shrinkage_covariance(train_retx)
        except:
            estimators["NonlinearShrinkage"] = None

        # --- ウェイト計算とリターン計測 ---
        for name, sigma in estimators.items():
            w = None
            if sigma is not None:
                try:
                    w = calculate_mvp_weights(sigma)
                except np.linalg.LinAlgError:
                    w = None
            
            # エラー時やNoneの場合は等配分
            if w is None:
                w = np.ones(N) / N
            
            # ポートフォリオリターン
            r_p = np.dot(test_return, w)
            returns_storage[name].append(r_p)

    if not silent:
        print("\n--- バックテスト完了。結果を集計中... ---")

    # --- 結果の集計とCSV出力 ---
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
        # 1. リターン時系列の保存
        df_returns = pd.DataFrame(returns_storage)
        returns_csv_path = os.path.join(output_dir, "backtest_returns.csv")
        df_returns.to_csv(returns_csv_path, index=False)
        if not silent:
            print(f"リターン時系列を保存しました: {returns_csv_path}")

    # 2. パフォーマンス指標の計算
    performance_records = []
    
    for name, ret_list in returns_storage.items():
        if not ret_list: continue
        ret_arr = np.array(ret_list)
        
        # 年率リターン (単利換算: 平均日次リターン * 252)
        ann_return = np.mean(ret_arr) * 252
        
        # 年率ボラティリティ (日次標準偏差 * sqrt(252))
        ann_vol = np.std(ret_arr, ddof=1) * np.sqrt(252)
        
        # シャープ・レシオ (日次ベース)
        sharpe = calculate_sharpe_ratio(ret_arr)
        
        # 年率換算
        ann_sharpe = sharpe * np.sqrt(252)

        performance_records.append({
            "Method": name,
            "Annualized Return": ann_return,
            "Annualized Volatility": ann_vol,
            "Sharpe Ratio (Daily)": sharpe,
            "Sharpe Ratio (Ann)": ann_sharpe
        })

    df_perf = pd.DataFrame(performance_records)

    if output_dir:
        perf_csv_path = os.path.join(output_dir, "performance_summary.csv")
        df_perf.to_csv(perf_csv_path, index=False)
        if not silent:
            print(f"パフォーマンス指標を保存しました: {perf_csv_path}")

            print("\n--- 結果サマリー (Top 3 by Sharpe Ann) ---")
            if not df_perf.empty:
                print(df_perf.sort_values("Sharpe Ratio (Ann)", ascending=False).head(3).to_string(index=False))

    return df_perf
