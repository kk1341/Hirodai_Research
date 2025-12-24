import os
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

# 自作モジュールのインポート
from data_handler import prepare_data
from backtest_engine import run_backtest
from config import load_config

def run_single_simulation(sim_i : int, n_count : int, universe_size : int, universe_retx : np.ndarray, universe_cols : list, train_duration : int, pca_rank : int):
    '''
        単一のシミュレーションを実行する関数
        マルチプロセッシング用

        sim_i : シミュレーションID
        n_count : サンプルサイズ
        universe_size : ユニバースサイズ
        universe_retx : ユニバースのリターンデータ
        universe_cols : ユニバースの銘柄名リスト
        train_duration : 学習期間
        pca_rank : PCAのランク
    '''

    # ランダムサンプリング (非復元抽出)
    selected_indices = np.random.choice(universe_size, n_count, replace=False)
    
    # データのサブセット作成
    sample_retx = universe_retx[:, selected_indices]
    sample_cols = [universe_cols[i] for i in selected_indices]
    
    # バックテスト実行 (静音モード)
    # 個別のファイル出力はしない (output_dir=None)
    df_res = run_backtest(
        sample_retx, 
        train_duration, 
        sample_cols, 
        output_dir=None, 
        pca_rank=pca_rank,
        silent=True
    )

    if not df_res.empty:
        df_res["N"] = n_count
        df_res["SimID"] = sim_i
    
    return df_res

# メイン実行ブロック
if __name__ == "__main__":
    # 設定の読み込み
    try:
        config = load_config()
    except Exception as e:
        print(f"設定の読み込みに失敗しました: {e}")
        sys.exit(1)

    # 0. 設定値の取得
    INTERPOLATION_METHOD = config["backtest"]["interpolation_method"]
    start_date = config["backtest"]["start_date"]
    end_date = config["backtest"]["end_date"]
    train_duration = config["backtest"]["train_duration"]
    pca_rank = config["backtest"]["pca_rank"]

    # 実験設定
    n_values = config["experiment"]["n_values"]
    num_sims = config["experiment"]["num_simulations"]
    random_seed = config["experiment"]["random_seed"]

    np.random.seed(random_seed)

    # 1. パスの設定
    input_path = config["paths"]["input_dir"]
    output_path = config["paths"]["output_dir"]

    # 2. 処理対象のファイル名リスト (自動取得)
    if not os.path.exists(input_path):
         print(f"エラー: 入力フォルダ '{input_path}' が存在しません。")
         sys.exit(1)

    stock_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    
    if not stock_files:
        print(f"エラー: 指定されたフォルダ '{input_path}' にCSVファイルが見つかりませんでした。")
        sys.exit(1)
        
    print(f"フォルダ '{input_path}' から {len(stock_files)} 件のCSVファイルを検出しました (universe)。")

    # 4. データ準備とクレンジング (ユニバース全体を読み込み)
    print("ユニバースデータの読み込み中...")
    universe_retx, universe_cols = prepare_data(
        input_path, stock_files, start_date, end_date, method=INTERPOLATION_METHOD
    )
    
    universe_size = universe_retx.shape[1]
    print(f"データ準備完了: {universe_retx.shape} (T={universe_retx.shape[0]}, N_universe={universe_size})")

    # --- モンテカルロ・シミュレーション実験 ---
    
    experiment_results = []
    
    print(f"\n--- 実験開始: N={n_values}, Sims={num_sims} per N ---")

    for n_count in n_values:
        if n_count > universe_size:
            print(f"Skip: N={n_count} > Universe({universe_size})")
            continue
            
        print(f"\nRunning simulations for N = {n_count}...")

        results_generator = Parallel(n_jobs=-1, return_as="generator")(
            delayed(run_single_simulation)(sim_i, n_count, universe_size, universe_retx, universe_cols, train_duration, pca_rank) 
            for sim_i in range(num_sims)
        )
        
        for sim_i, res in enumerate(tqdm(results_generator, desc=f"Simulations (N={n_count})")):
            if not res.empty:
                experiment_results.append(res)

    # --- 結果の集計と保存 ---
    if experiment_results:
        print("\n実験完了。結果を保存中...")
        all_results_df = pd.concat(experiment_results, ignore_index=True)
        
        # 1. Raw Data (全シミュレーション結果)
        raw_output_path = os.path.join(output_path, "experiment_results_raw.csv")
        all_results_df.to_csv(raw_output_path, index=False)
        print(f"全シミュレーション結果を保存: {raw_output_path}")
        
        # 2. Summary (Nごとの平均/標準偏差)
        # N と Method でグループ化し、Sharpe Ratio (Ann) の平均と標準偏差を計算
        summary_group = all_results_df.groupby(["N", "Method"])["Sharpe Ratio (Ann)"]
        summary_df = summary_group.agg(["mean", "std", "count"]).reset_index()
        summary_df.columns = ["N", "Method", "Mean Sharpe", "Std Sharpe", "Count"]
        
        summary_output_path = os.path.join(output_path, "experiment_results_summary.csv")
        summary_df.to_csv(summary_output_path, index=False)
        print(f"サマリー結果を保存: {summary_output_path}")
        
        print("\n--- Summary (N vs Mean Sharpe) ---")
        # 見やすいようにピボットテーブルで表示
        pivot_summary = summary_df.pivot(index="N", columns="Method", values="Mean Sharpe")
        print(pivot_summary.to_string())
        
    else:
        print("警告: シミュレーション結果が得られませんでした。")
