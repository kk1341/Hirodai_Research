import os
import sys

# 自作モジュールのインポート
from data_handler import prepare_data
from backtest_engine import run_backtest
from config import load_config

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
        
    print(f"フォルダ '{input_path}' から {len(stock_files)} 件のCSVファイルを検出しました。")


    # 4. データ準備とクレンジング
    retx_data, retx_cols = prepare_data(
        input_path, stock_files, start_date, end_date, method=INTERPOLATION_METHOD
    )

    # 5. ロール・オーバー・テストの実行
    if retx_data.shape[0] > train_duration and retx_data.shape[1] >= 2:
        run_backtest(
            retx_data, 
            train_duration, 
            retx_cols, 
            output_dir=output_path,
            pca_rank=pca_rank
        )
    else:
        print(
            "\nエラー: 訓練期間または銘柄数が不十分なため、バックテストを実行できません。"
        )
