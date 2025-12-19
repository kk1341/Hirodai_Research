import os
import sys

# 自作モジュールのインポート
from data_handler import prepare_data
from backtest_engine import run_backtest

# メイン実行ブロック
if __name__ == "__main__":
    # 0. 補間方法の設定 ('zero', 'linear', 'spline', 'ffill')
    INTERPOLATION_METHOD = "linear"

    # 1. パスの設定
    input_path = "C:/Users/scarl/Documents/Research/data/input/"
    output_path = "C:/Users/scarl/Documents/Research/data/output/"

    # 2. 処理対象のファイル名リスト (自動取得)
    if not os.path.exists(input_path):
         print(f"エラー: 入力フォルダ '{input_path}' が存在しません。")
         sys.exit(1)

    stock_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    
    if not stock_files:
        print(f"エラー: 指定されたフォルダ '{input_path}' にCSVファイルが見つかりませんでした。")
        sys.exit(1)
        
    print(f"フォルダ '{input_path}' から {len(stock_files)} 件のCSVファイルを検出しました。")


    # 3. 抽出したい期間と訓練期間の設定
    start_date = "1995-03-01"
    end_date = "1996-03-01"  # 検証用短縮期間のまま維持
    train_duration = 21  # T_train (訓練期間の長さ)

    # 4. データ準備とクレンジング
    retx_data, retx_cols = prepare_data(
        input_path, stock_files, start_date, end_date, method=INTERPOLATION_METHOD
    )

    # 5. ロール・オーバー・テストの実行
    if retx_data.shape[0] > train_duration and retx_data.shape[1] >= 2:
        run_backtest(retx_data, train_duration, retx_cols, output_dir=output_path)
    else:
        print(
            "\nエラー: 訓練期間または銘柄数が不十分なため、バックテストを実行できません。"
        )
