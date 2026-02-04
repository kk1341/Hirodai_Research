import pandas as pd
import os
import numpy as np

def prepare_data(input_path, stock_files, start_date, end_date, method="spline"):
    """
    複数のCSVファイルを読み込み、指定期間で結合し、指定された方法で欠損値を補間する。
    method: 'zero', 'linear', 'spline', 'ffill'
    """
    # 期待される日付のインデックスを生成 (欠損値補間のベースとなる)
    date_index = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq="D"))
    data_series_list = []
    
    # マーケットファクター (sprtrn) を保持する変数
    market_series = None

    for file_name in stock_files:
        full_path = os.path.join(input_path, file_name)
        try:
            # 必要な列のみ読み込み
            df = pd.read_csv(
                full_path, usecols=["date", "RETX", "sprtrn"], dtype={"date": str, "RETX": str, "sprtrn": str}
            )
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df.set_index("date", inplace=True)
            
            # 重複データの削除
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]


            # RETXを数値に変換し、変換できない値をNaNとする
            df["RETX"] = pd.to_numeric(df["RETX"], errors="coerce")
            df["sprtrn"] = pd.to_numeric(df["sprtrn"], errors="coerce")

            stock_name = os.path.splitext(file_name)[0]
            return_series = df["RETX"].rename(stock_name)
            
            # リストに追加
            data_series_list.append(return_series)
            
            # マーケットファクターの取得 (未取得の場合のみ)
            if market_series is None and not df["sprtrn"].dropna().empty:
                market_series = df["sprtrn"].rename("sprtrn")

        except Exception as e:
            print(f"警告: '{file_name}' の処理中に問題が発生: {e}")

    # 結合
    if data_series_list:
        print("-> データを結合中...")
        master_df = pd.concat(data_series_list, axis=1)
        # 指定された期間の日付インデックスに合わせる (欠損日はNaNになる)
        master_df = master_df.reindex(date_index)
    else:
        print("警告: 有効なデータが見つかりませんでした。")


    # --- 欠損値補間 ---
    print(f"-> 欠損値補間を実行中... (手法: {method})")

    try:
        if method == "zero":
            master_df.fillna(0, inplace=True)

        elif method == "linear":
            master_df.interpolate(method="linear", inplace=True)

        elif method == "spline":
            try:
                master_df.interpolate(method="spline", order=3, inplace=True)
            except Exception as e:
                print(f"警告: スプライン補間に失敗 ({e})。線形補間を試みます。")
                master_df.interpolate(method="linear", inplace=True)

        elif method == "ffill":
            master_df.fillna(method="ffill", inplace=True)

        else:
            print(f"警告: 未知の手法 '{method}' が指定されました。線形補間を適用します。")
            master_df.interpolate(method="linear", inplace=True)

    except Exception as e:
        print(f"警告: 補間処理中にエラーが発生 ({e})。ゼロ埋めを適用します。")
        master_df.fillna(0, inplace=True)

    # 補間で埋めきれなかった欠損値を0で補間
    master_df.fillna(0, inplace=True)

    # 期間外のデータを削除
    master_df = master_df.loc[start_date:end_date]

    # RETXデータ (NumPy配列)
    retx_data = master_df.to_numpy()
    retx_cols = master_df.columns.tolist()

    # Market Factor (sprtrn) の処理
    if market_series is not None:
         # 時間軸を合わせる
         market_series = market_series.reindex(date_index)
         
         # 期間外削除 (master_dfと同じ期間にする)
         market_series = market_series.loc[start_date:end_date]

         # 補間処理 (Market Factorも同様に補間すべき)
         try:
             if method == "zero":
                 market_series.fillna(0, inplace=True)
             else:
                 market_series.interpolate(method="linear", inplace=True)
                 market_series.fillna(0, inplace=True) # 端のNaN対策
         except:
             market_series.fillna(0, inplace=True)

         market_factor = market_series.to_numpy().reshape(-1, 1) # (T, 1)
    else:
         print("警告: sprtrnデータが見つかりませんでした。ゼロで埋めます。")
         market_factor = np.zeros((retx_data.shape[0], 1))

    print(f"データ結合完了。期間: {master_df.index.min()} - {master_df.index.max()}")
    print(f"データ形状: {retx_data.shape} ({len(retx_cols)} 銘柄)")
    print(f"Market Factor形状: {market_factor.shape}")

    return retx_data, retx_cols, market_factor
