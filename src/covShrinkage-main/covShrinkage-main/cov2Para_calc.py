import pandas as pd
import os

# 1. 銘柄のCSVファイルが保存されているフォルダのパス
folder_path = r'C:\Users\scarl\Documents\Research\data'

# 2. 処理対象のファイル名リスト
stock_files = [
    '80837.csv',
    '80864.csv',
    '80912.csv',
    '80928.csv',
    '80951.csv'
]

# 3. 抽出したい期間
start_date = '1994-10-03'
end_date = '1994-10-07'

# -----------------------------

# --- メインの処理 ---
if __name__ == '__main__':
    if folder_path == r'ここにフォルダのパスを貼り付け':
        print("エラー: プログラム上部の 'folder_path' を書き換えてください。")
    else:
        # 日付インデックスのみを持つ空のDataFrameを作成
        date_index = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq='D'))
        master_df = pd.DataFrame(index=date_index)
        
        print(f"--- {start_date} から {end_date} までの空の表を作成 ---")

        # 各銘柄ファイルをループで処理し、master_dfに結合していく
        for file_name in stock_files:
            try:
                full_path = os.path.join(folder_path, file_name)
                
                # CSVを読み込み、日付形式を整える
                df = pd.read_csv(full_path)
                df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
                df.set_index('date', inplace=True)
                
                # RETX列を数値に変換し、銘柄名でリネーム
                df['RETX'] = pd.to_numeric(df['RETX'], errors='coerce')
                stock_name = os.path.splitext(file_name)[0]
                return_series = df['RETX'].rename(stock_name)
                
                # master_dfに日付を基準に結合(join)する
                master_df = master_df.join(return_series)
                
                print(f"'{file_name}' のデータを結合")

            except Exception as e:
                print(f"エラー: '{file_name}' の処理中に問題が発生: {e}")

        # 結合後に存在する可能性のある欠損値を0で埋める
        master_df.fillna(0, inplace=True)

        print("\n最終的な抽出結果:")
        master_df.index.name = 'date'
        print(master_df)
        master_df.to_csv('retx.csv')