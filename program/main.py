import numpy as np
import pandas as pd
import os
from tqdm import tqdm  # プログレスバー用
import scipy

# --- 1. Ledoit-Wolf Shrinkage Estimation Function ---


def linear_shrinkage_identity(X, assume_zero_mean=False):

    """
    Ledoit-Wolf線形収縮推定器（ターゲット行列はスケーリングされた単位行列 I）。

    Parameters
    ----------
    X : ndarray, shape (T, N)
        データ行列: T観測値 (行), N変数 (列).
    assume_zero_mean : bool
        Trueの場合、中心化をスキップ。Falseの場合、共分散推定前にデータを中心化する。

    Returns
    -------
    S : ndarray, shape (N, N)
        標本共分散行列 S.
    Sigma_hat : ndarray, shape (N, N)
        収縮共分散行列 c * mu * I + (1−c) * S.
    """
    T, N = X.shape

    # データの中心化
    if not assume_zero_mean:
        X = X - X.mean(axis=0, keepdims=True)

    # 標本共分散行列 S
    S = (1.0 / T) * (X.T @ X)

    # ターゲット行列の平均 mu_hat = (1/N) * Tr(S)
    mu_hat = (1.0 / N) * np.trace(S)

    # delta^2 の推定: || S − mu I ||^2_F
    S_minus = S - mu_hat * np.eye(N)
    delta2_hat = np.sum(S_minus * S_minus)

    # beta^2 の推定: E|| S − Sigma ||^2_F
    # ベクトル化による高速化 (ループ処理の排除)
    # X_outer: (T, N, N) - 各時点 t における x_t @ x_t.T
    X_outer = X[:, :, np.newaxis] * X[:, np.newaxis, :]
    diff = X_outer - S
    beta2_hat = np.sum(diff**2) / (T**2)

    # Shrinkage Intensity c_hat の計算
    if delta2_hat <= 0:
        c_hat = 0.0
    else:
        c_hat = beta2_hat / delta2_hat

    c_hat = np.clip(c_hat, 0.0, 1.0)  # 0 <= c_hat <= 1 にクリップ

    # 収縮共分散行列の構築
    Sigma_hat = c_hat * (mu_hat * np.eye(N)) + (1.0 - c_hat) * S

    return S, Sigma_hat


# --- 2. Utility Functions for Portfolio ---


def calculate_mvp_weights(cov_matrix):
    """
    推定された共分散行列に基づき、最小分散ポートフォリオ (MVP) の重みを計算する。
    """
    N = cov_matrix.shape[0]
    ones = np.ones((N, 1))

    # 逆行列を用いた重み計算 w = A^-1 * 1 / (1^T * A^-1 * 1)
    # np.linalg.solve を使用して高速化・安定化 (A * w_tmp = 1 を解く)
    try:
        w = np.linalg.solve(cov_matrix, ones)
    except np.linalg.LinAlgError:
        # 特異行列などで逆行列が計算できない場合のエラーを送出
        raise

    # 正規化 (和が1になるように)
    w = w / w.sum()

    return w.flatten()  # 1次元ベクトルとして返す


def calculate_sharpe_ratio(returns_list):
    """
    リターン系列からシャープ・レシオを計算する (リスクフリーレート R_f = 0)。
    """
    R_p = np.array(returns_list)

    if len(R_p) < 2:
        # 観測が少なすぎて標準偏差が計算できない場合
        return 0.0

    r_mean = np.mean(R_p)
    # ddof=1: ベッセル補正 (N-1) を用いた不偏標準偏差 (実測ボラティリティ)
    r_std = np.std(R_p, ddof=1)

    # ボラティリティがゼロの場合、Sharpe Ratioは計算不能
    if r_std == 0:
        return 0.0

    return r_mean / r_std


# --- 3. Data Preparation Function (3次スプライン補間に変更) ---



def prepare_data(input_path, stock_files, start_date, end_date, method="spline"):
    """
    複数のCSVファイルを読み込み、指定期間で結合し、指定された方法で欠損値を補間する。
    method: 'zero', 'linear', 'spline', 'ffill'
    """
    # 期待される日付のインデックスを生成 (欠損値補間のベースとなる)
    date_index = pd.to_datetime(pd.date_range(start=start_date, end=end_date, freq="D"))
    data_series_list = []  # データシリーズを格納するリスト

    for file_name in stock_files:
        full_path = os.path.join(input_path, file_name)
        try:
            # 必要な列のみ読み込み
            df = pd.read_csv(
                full_path, usecols=["date", "RETX"], dtype={"date": str, "RETX": str}
            )
            df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
            df.set_index("date", inplace=True)
            
            # 重複データの削除 (同日に複数のデータがある場合、最後を採用)
            if df.index.duplicated().any():
                df = df[~df.index.duplicated(keep='last')]


            # RETXを数値に変換し、変換できない値をNaNとする
            df["RETX"] = pd.to_numeric(df["RETX"], errors="coerce")
            stock_name = os.path.splitext(file_name)[0]
            return_series = df["RETX"].rename(stock_name)
            
            # リストに追加
            data_series_list.append(return_series)

        except Exception as e:
            print(f"警告: '{file_name}' の処理中に問題が発生: {e}")

    # 一括結合 (パフォーマンス向上)
    if data_series_list:
        print("-> データを結合中...")
        master_df = pd.concat(data_series_list, axis=1)
        # 指定された期間の日付インデックスに合わせる (欠損日はNaNになる)
        master_df = master_df.reindex(date_index)
    else:
        print("警告: 有効なデータが見つかりませんでした。")


    # --- 💡 欠損値補間ロジックの変更 ---
    print(f"-> 欠損値補間を実行中... (手法: {method})")

    try:
        if method == "zero":
            master_df.fillna(0, inplace=True)

        elif method == "linear":
            master_df.interpolate(method="linear", inplace=True)

        elif method == "spline":
            # データ数が極端に少ない場合、スプライン補間は失敗することがあります。
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

    # 補間で埋めきれなかった欠損値（時系列の最初など）を0で補間
    master_df.fillna(0, inplace=True)

    # 期間外のデータがあれば削除
    master_df = master_df.loc[start_date:end_date]

    # RETXデータ (NumPy配列)
    retx_data = master_df.to_numpy()
    retx_cols = master_df.columns.tolist()

    print(f"データ結合完了。期間: {master_df.index.min()} - {master_df.index.max()}")
    print(f"データ形状: {retx_data.shape} ({len(retx_cols)} 銘柄)")

    return retx_data, retx_cols


# --- 4. Main Rolling Window Execution Function ---


def run_backtest(retx_data, train_duration, retx_cols):
    """
    ロール・オーバー・ウィンドウによるバックテストを実行し、シャープ・レシオを比較する。
    """
    T_total, N = retx_data.shape

    # アウトオブサンプル・テストが可能な期間の総回数
    num_test_steps = T_total - train_duration

    # リターン系列を格納するリスト
    r_portfolio_return = []  # 標本共分散行列 S のリターン
    sh_portfolio_return = []  # 収縮共分散行列 Sigma_hat のリターン

    print(f"\n--- バックテスト実行 (T_train={train_duration} / N={N}) ---")
    print(f"総ステップ数: {num_test_steps} 回のテストを実行")

    # i: 訓練期間の終了インデックス、つまりテストするリターンの前日
    for i in tqdm(range(T_total - train_duration), desc="Backtest Progress"):

        # 1. 訓練期間の抽出 (ロール・オーバー)
        train_retx = retx_data[i : i + train_duration, :]

        # 2. 共分散行列の推定
        sample_matrix, shrunken_matrix = linear_shrinkage_identity(train_retx)

        # 3. ポートフォリオ重みの計算
        # S (標本) の重み
        try:
            w_S = calculate_mvp_weights(sample_matrix)
        except np.linalg.LinAlgError:
            # print(
            #     f"警告: ステップ {i} で標本行列の逆行列計算に失敗しました。重みを等配分(1/N)とします。"
            # )
            w_S = np.ones(N) / N

        # Sigma_hat (収縮) の重み
        try:
            w_Sh = calculate_mvp_weights(shrunken_matrix)
        except np.linalg.LinAlgError:
            # print(
            #     f"警告: ステップ {i} で収縮行列の逆行列計算に失敗しました。重みを等配分(1/N)とします。"
            # )
            w_Sh = np.ones(N) / N

        # 4. アウトオブサンプル・リターンの計算
        test_return = retx_data[i + train_duration, :]

        # ポートフォリオリターン = w^T @ r
        r_s = np.dot(test_return, w_S)
        r_sh = np.dot(test_return, w_Sh)

        # 5. リターンの記録
        r_portfolio_return.append(r_s)
        sh_portfolio_return.append(r_sh)

        # デバッグ出力（最初のステップのみ）
        if i == 0:
            print(f"  初回テスト日インデックス: {i + train_duration}")
            print(f"  標本重み (w_S) - 最初の5つ:\n{w_S[:5]}")
            print(f"  縮小重み (w_Sh) - 最初の5つ:\n{w_Sh[:5]}")

    print("\n--- シャープ・レシオの評価 ---")

    # 6. シャープ・レシオの計算
    r_sharpe = calculate_sharpe_ratio(r_portfolio_return)
    sh_sharpe = calculate_sharpe_ratio(sh_portfolio_return)

    # 7. 実測ボラティリティの計算（比較のため）
    r_std = np.std(r_portfolio_return, ddof=1)
    sh_std = np.std(sh_portfolio_return, ddof=1)

    print(
        f"標本ポートフォリオリターン系列 (最初の5つ): {np.array(r_portfolio_return).flatten()[:5]}"
    )
    print(
        f"収縮ポートフォリオリターン系列 (最初の5つ): {np.array(sh_portfolio_return).flatten()[:5]}"
    )

    print(f"\n--- 結果比較 ---")
    print(f"標本共分散行列 (S) を用いたときの実測ボラティリティ: {r_std:.6f}")
    print(f"収縮共分散行列 (Sigma_hat) を用いたときの実測ボラティリティ: {sh_std:.6f}")
    print(f"標準共分散行列 (S) を用いたときのシャープ・レシオ: {r_sharpe:.4f}")
    print(f"縮小共分散行列 (Sigma_hat) を用いたときのシャープ・レシオ: {sh_sharpe:.4f}")


# --- 5. メイン実行ブロック ---

if __name__ == "__main__":
    # 0. 補間方法の設定 ('zero', 'linear', 'spline', 'ffill')
    INTERPOLATION_METHOD = "linear"

    # 1. パスの設定（元のコードの値をそのまま使用）
    # これらのパスは実行環境に合わせて変更が必要です
    input_path = "C:/Users/scarl/Documents/Research/data/input/"

    # 2. 処理対象のファイル名リスト (自動取得)
    stock_files = [f for f in os.listdir(input_path) if f.endswith(".csv")]
    
    if not stock_files:
        print(f"エラー: 指定されたフォルダ '{input_path}' にCSVファイルが見つかりませんでした。")
        # 処理を中断するか、空のリストのまま進むか。ここでは中断が安全。
        import sys
        sys.exit(1)
        
    print(f"フォルダ '{input_path}' から {len(stock_files)} 件のCSVファイルを検出しました。")


    # 3. 抽出したい期間と訓練期間の設定
    start_date = "1995-03-01"
    end_date = "2023-12-19"
    train_duration = 21  # T_train (訓練期間の長さ)

    # 4. データ準備とクレンジング
    retx_data, retx_cols = prepare_data(
        input_path, stock_files, start_date, end_date, method=INTERPOLATION_METHOD
    )

    # 5. ロール・オーバー・テストの実行
    if retx_data.shape[0] > train_duration and retx_data.shape[1] >= 2:
        run_backtest(retx_data, train_duration, retx_cols)
    else:
        print(
            "\nエラー: 訓練期間または銘柄数が不十分なため、バックテストを実行できません。"
        )

