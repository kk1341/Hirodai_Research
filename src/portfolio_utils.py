import numpy as np

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
