"""
model.py - Phần 3: mô hình bài toán hoạch định danh mục nguồn lực (bản rút gọn).

Giao diện (theo tab "Thông số chung"):
    sample_scenarios(N, seed) -> ndarray [N, 8, 3]
    repair(x)                 -> x  (đã sửa cho khả thi)
    evaluate(x, scenarios)    -> (fitness, mean_profit, mad)
    decode(x)                 -> dict số máy nguyên và sản lượng
    encode(dict)              -> x

Quy mô giữ như bài báo Wang, Wang & Chen (2008): 3 tester (M1-M3), 4 handler
(H1-H4), 3 sản phẩm (SP1 MTS; SP2, SP3 MTO), 8 quý.
Số liệu lấy từ bài báo: vốn ban đầu 20 triệu, lãi 2%/kỳ, hiệu suất mục tiêu 80%,
1800 giờ/kỳ, sigma = 7000, k = 0.5, số máy ban đầu (Bảng 1, kỳ 0).
Các số liệu còn lại (đánh dấu GIẢ ĐỊNH) là giả định vì bài báo không công bố.
"""
import numpy as np

# ----------------------------------------------------------------- kích thước
P, M, A, T = 8, 3, 4, 3                  # quý, tester, handler, sản phẩm
MTS = np.array([True, False, False])     # SP1 make-to-stock, SP2-SP3 make-to-order

# --------------------------------------------------------- số liệu từ bài báo
F0 = 20_000_000.0        # vốn ban đầu
I_RATE = 0.02            # lãi suất mỗi kỳ
UTIL = 0.80              # hiệu suất mục tiêu (tester và handler)
HOURS = 1800.0           # giờ làm việc mỗi kỳ
SIGMA = 7000.0           # độ lệch chuẩn nhu cầu
K_RISK = 0.5             # hệ số đánh đổi lợi nhuận - rủi ro
K0_M = np.array([2, 1, 1])           # số tester ban đầu (Bảng 1a, kỳ 0)
K0_A = np.array([2, 3, 3, 2])        # số handler ban đầu (Bảng 1b, kỳ 0)

# ---------------------------------------------------------- số liệu GIẢ ĐỊNH
MU = np.array([25_000.0, 22_000.0, 15_000.0])   # nhu cầu trung bình mỗi quý
B = np.array([200.0, 230.0, 260.0])             # lợi nhuận đơn vị b_t
J_HOLD = np.array([10.0, 0, 0])                 # chi phí tồn kho/đv/kỳ (chỉ MTS)
L_BACK = np.array([40.0, 0, 0])                 # chi phí thiếu hàng/đv/kỳ (chỉ MTS)

# c_{m,t}: tester nào làm được sản phẩm nào (theo cấu trúc Bảng 1c)
C_MT = np.array([[1, 1, 0],    # M1: SP1, SP2
                 [1, 0, 1],    # M2: SP1, SP3
                 [0, 1, 1]])   # M3: SP2, SP3
# throughput lý thuyết r_{m,t} (đv/giờ)
R_MT = np.array([[3.0, 2.6, 0.0],
                 [3.4, 0.0, 2.4],
                 [0.0, 3.0, 2.8]])
# cấu hình đơn giản hóa (3): mỗi cặp (tester, sản phẩm) đi với 1 handler cố định
H_OF = np.array([[0, 1, -1],   # M1-SP1 -> H1, M1-SP2 -> H2
                 [1, -1, 2],   # M2-SP1 -> H2, M2-SP3 -> H3
                 [-1, 3, 2]])  # M3-SP2 -> H4, M3-SP3 -> H3
R_H = np.array([4.0, 3.6, 3.2, 3.8])            # throughput handler (đv/giờ)

E_M = np.array([1_500_000.0, 1_700_000.0, 1_900_000.0])  # giá mua tester
D_M = 0.4 * E_M                                          # giá thanh lý tester
U_M = np.array([140_000.0, 160_000.0, 180_000.0])        # chi phí chuyển/thuê tester/quý
E_A = np.array([150_000.0, 160_000.0, 170_000.0, 155_000.0])  # giá mua handler
D_A = 0.4 * E_A
U_A = np.array([14_000.0, 15_000.0, 16_000.0, 14_500.0])

# ---------------------------------------------------------------- giải mã gen
K_EXTRA_M, K_EXTRA_A = 6, 8      # mua thêm tối đa so với ban đầu
X_MAX_M, X_MAX_A = 6, 8          # chuyển/thuê tối đa mỗi quý
Q_MAX = 1.5 * MU                 # sản lượng tối đa 1 tester làm cho 1 sản phẩm/quý

N_GENES = M + A + P * M + P * A + P * M * T      # 3 + 4 + 24 + 32 + 72 = 135
SL_KM = slice(0, 3)
SL_KA = slice(3, 7)
SL_XM = slice(7, 31)
SL_XA = slice(31, 63)
SL_Q = slice(63, 135)

CAP_M = R_MT * HOURS * UTIL      # sản lượng 1 tester/quý theo sản phẩm [M, T]
CAP_H = R_H * HOURS * UTIL       # sản lượng 1 handler/quý [A]


def decode(x):
    """Gen [0,1] -> dict số nguyên và sản lượng."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return {
        "K_m": K0_M + np.rint(x[SL_KM] * K_EXTRA_M).astype(int),
        "K_a": K0_A + np.rint(x[SL_KA] * K_EXTRA_A).astype(int),
        "X_m": np.rint(x[SL_XM].reshape(P, M) * X_MAX_M).astype(int),
        "X_a": np.rint(x[SL_XA].reshape(P, A) * X_MAX_A).astype(int),
        "Q": x[SL_Q].reshape(P, M, T) * Q_MAX[None, None, :] * C_MT[None, :, :],
    }


def encode(d):
    """dict -> gen [0,1] (ngược của decode)."""
    x = np.empty(N_GENES)
    x[SL_KM] = (np.asarray(d["K_m"]) - K0_M) / K_EXTRA_M
    x[SL_KA] = (np.asarray(d["K_a"]) - K0_A) / K_EXTRA_A
    x[SL_XM] = (np.asarray(d["X_m"]) / X_MAX_M).ravel()
    x[SL_XA] = (np.asarray(d["X_a"]) / X_MAX_A).ravel()
    x[SL_Q] = (np.asarray(d["Q"], dtype=float) / Q_MAX[None, None, :]).ravel()
    return np.clip(x, 0.0, 1.0)


def _loads(d):
    """Số máy cần (tính bằng máy tương đương) cho tester [P,M] và handler [P,A]."""
    Q = d["Q"]
    need_m = np.where(C_MT > 0, Q / np.where(CAP_M > 0, CAP_M, 1.0), 0.0).sum(axis=2)
    need_a = np.zeros((P, A))
    for m in range(M):
        for t in range(T):
            a = H_OF[m, t]
            if a >= 0:
                need_a[:, a] += Q[:, m, t] / CAP_H[a]
    return need_m, need_a


def repair(x):
    """Sửa nghiệm cho thỏa (2) và (4): nếu năng lực máy không đủ thì co sản lượng.
    Gen của cặp (tester, sản phẩm) không tương thích được đặt về 0."""
    d = decode(x)
    Q = d["Q"]
    # (4) handler: co sản lượng các cặp (m,t) dùng handler bị quá tải
    _, need_a = _loads(d)
    have_a = d["K_a"][None, :] + d["X_a"]
    ratio_a = np.minimum(1.0, have_a / np.maximum(need_a, 1e-9))
    for m in range(M):
        for t in range(T):
            a = H_OF[m, t]
            if a >= 0:
                Q[:, m, t] *= ratio_a[:, a]
    # (2) tester
    need_m, _ = _loads(d)
    have_m = d["K_m"][None, :] + d["X_m"]
    ratio_m = np.minimum(1.0, have_m / np.maximum(need_m, 1e-9))
    Q *= ratio_m[:, :, None]
    d["Q"] = Q
    return encode(d)


def sample_scenarios(N=30, seed=0):
    """N kịch bản nhu cầu chuẩn N(MU, SIGMA), cắt tại 0. Trả về [N, 8, 3]."""
    rng = np.random.default_rng(seed)
    o = rng.normal(MU[None, None, :], SIGMA, size=(N, P, T))
    return np.maximum(o, 0.0)


def scenario_profits(x, scenarios):
    """Lợi nhuận h của từng kịch bản theo (5), (6), (7), (8), (9)."""
    d = decode(x)
    Q = d["Q"]                                   # [P, M, T]
    prod = (C_MT[None] * Q).sum(axis=1)          # [P, T] sản lượng theo sản phẩm
    o = scenarios                                # [N, P, T]
    N = o.shape[0]

    # chi phí chuyển/thuê mỗi quý (không phụ thuộc kịch bản)
    out_cost = (d["X_m"] * U_M[None]).sum(axis=1) + (d["X_a"] * U_A[None]).sum(axis=1)

    F = np.full(N, F0)
    S = np.zeros(N)                              # tồn kho ròng SP1 (MTS)
    for p in range(P):
        # (5) cân bằng tồn kho MTS
        S = S + prod[p, 0] - o[:, p, 0]
        # (7) chi phí tồn kho / thiếu hàng
        V = np.where(S >= 0, J_HOLD[0] * S, L_BACK[0] * (-S))
        # (6) MTO: chỉ bán được tối đa bằng nhu cầu, phần làm dư không có doanh thu
        mto_sold = np.minimum(prod[p, 1:][None, :], o[:, p, 1:])
        income = B[0] * o[:, p, 0] + (B[1:][None, :] * mto_sold).sum(axis=1)
        # (8) cân bằng vốn
        F = F * (1 + I_RATE) - out_cost[p] - V + income
    # (9) lợi nhuận = giá trị hiện tại của vốn cuối kỳ - chi phí đầu tư ròng
    invest = ((E_M - D_M) * (d["K_m"] - K0_M)).sum() + ((E_A - D_A) * (d["K_a"] - K0_A)).sum()
    return F / (1 + I_RATE) ** P - invest


def evaluate(x, scenarios):
    """Fitness = (1-k)*mean(h) - k*MAD(h). Trả về (fitness, mean_profit, mad)."""
    h = scenario_profits(x, scenarios)
    mean = float(h.mean())
    mad = float(np.abs(h - mean).mean())
    return (1 - K_RISK) * mean - K_RISK * mad, mean, mad
