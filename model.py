"""
model.py - MÔ HÌNH BÀI TOÁN: cho một phương án mua/thuê máy, tính ra lợi nhuận.

File này KHÔNG có thuật toán tìm kiếm. Nó chỉ trả lời câu hỏi:
    "Nếu mua/thuê máy và phân bổ sản lượng như thế này thì lãi bao nhiêu, rủi ro thế nào?"

Quan hệ với hai file còn lại:
    config.py (số liệu)  ->  model.py (file này)  ->  solver.py (thuật toán dùng model để chấm điểm)
Mọi con số đọc từ config.py dưới dạng cfg.TÊN, tại thời điểm gọi hàm. Nhờ vậy khi demo đổi
cfg.K_RISK, cfg.SIGMA hay cfg.MU thì model dùng ngay giá trị mới.

Thứ tự đọc nên theo:
    1. Hằng số dẫn xuất (cách một phương án được ghi thành 135 gen)
    2. decode / encode  : đổi qua lại giữa "135 gen" và "phương án đọc được" (số máy, sản lượng)
    3. _loads / repair  : kiểm tra máy có đủ công suất không, nếu không thì giảm sản lượng
    4. sample_scenarios : sinh các kịch bản nhu cầu
    5. scenario_profits : lợi nhuận của phương án trong từng kịch bản
    6. evaluate         : điểm Z cuối cùng (hàm duy nhất mà thuật toán gọi)

Quy mô như bài báo Wang, Wang & Chen (2008): 3 tester (M1-M3), 4 handler (H1-H4),
3 sản phẩm (SP1 làm để dự trữ, SP2 và SP3 làm theo đơn), 8 quý.
"""
import numpy as np

import config as cfg

# ---------------------------------------------------------------- hằng số dẫn xuất
# Một phương án (cá thể) được ghi thành 1 vector 135 số trong [0, 1] gọi là "gen":
#   3 gen  : số tester mua thêm của từng loại
#   4 gen  : số handler mua thêm của từng loại
#   24 gen : số tester thuê mỗi quý (8 quý x 3 loại)
#   32 gen : số handler thuê mỗi quý (8 quý x 4 loại)
#   72 gen : sản lượng mỗi quý, mỗi tester, mỗi sản phẩm (8 x 3 x 3)
N_GENES = cfg.M + cfg.A + cfg.P * cfg.M + cfg.P * cfg.A + cfg.P * cfg.M * cfg.T   # 3 + 4 + 24 + 32 + 72 = 135
SL_KM = slice(0, 3)       # vị trí các gen "tester mua thêm"
SL_KA = slice(3, 7)       # vị trí các gen "handler mua thêm"
SL_XM = slice(7, 31)      # vị trí các gen "tester thuê theo quý"
SL_XA = slice(31, 63)     # vị trí các gen "handler thuê theo quý"
SL_Q = slice(63, 135)     # vị trí các gen "sản lượng"

CAP_M = cfg.R_MT * cfg.HOURS * cfg.UTIL      # sản lượng 1 tester làm được trong 1 quý, theo sản phẩm [M, T] (đơn vị/quý)
CAP_H = cfg.R_H * cfg.HOURS * cfg.UTIL       # sản lượng 1 handler làm được trong 1 quý [A] (đơn vị/quý)


def q_max():
    """Làm gì: tính sản lượng tối đa 1 tester được làm cho mỗi sản phẩm trong 1 quý.
    Nhận vào: không có (đọc cfg.Q_MAX_FACTOR và cfg.MU tại thời điểm gọi).
    Trả ra: mảng 3 số [SP1, SP2, SP3], đơn vị sản phẩm/quý.
    Ví dụ: nhu cầu trung bình SP1 là 25 000 thì 1 tester làm tối đa 1.5 x 25 000 = 37 500 SP1 mỗi quý.
    Vì phải theo cfg.MU hiện tại (demo có thể đổi MU) nên đây là hàm chứ không phải hằng số."""
    return cfg.Q_MAX_FACTOR * cfg.MU


def decode(x):
    """Làm gì: đổi 135 gen trong [0, 1] thành phương án đọc được (số máy nguyên và sản lượng).
    Nhận vào: x, vector 135 số (số ngoài [0, 1] sẽ bị cắt về [0, 1]).
    Trả ra: dict gồm
        K_m [3]       số tester sở hữu mỗi loại,   K_a [4]  số handler sở hữu mỗi loại,
        X_m [8, 3]    số tester thuê mỗi quý,       X_a [8, 4]  số handler thuê mỗi quý,
        Q [8, 3, 3]   sản lượng theo (quý, tester, sản phẩm); ô tester không làm được sản phẩm thì bằng 0.
    Ví dụ: gen 0.5 ở ô "tester M1 mua thêm" với tối đa mua thêm 6 máy -> mua thêm 3 máy, M1 có 2 + 3 = 5 máy."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return {
        "K_m": cfg.K0_M + np.rint(x[SL_KM] * cfg.K_EXTRA_M).astype(int),
        "K_a": cfg.K0_A + np.rint(x[SL_KA] * cfg.K_EXTRA_A).astype(int),
        "X_m": np.rint(x[SL_XM].reshape(cfg.P, cfg.M) * cfg.X_MAX_M).astype(int),
        "X_a": np.rint(x[SL_XA].reshape(cfg.P, cfg.A) * cfg.X_MAX_A).astype(int),
        "Q": x[SL_Q].reshape(cfg.P, cfg.M, cfg.T) * q_max()[None, None, :] * cfg.C_MT[None, :, :],
    }


def encode(plan):
    """Làm gì: đổi phương án đọc được về lại 135 gen (ngược của decode).
    Nhận vào: plan, dict có đủ 5 khóa K_m, K_a, X_m, X_a, Q như decode trả ra.
    Trả ra: vector 135 số trong [0, 1].
    Ví dụ: M1 có 5 máy (ban đầu 2, tối đa thêm 6) -> gen = (5 - 2) / 6 = 0.5."""
    x = np.empty(N_GENES)
    x[SL_KM] = (np.asarray(plan["K_m"]) - cfg.K0_M) / cfg.K_EXTRA_M
    x[SL_KA] = (np.asarray(plan["K_a"]) - cfg.K0_A) / cfg.K_EXTRA_A
    x[SL_XM] = (np.asarray(plan["X_m"]) / cfg.X_MAX_M).ravel()
    x[SL_XA] = (np.asarray(plan["X_a"]) / cfg.X_MAX_A).ravel()
    x[SL_Q] = (np.asarray(plan["Q"], dtype=float) / q_max()[None, None, :]).ravel()
    return np.clip(x, 0.0, 1.0)


def _loads(plan):
    """Làm gì: tính mỗi quý cần bao nhiêu máy để làm hết sản lượng đã định.
    Nhận vào: plan, dict đã decode.
    Trả ra: (need_m [8, 3], need_a [8, 4]) = số tester và số handler cần, tính bằng "máy tương đương"
        (có thể lẻ, ví dụ 2.4 máy nghĩa là cần hơn 2 máy làm việc ở hiệu suất mục tiêu).
    Ví dụ: sản lượng quý 1 của M1 cần 2.4 máy mà chỉ có 2 máy -> không đủ công suất, repair sẽ giảm sản lượng."""
    Q = plan["Q"]
    need_m = np.where(cfg.C_MT > 0, Q / np.where(CAP_M > 0, CAP_M, 1.0), 0.0).sum(axis=2)
    need_a = np.zeros((cfg.P, cfg.A))
    for m in range(cfg.M):
        for t in range(cfg.T):
            a = cfg.H_OF[m, t]
            if a >= 0:
                need_a[:, a] += Q[:, m, t] / CAP_H[a]
    return need_m, need_a


def repair(x):
    """Làm gì: sửa phương án cho khả thi - nếu máy không đủ công suất thì giảm sản lượng cho vừa.
    Nhận vào: x, vector 135 gen có thể vi phạm công suất.
    Trả ra: vector 135 gen đã sửa (số máy giữ nguyên, chỉ co các gen sản lượng).
    Ví dụ: cần 2.4 handler H1 mà chỉ có 2 -> mọi sản lượng đi qua H1 bị nhân 2 / 2.4 = 0.83.
    Gen của cặp (tester, sản phẩm) không tương thích được đặt về 0."""
    plan = decode(x)
    Q = plan["Q"]
    # (4) handler: co sản lượng các cặp (m, t) dùng handler bị quá tải
    _, need_a = _loads(plan)
    have_a = plan["K_a"][None, :] + plan["X_a"]
    ratio_a = np.minimum(1.0, have_a / np.maximum(need_a, 1e-9))
    for m in range(cfg.M):
        for t in range(cfg.T):
            a = cfg.H_OF[m, t]
            if a >= 0:
                Q[:, m, t] *= ratio_a[:, a]
    # (2) tester: co sản lượng của tester bị quá tải
    need_m, _ = _loads(plan)
    have_m = plan["K_m"][None, :] + plan["X_m"]
    ratio_m = np.minimum(1.0, have_m / np.maximum(need_m, 1e-9))
    Q *= ratio_m[:, :, None]
    plan["Q"] = Q
    return encode(plan)


def sample_scenarios(N=30, seed=0, mu=None, sigma=None):
    """Làm gì: sinh N kịch bản nhu cầu ngẫu nhiên (phân phối chuẩn quanh nhu cầu trung bình, cắt tại 0).
    Nhận vào: N số kịch bản; seed hạt giống (cùng seed cho cùng kịch bản);
        mu, sigma tùy chọn - nếu bỏ trống thì đọc cfg.MU, cfg.SIGMA (truyền vào khi chạy nền
        để không bị ảnh hưởng nếu demo đổi cfg giữa chừng).
    Trả ra: mảng [N, 8, 3] = nhu cầu của từng kịch bản, từng quý, từng sản phẩm.
    Ví dụ: kịch bản số 5, quý 2, SP1 có thể là 31 200 đơn vị (trung bình 25 000, lệch chuẩn 7 000)."""
    mu = cfg.MU if mu is None else np.asarray(mu, dtype=float)
    sig = cfg.SIGMA if sigma is None else float(sigma)
    rng = np.random.default_rng(seed)
    demand = rng.normal(mu[None, None, :], sig, size=(N, cfg.P, cfg.T))
    return np.maximum(demand, 0.0)


def scenario_profits(x, scenarios):
    """Làm gì: tính lợi nhuận của phương án trong TỪNG kịch bản nhu cầu (công thức (5) đến (9) của mô hình).
    Nhận vào: x vector 135 gen; scenarios mảng [N, 8, 3] từ sample_scenarios.
    Trả ra: mảng N số, lợi nhuận (đồng) ở mỗi kịch bản.
    Ví dụ: kịch bản nhu cầu cao thì bán được nhiều, lãi cao; kịch bản nhu cầu thấp thì tồn kho,
    lãi thấp - hàm này cho biết lãi ở từng kịch bản đó."""
    plan = decode(x)
    Q = plan["Q"]                                   # [P, M, T]
    prod = (cfg.C_MT[None] * Q).sum(axis=1)         # [P, T] sản lượng theo sản phẩm
    demand = scenarios                              # [N, P, T]
    N = demand.shape[0]

    # chi phí chuyển/thuê mỗi quý (không phụ thuộc kịch bản)
    out_cost = (plan["X_m"] * cfg.U_M[None]).sum(axis=1) + (plan["X_a"] * cfg.U_A[None]).sum(axis=1)

    F = np.full(N, cfg.F0)                          # vốn tiền mặt của từng kịch bản
    S = np.zeros(N)                                 # tồn kho ròng SP1 (MTS)
    for p in range(cfg.P):
        # (5) cân bằng tồn kho MTS
        S = S + prod[p, 0] - demand[:, p, 0]
        # (7) chi phí tồn kho / thiếu hàng
        V = np.where(S >= 0, cfg.J_HOLD[0] * S, cfg.L_BACK[0] * (-S))
        # (6) MTO: chỉ bán được tối đa bằng nhu cầu, phần làm dư không có doanh thu
        mto_sold = np.minimum(prod[p, 1:][None, :], demand[:, p, 1:])
        income = cfg.B[0] * demand[:, p, 0] + (cfg.B[1:][None, :] * mto_sold).sum(axis=1)
        # (8) cân bằng vốn
        F = F * (1 + cfg.I_RATE) - out_cost[p] - V + income
    # (9) lợi nhuận = giá trị hiện tại của vốn cuối kỳ - chi phí đầu tư ròng
    invest = (((cfg.E_M - cfg.D_M) * (plan["K_m"] - cfg.K0_M)).sum()
              + ((cfg.E_A - cfg.D_A) * (plan["K_a"] - cfg.K0_A)).sum())
    return F / (1 + cfg.I_RATE) ** cfg.P - invest


def evaluate(x, scenarios):
    """Làm gì: chấm điểm Z của phương án = (1 - k) x lợi nhuận trung bình - k x MAD (MAD = độ dao động của lợi nhuận).
    Nhận vào: x vector 135 gen; scenarios mảng kịch bản nhu cầu.
    Trả ra: bộ ba (Z, lợi nhuận trung bình, MAD), đơn vị đồng. Z càng lớn càng tốt.
    Ví dụ: lãi trung bình 50 triệu nhưng dao động 10 triệu, k = 0.5 -> Z = 0.5 x 50 - 0.5 x 10 = 45 triệu."""
    profits = scenario_profits(x, scenarios)
    mean = float(profits.mean())
    mad = float(np.abs(profits - mean).mean())
    return (1 - cfg.K_RISK) * mean - cfg.K_RISK * mad, mean, mad
