"""
solver.py - TẤT CẢ THUẬT TOÁN: SPGA gốc, GA-TS (GA + Tabu Search), GA-SA (GA + Simulated Annealing),
cùng bài toán MILP dùng làm "đáp án chuẩn" để so sánh.

QUAN HỆ VỚI HAI FILE CÒN LẠI
    config.py (số liệu)  ->  model.py (tính điểm một phương án)  ->  solver.py (file này: đi tìm phương án tốt)
solver.py chỉ dùng model.py để chấm điểm, và đọc tham số từ config.py (cfg.TÊN).

CÁCH ĐỌC FILE NÀY (từ trên xuống dưới)
    1. Bộ đếm đánh giá         : đếm số lần tính Z, quyết định lúc nào dừng
    2. Toán tử GA              : chọn cha mẹ, lai ghép, đột biến
    3. Bước lân cận            : các thay đổi nhỏ trên một phương án
    4. Tabu Search             : cách đi thứ nhất để mài một phương án (BƯỚC 1 đến 9)
    5. Simulated Annealing     : cách đi thứ hai (BƯỚC 1 đến 6), kèm ước lượng nhiệt độ T0 (BƯỚC 1 đến 4)
    6. Hill climbing           : cách đi thứ ba, CHỈ giữ để so sánh phụ (ablation), không thuộc 3 thuật toán chính
    7. Khung GA chung          : vòng lặp chính (BƯỚC 1 đến 9) dùng chung cho cả 3 thuật toán
    8. CÁC HÀM SOLVE (cuối file): solve_spga, solve_ga_ts, solve_ga_sa, solve_milp
Số "BƯỚC n" trong chú thích TRÙNG với số trong MA_GIA.md và flowcharts.html.

BA THUẬT TOÁN KHÁC NHAU Ở ĐÂU
    Cả 3 dùng CHUNG một khung GA (giống SPGA của Wang, Wang & Chen 2008): quần thể 30, lai ghép 0.9,
    tournament 2, giữ 1 cá thể tốt nhất, lai ghép hybrid, đột biến 0.05, ngân sách 6000 lần tính Z
    (tính cả các lần tính trong tìm kiếm cục bộ).
        SPGA gốc : chỉ có khung GA.
        GA-TS    : khung GA + mài bằng Tabu Search.
        GA-SA    : khung GA + mài bằng Simulated Annealing.
    GA-TS và GA-SA có CÙNG điểm kích hoạt (cứ 10 thế hệ, mài 3 cá thể tốt nhất, mỗi cá thể tối đa 20 lần tính Z),
    CÙNG tập 174 bước lân cận và cùng ghi ngược nghiệm tốt nhất vào quần thể. Chúng chỉ khác nhau ở
    CÁCH ĐI trong lân cận, nên so sánh là công bằng.

THUẬT NGỮ
    cá thể (nghiệm) x : một phương án mua/thuê máy và phân bổ sản lượng, ghi thành 135 số gọi là "gen" (mỗi gen trong [0, 1]).
    gen               : một con số trong cá thể; nhóm gen cho biết số máy mua, số máy thuê mỗi quý, sản lượng mỗi quý.
    Z (fitness)       : điểm của một phương án = (1 - k) x lợi nhuận trung bình - k x MAD, tính trên 30 kịch bản nhu cầu.
                        Z càng lớn càng tốt (đơn vị đồng, thường đọc theo triệu).
    evaluate          : một lần tính Z cho một cá thể. Đây là "đồng tiền" để đo công bằng: mọi thuật toán
                        có cùng số lần evaluate (ngân sách 6000) thì mới so sánh được.
    bước lân cận      : một thay đổi nhỏ trên phương án, ví dụ mua thêm 1 tester, bớt 1 handler thuê ở quý 3,
                        chuyển 10% sản lượng từ tester này sang tester kia. Có 174 bước như vậy.
    tabu              : danh sách các bước đang bị cấm (bộ nhớ ngắn hạn của Tabu Search).
    nhiệt độ T        : trong Simulated Annealing, T càng cao thì càng dễ nhận một bước làm Z tệ đi; T giảm dần theo thời gian.
"""
import math
import sys
import time

import numpy as np

import config as cfg
import model as mdl


# =====================================================================================
# 1. BỘ ĐẾM ĐÁNH GIÁ
# =====================================================================================
class EvaluationCounter:
    """Làm gì: bọc hàm evaluate để đếm số lần gọi, đo thời gian thực và ghi lịch sử Z tốt nhất.
    Nhận vào (khi tạo): evaluate (hàm x -> (Z, lợi nhuận TB, MAD)), budget (số lần được gọi),
        time_limit_sec (giây, tùy chọn), stopping_criterion ("evaluations" hoặc "time").
    Trả ra: một đối tượng dùng như hàm: counted_evaluate(x) trả về Z của x.

    Mọi lần tính Z (kể cả trong Tabu, SA và lúc ước lượng T0) đều đi qua đây, nên điều kiện dừng
    nằm ở MỘT chỗ duy nhất -> so sánh công bằng giữa các thuật toán.
    Hai chế độ dừng:
      "evaluations": dừng khi đủ `budget` lần evaluate (mặc định).
      "time"       : dừng khi thời gian thực trôi qua >= time_limit_sec.
    Bộ đếm cũng tự nhớ phương án tốt nhất từng gặp (best_x, best_f) và hai loại lịch sử hội tụ:
      history      : [(số lần evaluate, best_f)]                     -> vẽ theo trục số lần đánh giá
      time_history : [(giây, số lần evaluate, best_f, mean, mad)]     -> vẽ theo trục thời gian
    Ví dụ: ngân sách 6000, đã gọi 5990 lần -> còn 10 lần; sau lần thứ 6000, should_stop() trả True."""

    def __init__(self, evaluate, budget, time_limit_sec=None, stopping_criterion="evaluations"):
        self.evaluate = evaluate     # hàm gốc: x -> (Z, lợi nhuận TB, MAD)
        self.budget = budget         # tổng số lần được phép gọi (chỉ áp cho mode 'evaluations')
        self.time_limit_sec = time_limit_sec
        self.stopping_criterion = stopping_criterion   # "evaluations" | "time"
        self.start_time = time.perf_counter()
        self.n_calls = 0             # số lần đã gọi
        self.best_f = -np.inf        # Z tốt nhất từ đầu lần chạy
        self.best_x = None           # phương án tương ứng
        self.history = []            # [(n_calls, best_f)] để vẽ đường hội tụ theo số lần đánh giá
        self.time_history = []       # [(giây, n_calls, best_f, mean, mad)] theo thời gian thực

    def elapsed(self):
        """Làm gì: cho biết thời gian thực đã trôi qua. Trả ra: số giây kể từ lúc tạo bộ đếm."""
        return time.perf_counter() - self.start_time

    def time_left(self):
        """Làm gì: cho biết thời gian còn lại. Trả ra: số giây còn lại, hoặc None nếu không giới hạn thời gian."""
        if self.time_limit_sec is None:
            return None
        return self.time_limit_sec - self.elapsed()

    def left(self):
        """Làm gì: cho biết ngân sách còn lại (mode 'evaluations'). Trả ra: số lần evaluate còn lại."""
        return self.budget - self.n_calls

    def available(self):
        """Làm gì: cho biết còn được evaluate thêm tối đa bao nhiêu lần trước khi dừng.
        Trả ra: số lần còn lại; vô hạn nếu đang dừng theo thời gian."""
        if self.stopping_criterion == "time" and self.time_limit_sec is not None:
            return float("inf")
        return self.budget - self.n_calls

    def should_stop(self):
        """Làm gì: kiểm tra đã đến lúc dừng chưa. Trả ra: True khi hết ngân sách đánh giá HOẶC hết giờ."""
        if self.stopping_criterion == "time" and self.time_limit_sec is not None:
            return self.elapsed() >= self.time_limit_sec
        return self.n_calls >= self.budget

    def __call__(self, x):
        """Làm gì: tính Z của x (tốn đúng 1 lần evaluate), cập nhật kỷ lục và lịch sử.
        Nhận vào: x cá thể 135 gen. Trả ra: Z của x."""
        z, mean, mad = self.evaluate(x)
        self.n_calls += 1
        if z > self.best_f:          # cập nhật kỷ lục
            self.best_f, self.best_x = z, x.copy()
            self.time_history.append((self.elapsed(), self.n_calls, z, mean, mad))
        self.history.append((self.n_calls, self.best_f))
        return z


def time_history_to_df(time_history):
    """Làm gì: chuyển lịch sử hội tụ theo thời gian thực thành bảng pandas.DataFrame.
    Nhận vào: time_history, danh sách bộ (giây, số lần evaluate, Z tốt nhất, lợi nhuận TB, MAD).
    Trả ra: DataFrame có cột elapsed_s, evaluations, best_z, mean_profit, mad.
    pandas chỉ được import trong hàm này để solver.py vẫn chạy được khi chưa cài pandas."""
    import pandas as pd
    return pd.DataFrame(
        [list(row) for row in time_history],
        columns=["elapsed_s", "evaluations", "best_z", "mean_profit", "mad"])


# =====================================================================================
# 2. TOÁN TỬ GA (chọn cha mẹ, lai ghép, đột biến)
# =====================================================================================
def tournament(fitness, rng, k=cfg.TOURNAMENT):
    """Làm gì: chọn một cha/mẹ bằng "giải đấu": bốc ngẫu nhiên k cá thể, lấy cá thể có Z cao nhất.
    Nhận vào: fitness mảng Z của cả quần thể; rng bộ sinh số ngẫu nhiên; k số cá thể dự thi.
    Trả ra: chỉ số (vị trí) của cá thể thắng.
    Ví dụ: k = 2, bốc được cá thể có Z 47 triệu và 45 triệu -> chọn cá thể 47 triệu.
    k = 2 cho áp lực chọn lọc vừa phải: cá thể kém vẫn có cơ hội làm cha mẹ (giữ đa dạng)."""
    idx = rng.integers(0, len(fitness), size=k)
    return idx[np.argmax(fitness[idx])]


def crossover(parent_a, parent_b, rng):
    """Làm gì: lai ghép hybrid - mỗi lần chọn ngẫu nhiên 1 trong 4 kiểu để tạo 2 con từ 2 cha mẹ.
    Nhận vào: parent_a, parent_b hai cá thể cha mẹ (135 gen); rng bộ sinh số ngẫu nhiên.
    Trả ra: (con 1, con 2).
    Bốn kiểu:
        0: 1 điểm     - cắt tại vị trí c, ghép đầu của cha với đuôi của mẹ
        1: 2 điểm     - đổi đoạn gen [a, b) giữa cha và mẹ
        2: uniform    - từng gen tung đồng xu xem lấy của cha hay của mẹ
        3: arithmetic - con = lam x cha + (1 - lam) x mẹ (trộn giá trị, hợp với gen liên tục)
    Trộn nhiều kiểu giúp vừa giữ được khối gen tốt (1, 2 điểm) vừa thử giá trị mới (arithmetic)."""
    n = len(parent_a)
    kind = rng.integers(4)
    if kind == 0:                                   # 1 điểm
        c = rng.integers(1, n)
        return np.r_[parent_a[:c], parent_b[c:]], np.r_[parent_b[:c], parent_a[c:]]
    if kind == 1:                                   # 2 điểm
        a, b = sorted(rng.choice(np.arange(1, n), 2, replace=False))
        child1, child2 = parent_a.copy(), parent_b.copy()
        child1[a:b], child2[a:b] = parent_b[a:b], parent_a[a:b]
        return child1, child2
    if kind == 2:                                   # uniform
        mask = rng.random(n) < 0.5
        return np.where(mask, parent_a, parent_b), np.where(mask, parent_b, parent_a)
    lam = rng.random()                              # arithmetic
    return lam * parent_a + (1 - lam) * parent_b, lam * parent_b + (1 - lam) * parent_a


def mutate(x, rng, pm=cfg.PM):
    """Làm gì: đột biến uniform - mỗi gen có xác suất pm bị thay bằng một số ngẫu nhiên mới trong [0, 1].
    Nhận vào: x cá thể 135 gen; rng bộ sinh số ngẫu nhiên; pm xác suất đột biến mỗi gen.
    Trả ra: cá thể mới (x gốc không bị sửa).
    Ví dụ: 135 gen với pm = 0.05 -> trung bình khoảng 7 gen bị đổi mỗi cá thể con."""
    mask = rng.random(len(x)) < pm
    x = x.copy()
    x[mask] = rng.random(mask.sum())
    return x


# =====================================================================================
# 3. BƯỚC LÂN CẬN
# Một "bước lân cận" là một thay đổi nhỏ, có ý nghĩa nghiệp vụ, trên phương án đã giải mã.
# Mỗi bước là bộ (kind, idx, step):
#   ("K_m", m, ±1)              : mua thêm / bớt 1 tester loại m           ->  3 x 2  =  6 bước
#   ("K_a", a, ±1)              : mua thêm / bớt 1 handler loại a          ->  4 x 2  =  8 bước
#   ("X_m", (p, m), ±1)         : thuê thêm / bớt 1 tester m ở quý p        ->  8x3x2  = 48 bước
#   ("X_a", (p, a), ±1)         : thuê thêm / bớt 1 handler a ở quý p       ->  8x4x2  = 64 bước
#   ("shift", (p,t,src,dst), .1): chuyển 10% sản lượng SP t quý p từ tester src sang dst
#                                  (chỉ giữa các tester làm được SP t)       ->        48 bước
#                                                                              Tổng = 174 bước
# =====================================================================================
def _neighbor_moves():
    """Làm gì: liệt kê toàn bộ 174 bước lân cận theo 5 kiểu ở bảng trên.
    Nhận vào: không có. Trả ra: danh sách các bước (kind, idx, step)."""
    moves = []
    for m in range(cfg.M):                           # (1) K_m ± 1
        moves += [("K_m", m, +1), ("K_m", m, -1)]
    for a in range(cfg.A):                           # (1) K_a ± 1
        moves += [("K_a", a, +1), ("K_a", a, -1)]
    for p in range(cfg.P):                           # (2) X ± 1 trong 1 quý
        for m in range(cfg.M):
            moves += [("X_m", (p, m), +1), ("X_m", (p, m), -1)]
        for a in range(cfg.A):
            moves += [("X_a", (p, a), +1), ("X_a", (p, a), -1)]
    for p in range(cfg.P):                           # (3) chuyển 10% sản lượng
        for t in range(cfg.T):
            testers = [m for m in range(cfg.M) if cfg.C_MT[m, t]]   # các tester làm được SP t
            for src in testers:
                for dst in testers:
                    if src != dst:
                        moves.append(("shift", (p, t, src, dst), cfg.LS_SHIFT))
    return moves


MOVES = _neighbor_moves()


def apply_move(plan, move):
    """Làm gì: áp một bước lân cận lên phương án đã giải mã.
    Nhận vào: plan dict (K_m, K_a, X_m, X_a, Q) từ mdl.decode; move một bước (kind, idx, step).
    Trả ra: dict MỚI (plan gốc không bị sửa), hoặc None nếu bước vô nghĩa hay vượt giới hạn.
    Ví dụ: bớt 1 tester khi đã về mức ban đầu -> None; thuê quá giới hạn mỗi quý -> None;
    chuyển sản lượng khi tester nguồn đang làm 0 -> None."""
    kind, idx, step = move
    plan = {name: np.array(value, copy=True) for name, value in plan.items()}   # sao chép để không sửa bản gốc
    if kind == "K_m":                                           # số tester sở hữu
        new = plan["K_m"][idx] + step
        if not (cfg.K0_M[idx] <= new <= cfg.K0_M[idx] + cfg.K_EXTRA_M):
            return None                                         # không bán dưới mức ban đầu
        plan["K_m"][idx] = new
    elif kind == "K_a":                                         # số handler sở hữu
        new = plan["K_a"][idx] + step
        if not (cfg.K0_A[idx] <= new <= cfg.K0_A[idx] + cfg.K_EXTRA_A):
            return None
        plan["K_a"][idx] = new
    elif kind == "X_m":                                         # số tester thuê ở 1 quý
        new = plan["X_m"][idx] + step
        if not (0 <= new <= cfg.X_MAX_M):
            return None
        plan["X_m"][idx] = new
    elif kind == "X_a":                                         # số handler thuê ở 1 quý
        new = plan["X_a"][idx] + step
        if not (0 <= new <= cfg.X_MAX_A):
            return None
        plan["X_a"][idx] = new
    else:                                                       # shift: chuyển sản lượng
        p, t, src, dst = idx
        amount = step * plan["Q"][p, src, t]                    # 10% lượng tester nguồn đang làm
        if amount <= 0:
            return None
        plan["Q"][p, src, t] -= amount
        plan["Q"][p, dst, t] = min(plan["Q"][p, dst, t] + amount, mdl.q_max()[t])
    return plan


def reverse_move(move):
    """Làm gì: cho bước ngược lại của một bước (dùng cho danh sách tabu).
    Nhận vào: move (kind, idx, step). Trả ra: bước ngược.
    Ví dụ: "thêm 1 tester M2 thuê ở quý 4" <-> "bớt 1 tester M2 thuê ở quý 4";
    "chuyển sản lượng từ M1 sang M2" <-> "chuyển từ M2 sang M1"."""
    kind, idx, step = move
    if kind == "shift":
        p, t, src, dst = idx
        return (kind, (p, t, dst, src), step)
    return (kind, idx, -step)


def _sample_valid_moves(plan, moves, rng, n):
    """Làm gì: bốc ngẫu nhiên tối đa n bước HỢP LỆ khác nhau từ danh sách moves.
    Nhận vào: plan phương án đã giải mã; moves danh sách bước; rng bộ sinh số ngẫu nhiên; n số bước cần.
    Trả ra: danh sách (bước, phương án sau khi áp bước đó). Duyệt theo thứ tự ngẫu nhiên, bỏ qua bước không hợp lệ."""
    out = []
    for i in rng.permutation(len(moves)):
        if len(out) >= n:
            break
        new_plan = apply_move(plan, moves[i])
        if new_plan is not None:
            out.append((moves[i], new_plan))
    return out


def _make_trace_step(kind, move, z_from, z_to, **extra):
    """Làm gì: ghi một bước vào "dấu vết" để demo/log hiển thị.
    Nhận vào: kind (better / worse / equal / aspiration / blocked); move; z_from, z_to là Z trước và sau bước.
    Trả ra: dict mô tả bước (khóa kind, move, f_from, f_to và các khóa phụ)."""
    return dict(kind=kind, move=move, f_from=z_from, f_to=z_to, **extra)


def _classify_step(new_z, current_z):
    """Làm gì: phân loại bước dựa trên Z mới so với Z cũ. Trả ra: "worse", "better" hoặc "equal"."""
    return "worse" if new_z < current_z else ("better" if new_z > current_z else "equal")


# =====================================================================================
# 4. TABU SEARCH (mài một cá thể bằng cách đi có "bộ nhớ")
# =====================================================================================
class TabuList:
    """Làm gì: danh sách tabu - bộ nhớ ngắn hạn của Tabu Search.
    Nhận vào (khi tạo): tenure, số vòng một bước bị cấm.
    Trả ra: đối tượng có tick() (sang vòng mới), forbid(move) (cấm bước ngược), is_tabu(move) (hỏi có bị cấm không).

    Mỗi lần đi 1 bước, BƯỚC NGƯỢC của nó bị cấm trong `tenure` vòng lặp.
    Ví dụ: vừa "bớt 1 tester M2 thuê ở quý 4" thì cấm "thêm 1 tester M2 thuê ở quý 4" trong 7 vòng.
    Mục đích: không quay lại phương án vừa rời đi -> không đi vòng tròn, buộc khám phá chỗ mới.
    Cài đặt: until[move] = vòng cuối cùng mà move còn bị cấm; it = đồng hồ vòng lặp hiện tại."""

    def __init__(self, tenure=cfg.TS_TENURE):
        self.tenure = tenure
        self.it = 0
        self.until = {}

    def tick(self):
        """Sang vòng lặp mới (đồng hồ cộng 1)."""
        self.it += 1

    def forbid(self, move):
        """Vừa đi `move` -> cấm bước ngược của nó thêm `tenure` vòng."""
        self.until[reverse_move(move)] = self.it + self.tenure

    def is_tabu(self, move):
        """Trả ra True nếu move còn bị cấm ở vòng hiện tại."""
        return self.until.get(move, -1) >= self.it


def tabu_search(start_x, start_z, counted_evaluate, rng, max_evals=cfg.LS_MAX_EVALS, moves=None, tabu=None,
                trace=None, n_cand=cfg.TS_N_CAND):
    """Làm gì: Tabu Search có lấy mẫu lân cận - mài một cá thể bằng cách đi từng bước, kể cả bước làm Z giảm,
    nhưng không đi lại bước vừa rời.
    Nhận vào: start_x cá thể cần mài; start_z Z của nó; counted_evaluate bộ đếm (EvaluationCounter);
        rng bộ sinh số ngẫu nhiên; max_evals số lần evaluate tối đa (20); moves danh sách bước (mặc định MOVES);
        tabu đối tượng TabuList (truyền CÙNG một đối tượng cho 3 cá thể để dùng chung); trace danh sách ghi dấu vết;
        n_cand số bước ứng viên mỗi vòng (5).
    Trả ra: (cá thể tốt nhất gặp được, Z của nó, số lần evaluate đã dùng).
    Ví dụ: cá thể có Z 47.0 triệu; vòng 1 xem 5 bước, bước tốt nhất cho 47.2 triệu -> đi; vòng 2 bước tốt nhất
    chỉ còn 47.1 triệu (tệ hơn) nhưng vẫn đi để thoát khỏi chỗ kẹt; cuối cùng trả về cá thể 47.2 triệu.

    BƯỚC 1: Khởi tạo: lấy cá thể xuất phát làm kỷ lục tốt nhất (best), chưa dùng lần evaluate nào.
    BƯỚC 2: Sang vòng mới: đồng hồ tabu cộng 1.
    BƯỚC 3: Bốc ngẫu nhiên n_cand bước hợp lệ, tính Z của từng bước (tốn n_cand lần evaluate).
    BƯỚC 4: Lọc bước được phép: bước KHÔNG bị cấm, cộng bước bị cấm nhưng Z vượt kỷ lục toàn cục (aspiration).
    BƯỚC 5: Nếu chưa có bước nào được phép thì bốc lại (tối đa TS_REDRAWS = 3 lần nữa); vẫn không có thì dừng.
    BƯỚC 6: Chọn bước có Z lớn nhất trong các bước được phép và ĐI TỚI đó (kể cả khi Z giảm).
    BƯỚC 7: Cấm bước ngược của bước vừa đi trong tenure (7) vòng.
    BƯỚC 8: Nếu Z mới vượt kỷ lục best thì cập nhật best.
    BƯỚC 9: Còn ngân sách (chưa đủ max_evals) thì quay lại BƯỚC 2; hết thì trả về best.
    (Trả best nên không bao giờ làm tệ đi cá thể được mài.)

    ĐIỂM MỚI (đóng góp chính): danh sách tabu dùng CHUNG cho 3 cá thể của cùng một lần kích hoạt.
    Đồng hồ vòng lặp chạy nối tiếp qua các cá thể và danh sách được tạo mới ở đầu mỗi lần kích hoạt,
    nên cá thể sau không đảo ngược các bước mà cá thể trước vừa đi. Lý do: 3 cá thể tốt nhất thường rất giống
    nhau và hay lặp lại cùng một chuỗi bước; tabu chung buộc 3 lần mài đi theo 3 hướng khác nhau."""
    moves = MOVES if moves is None else moves
    tabu = TabuList() if tabu is None else tabu
    current_x, current_z = start_x, start_z
    best_x, best_z = start_x, start_z           # BƯỚC 1: kỷ lục trong lần mài này
    used = 0                                    # số lần evaluate đã dùng
    while used < max_evals and not counted_evaluate.should_stop():
        tabu.tick()                             # BƯỚC 2: vòng mới
        plan = mdl.decode(current_x)            # giải mã để áp bước
        chosen = None
        for _ in range(1 + cfg.TS_REDRAWS):     # BƯỚC 5: bốc lần đầu + tối đa 3 lần bốc lại
            n = min(n_cand, max_evals - used, counted_evaluate.available())   # không vượt ngân sách
            if n <= 0:
                break
            candidates = _sample_valid_moves(plan, moves, rng, n)             # BƯỚC 3
            if not candidates:
                break
            # phải chụp trước khi đánh giá, vì counted_evaluate tự cập nhật best_f khi được gọi
            record = counted_evaluate.best_f    # kỷ lục toàn cục (dùng cho aspiration)
            allowed = []                        # (Z, bước, cá thể mới, có phải aspiration không)
            for move, new_plan in candidates:
                new_x = mdl.repair(mdl.encode(new_plan))
                new_z = counted_evaluate(new_x)
                used += 1
                if not tabu.is_tabu(move):      # BƯỚC 4: bước bình thường
                    allowed.append((new_z, move, new_x, False))
                elif new_z > record:            # BƯỚC 4: bị cấm nhưng phá kỷ lục -> vẫn cho đi (aspiration)
                    allowed.append((new_z, move, new_x, True))
                elif trace is not None:         # bị cấm và không đủ tốt -> ghi lại để log
                    trace.append(_make_trace_step("blocked", move, current_z, new_z))
                if counted_evaluate.should_stop():   # hết giờ giữa chừng -> dừng ngay, giữ best
                    break
            if allowed:
                chosen = max(allowed, key=lambda c: c[0])   # BƯỚC 6: tốt nhất trong các bước được phép, kể cả khi tệ hơn
                break
        if chosen is None:                      # BƯỚC 5: bốc 4 lần đều bị cấm hết -> dừng
            break
        new_z, move, new_x, aspired = chosen
        if trace is not None:
            trace.append(_make_trace_step("aspiration" if aspired else _classify_step(new_z, current_z),
                                          move, current_z, new_z))
        tabu.forbid(move)                       # BƯỚC 7: cấm bước ngược
        current_x, current_z = new_x, new_z     # BƯỚC 6: ĐI TỚI bước đã chọn
        if current_z > best_z:                  # BƯỚC 8: nhớ phương án tốt nhất
            best_x, best_z = current_x, current_z
    return best_x, best_z, used                 # BƯỚC 9


# =====================================================================================
# 5. SIMULATED ANNEALING (mài một cá thể bằng cách đi ngẫu nhiên, càng về sau càng "khó tính")
# =====================================================================================
def simulated_annealing(start_x, start_z, counted_evaluate, rng, max_evals=cfg.LS_MAX_EVALS, moves=None,
                        T0=1.0, trace=None):
    """Làm gì: Simulated Annealing (luật Metropolis + làm nguội cấp số nhân) - mài một cá thể bằng cách thử
    từng bước ngẫu nhiên, đôi khi nhận cả bước tệ hơn, và càng về sau càng ít nhận bước tệ.
    Nhận vào: start_x cá thể cần mài; start_z Z của nó; counted_evaluate bộ đếm; rng bộ sinh số ngẫu nhiên;
        max_evals số lần evaluate tối đa (20); moves danh sách bước; T0 nhiệt độ ban đầu (từ estimate_t0);
        trace danh sách ghi dấu vết.
    Trả ra: (cá thể tốt nhất gặp được, Z của nó, số lần evaluate đã dùng).
    Ví dụ: lúc đầu T cao, một bước làm Z giảm 0.5 triệu có thể được nhận với xác suất 40%;
    cuối đợt T chỉ còn 1% T0 nên gần như chỉ nhận bước tốt hơn.

    BƯỚC 1: Khởi tạo: hệ số làm nguội alpha = SA_FINAL_RATIO^(1/max_evals) (sau 20 lần T còn 1% T0); best = cá thể xuất phát.
    BƯỚC 2: Bốc 1 bước lân cận hợp lệ ngẫu nhiên, áp lên cá thể hiện tại, tính Z mới.
    BƯỚC 3: Tính nhiệt độ hiện tại T = T0 x alpha^i (i là số lần đã evaluate).
    BƯỚC 4: Quyết định nhận: Z mới >= Z hiện tại thì nhận; ngược lại nhận với xác suất exp((Z mới - Z hiện tại) / T).
    BƯỚC 5: Nếu nhận: đi tới cá thể mới; nếu Z vượt kỷ lục best thì cập nhật best.
    BƯỚC 6: Chưa dùng đủ max_evals thì quay lại BƯỚC 2; đủ thì trả về best.

    Khác Tabu: không có bộ nhớ, mỗi vòng chỉ xem 1 bước (thay vì 5), và việc nhận bước tệ là NGẪU NHIÊN theo
    nhiệt độ. Với chỉ 20 lần evaluate, lịch làm nguội rất ngắn - đây là giả thuyết cần kiểm chứng:
    SA có thể không kịp phát huy so với Tabu."""
    moves = MOVES if moves is None else moves
    alpha = cfg.SA_FINAL_RATIO ** (1 / max_evals)   # BƯỚC 1: hệ số làm nguội
    current_x, current_z = start_x, start_z
    best_x, best_z = start_x, start_z
    used = 0
    while used < max_evals and not counted_evaluate.should_stop():
        candidates = _sample_valid_moves(mdl.decode(current_x), moves, rng, 1)   # BƯỚC 2: 1 bước hợp lệ ngẫu nhiên
        if not candidates:
            break
        move, new_plan = candidates[0]
        new_x = mdl.repair(mdl.encode(new_plan))
        new_z = counted_evaluate(new_x)
        used += 1
        T = T0 * alpha ** used                  # BƯỚC 3: nhiệt độ hiện tại
        if new_z >= current_z:                  # BƯỚC 4: tốt hơn hoặc bằng: luôn nhận
            accept, p = True, 1.0
        else:                                   # BƯỚC 4: tệ hơn: nhận với xác suất p
            p = math.exp((new_z - current_z) / T)
            accept = rng.random() < p
        if accept:                              # BƯỚC 5
            if trace is not None:
                trace.append(_make_trace_step(_classify_step(new_z, current_z), move, current_z, new_z, T=T, p=p))
            current_x, current_z = new_x, new_z
            if current_z > best_z:
                best_x, best_z = current_x, current_z
    return best_x, best_z, used                 # BƯỚC 6


def estimate_t0(population, fitness, counted_evaluate, rng, moves=None, p0=cfg.SA_P0, samples=cfg.SA_T0_SAMPLES):
    """Làm gì: ước lượng nhiệt độ ban đầu T0 cho SA (làm 1 lần ở đầu mỗi lần chạy GA-SA).
    Nhận vào: population mảng các cá thể; fitness mảng Z tương ứng; counted_evaluate bộ đếm; rng bộ sinh số
        ngẫu nhiên; moves danh sách bước; p0 xác suất nhận bước tệ điển hình lúc đầu (0.5); samples số lần thử (30).
    Trả ra: T0 (một số dương cùng thang đo với Z). Nếu không bước nào làm Z giảm thì trả 1.0.
    Ví dụ: 30 lần thử, mức giảm Z thường gặp là 0.3 triệu -> T0 = 0.3 / ln(2) = 0.43 triệu, nghĩa là lúc đầu
    một bước làm tệ 0.3 triệu được nhận với xác suất 50%.

    BƯỚC 1: Chuẩn bị danh sách mức giảm Z, ban đầu rỗng.
    BƯỚC 2: Lặp samples (30) lần: chọn ngẫu nhiên 1 cá thể, áp 1 bước ngẫu nhiên hợp lệ, tính Z.
    BƯỚC 3: Nếu Z giảm thì ghi mức giảm vào danh sách.
    BƯỚC 4: T0 = -median(mức giảm) / ln(p0), nên exp(-median / T0) = p0.

    Cách này tự co giãn theo thang đo của Z (hàng triệu) nên không phải chỉnh tay.
    30 lần evaluate ở đây đi qua bộ đếm, nên TÍNH VÀO ngân sách 6000."""
    moves = MOVES if moves is None else moves
    drops = []                                  # BƯỚC 1: các mức Z bị giảm
    for _ in range(min(samples, counted_evaluate.available())):      # BƯỚC 2
        if counted_evaluate.should_stop():      # hết giờ -> dừng ước lượng sớm, dùng T0 mặc định
            break
        i = rng.integers(len(population))
        candidates = _sample_valid_moves(mdl.decode(population[i]), moves, rng, 1)
        if not candidates:
            continue
        new_z = counted_evaluate(mdl.repair(mdl.encode(candidates[0][1])))
        if new_z < fitness[i]:                  # BƯỚC 3: chỉ quan tâm các bước làm tệ
            drops.append(fitness[i] - new_z)
    return -float(np.median(drops)) / math.log(p0) if drops else 1.0     # BƯỚC 4


# =====================================================================================
# 6. HILL CLIMBING - CHỈ GIỮ ĐỂ SO SÁNH PHỤ (ablation), KHÔNG thuộc 3 thuật toán chính
# =====================================================================================
def hill_climbing(start_x, start_z, counted_evaluate, rng, max_evals=cfg.LS_MAX_EVALS, moves=None, trace=None):
    """Làm gì: hill climbing "gặp bước tốt hơn là nhận ngay" (first improvement) - bản tìm kiếm cục bộ cũ
    của giai đoạn Memetic. Chỉ dùng để chạy ablation (so sánh phụ) trong cong_cu/main.py, kích hoạt bằng ls_method="hc".
    Nhận vào: start_x cá thể cần mài; start_z Z của nó; counted_evaluate bộ đếm; rng bộ sinh số ngẫu nhiên;
        max_evals số lần evaluate tối đa; moves danh sách bước; trace danh sách ghi dấu vết.
    Trả ra: (cá thể sau khi mài, Z của nó, số lần evaluate đã dùng).
    Cách làm: duyệt các bước theo thứ tự ngẫu nhiên; bước nào làm Z tăng thì nhận ngay và duyệt lại từ đầu;
    không còn bước nào tốt hơn thì dừng.
    Nhược điểm: chỉ nhận bước tốt hơn nên dừng ngay ở "đỉnh đồi" đầu tiên gặp, không thoát ra được."""
    moves = MOVES if moves is None else moves
    current_x, current_z = start_x, start_z
    used = 0
    improved = True
    while improved and used < max_evals and not counted_evaluate.should_stop():
        improved = False
        plan = mdl.decode(current_x)
        for i in rng.permutation(len(moves)):
            if used >= max_evals or counted_evaluate.should_stop():
                break
            new_plan = apply_move(plan, moves[i])
            if new_plan is None:
                continue
            new_x = mdl.repair(mdl.encode(new_plan))   # mã hóa lại thành gen và sửa cho khả thi
            new_z = counted_evaluate(new_x)
            used += 1
            if new_z > current_z:                      # first improvement
                if trace is not None:
                    trace.append(_make_trace_step("better", moves[i], current_z, new_z))
                current_x, current_z, improved = new_x, new_z, True
                break
    return current_x, current_z, used


def _summarize_local_search(idx, rank, z_before, z_after, used, trace, **extra):
    """Làm gì: gói kết quả của một lần mài (cho demo và log).
    Nhận vào: idx vị trí cá thể trong quần thể; rank hạng của nó (0 = tốt nhất); z_before, z_after Z trước và sau
        khi mài; used số lần evaluate đã dùng; trace danh sách bước đã ghi.
    Trả ra: dict gồm số bước đã đi (n_steps), số bước tệ hơn đã đi (n_worse), số lần aspiration (n_aspiration),
        số bước bị cấm (n_blocked) và mức Z tăng thêm (gain)."""
    def count(kind):
        return sum(step["kind"] == kind for step in trace)
    return dict(idx=int(idx), rank=rank, f_before=z_before, f_after=z_after, used=used, steps=trace,
                n_steps=len(trace) - count("blocked"), n_worse=count("worse"),
                n_aspiration=count("aspiration"), n_blocked=count("blocked"),
                gain=z_after - z_before, **extra)


# =====================================================================================
# 7. KHUNG GA CHUNG (dùng cho SPGA gốc, GA-TS, GA-SA)
# =====================================================================================
def run_ga_iter(evaluate, ls_method="none", seed=0, budget=cfg.BUDGET,
                time_limit_sec=None, stopping_criterion="evaluations",
                ls_freq=cfg.LS_FREQ, ls_top=cfg.LS_TOP, ls_max_evals=cfg.LS_MAX_EVALS,
                pop_size=cfg.POP_SIZE, pc=cfg.PC, pm=cfg.PM, tour=cfg.TOURNAMENT, elite=cfg.ELITE,
                ls_shift=cfg.LS_SHIFT, lamarck=True,
                ts_n_cand=cfg.TS_N_CAND, ts_tenure=cfg.TS_TENURE, ts_shared=True, sa_p0=cfg.SA_P0):
    """Làm gì: vòng lặp GA chung, viết dạng generator: sau MỖI thế hệ nó "nhả" ra trạng thái hiện tại
    để giao diện demo vẽ trực tiếp.
    Nhận vào: evaluate hàm x -> (Z, lợi nhuận TB, MAD) đã gắn bộ kịch bản; ls_method "none" = SPGA gốc,
        "tabu" = GA-TS, "sa" = GA-SA, "hc" = hill climbing (ablation); seed hạt giống (cùng seed cho cùng kết quả);
        budget / time_limit_sec / stopping_criterion điều kiện dừng; các tham số còn lại xem config.py
        (ts_shared=False: mỗi cá thể một tabu riêng, chỉ để ablation; lamarck=False: chỉ cập nhật điểm, không
        ghi cá thể mài được vào quần thể, chỉ để ablation).
    Trả ra: mỗi lần yield là một dict gồm gen, pop, fit, evals, best_x, best_f, history, time_history,
        ls_events, ls_calls, runtime, done. Mỗi phần tử của ls_events có thêm n_steps, n_worse, n_aspiration,
        n_blocked, gain.
    Ví dụ: ls_method="tabu", budget=6000 -> khoảng 200 thế hệ; cứ 10 thế hệ mài 3 cá thể tốt nhất bằng Tabu Search.

    BƯỚC 1: Khởi tạo quần thể: pop_size cá thể ngẫu nhiên, sửa cho khả thi (repair), tính Z từng cá thể.
    BƯỚC 2: (chỉ GA-SA) Ước lượng nhiệt độ T0 từ quần thể ban đầu (tốn 30 lần evaluate).
    BƯỚC 3: Giữ nguyên cá thể tốt nhất sang thế hệ mới (elitism).
    BƯỚC 4: Chọn 2 cha mẹ bằng tournament.
    BƯỚC 5: Lai ghép 2 cha mẹ với xác suất pc (không lai thì con = bản sao cha mẹ).
    BƯỚC 6: Đột biến mỗi con, sửa cho khả thi (repair), tính Z, thêm vào quần thể mới. Lặp BƯỚC 4 đến 6 cho đến khi đủ quần thể.
    BƯỚC 7: (chỉ khi có tìm kiếm cục bộ và thế hệ chia hết cho ls_freq) mài ls_top cá thể tốt nhất bằng Tabu Search /
            Simulated Annealing / hill climbing, mỗi cá thể tối đa ls_max_evals lần evaluate.
    BƯỚC 8: Ghi cá thể mài được đè lên cá thể cũ trong quần thể (Lamarckian).
    BƯỚC 9: Kiểm tra điều kiện dừng. Chưa dừng thì quay lại BƯỚC 3; dừng thì kết quả là cá thể tốt nhất từng gặp."""
    if ls_method not in cfg.LS_METHODS:
        raise ValueError(f"ls_method phải thuộc {cfg.LS_METHODS}")
    rng = np.random.default_rng(seed)            # cùng seed -> cùng kết quả (tái lập được)
    # mọi lần tính Z đều qua bộ đếm này; nó đồng thời là đồng hồ đo thời gian thực và
    # chốt điều kiện dừng (hết ngân sách đánh giá HOẶC hết giờ tùy stopping_criterion)
    counted_evaluate = EvaluationCounter(evaluate, budget, time_limit_sec, stopping_criterion)
    # nếu đổi % chuyển sản lượng thì tạo lại danh sách bước với tỉ lệ mới
    moves = MOVES if ls_shift == cfg.LS_SHIFT else [
        (k, i, ls_shift) if k == "shift" else (k, i, st) for (k, i, st) in MOVES]
    t_start = time.perf_counter()

    # ---- BƯỚC 1: khởi tạo quần thể
    population = np.array([mdl.repair(rng.random(mdl.N_GENES)) for _ in range(pop_size)])
    fitness = np.array([counted_evaluate(x) for x in population])
    # ---- BƯỚC 2: chỉ GA-SA: ước lượng T0 từ quần thể ban đầu (tốn 30 lần evaluate)
    sa_t0 = estimate_t0(population, fitness, counted_evaluate, rng, moves, sa_p0) if ls_method == "sa" else None
    gen = 0
    ls_calls = 0                                 # tổng số lần evaluate dùng cho tìm kiếm cục bộ
    yield dict(gen=0, pop=population, fit=fitness, evals=counted_evaluate.n_calls,
               best_x=counted_evaluate.best_x, best_f=counted_evaluate.best_f,
               history=counted_evaluate.history, time_history=counted_evaluate.time_history,
               ls_events=[], ls_calls=0, runtime=time.perf_counter() - t_start, done=False)

    done = False
    while not done:                              # ---- BƯỚC 9: lặp cho đến khi hết ngân sách hoặc hết giờ
        gen += 1
        # ---- BƯỚC 3: elitism - giữ nguyên cá thể tốt nhất sang thế hệ mới
        order = np.argsort(-fitness)
        new_population = [population[i].copy() for i in order[:elite]]
        new_fitness = [fitness[i] for i in order[:elite]]
        # ---- BƯỚC 4 đến 6: sinh con cho đến khi đủ quần thể
        while len(new_population) < pop_size and not counted_evaluate.should_stop():
            parent_a = population[tournament(fitness, rng, tour)]        # BƯỚC 4: chọn cha mẹ
            parent_b = population[tournament(fitness, rng, tour)]
            if rng.random() < pc:                                        # BƯỚC 5: lai ghép với xác suất pc
                child1, child2 = crossover(parent_a, parent_b, rng)
            else:
                child1, child2 = parent_a.copy(), parent_b.copy()
            for child in (child1, child2):
                if len(new_population) >= pop_size or counted_evaluate.should_stop():
                    break
                child = mdl.repair(mutate(child, rng, pm))               # BƯỚC 6: đột biến + sửa khả thi
                new_population.append(child)
                new_fitness.append(counted_evaluate(child))              # chấm điểm (tốn 1 lần evaluate)
        population, fitness = np.array(new_population), np.array(new_fitness)

        # ---- BƯỚC 7 và 8: tìm kiếm cục bộ, cứ ls_freq thế hệ mài ls_top cá thể tốt nhất
        events = []
        if ls_method != "none" and gen % ls_freq == 0:
            tabu = TabuList(ts_tenure)                   # reset ở đầu mỗi lần kích hoạt
            for rank, i in enumerate(np.argsort(-fitness)[:ls_top]):
                if counted_evaluate.should_stop():
                    break
                trace = []
                z_before = fitness[i]
                extra = {}
                # BƯỚC 7: mài cá thể i
                if ls_method == "tabu":
                    if not ts_shared:                    # ablation: tabu riêng từng cá thể
                        tabu = TabuList(ts_tenure)
                    # mặc định: CÙNG đối tượng tabu cho cả 3 cá thể (tabu chung)
                    new_x, new_z, used = tabu_search(population[i], fitness[i], counted_evaluate, rng,
                                                     ls_max_evals, moves, tabu, trace, ts_n_cand)
                elif ls_method == "sa":
                    new_x, new_z, used = simulated_annealing(population[i], fitness[i], counted_evaluate, rng,
                                                             ls_max_evals, moves, sa_t0, trace)
                    extra["sa_t0"] = sa_t0
                else:
                    new_x, new_z, used = hill_climbing(population[i], fitness[i], counted_evaluate, rng,
                                                       ls_max_evals, moves, trace)
                ls_calls += used
                events.append(_summarize_local_search(i, rank, z_before, new_z, used, trace, **extra))
                # BƯỚC 8: ghi kết quả mài vào quần thể
                if lamarck:
                    population[i], fitness[i] = new_x, new_z   # Lamarckian: ghi cá thể tốt nhất gặp được
                else:
                    fitness[i] = new_z                          # Baldwinian: chỉ cập nhật điểm
        done = counted_evaluate.should_stop()            # BƯỚC 9
        yield dict(gen=gen, pop=population, fit=fitness, evals=counted_evaluate.n_calls,
                   best_x=counted_evaluate.best_x, best_f=counted_evaluate.best_f,
                   history=counted_evaluate.history, time_history=counted_evaluate.time_history,
                   ls_events=events, ls_calls=ls_calls,
                   runtime=time.perf_counter() - t_start, done=done)


def run_ga(evaluate, ls_method="none", seed=0, budget=cfg.BUDGET,
           time_limit_sec=None, stopping_criterion="evaluations", **kw):
    """Làm gì: chạy trọn 1 lần GA (không cần xem từng thế hệ), dùng trong cong_cu/main.py và cong_cu/tests.py.
    Nhận vào: evaluate hàm x -> (Z, lợi nhuận TB, MAD) đã gắn sẵn bộ kịch bản; ls_method "none" SPGA gốc,
        "tabu" GA-TS, "sa" GA-SA, "hc" hill climbing (ablation); seed hạt giống; điều kiện dừng:
        stopping_criterion="evaluations" (dùng budget) hoặc "time" (dùng time_limit_sec);
        tham số khác (ls_freq, ts_n_cand, ...) truyền thẳng cho run_ga_iter.
    Trả ra: (best_x, history, runtime): cá thể tốt nhất, lịch sử [(số lần evaluate, Z tốt nhất)], thời gian chạy (giây).
    Sau khi chạy, run_ga.last_info chứa thống kê tổng: số thế hệ, số lần evaluate cho tìm kiếm cục bộ,
    tổng số bước đã đi / bước tệ hơn / aspiration / bị cấm, tổng mức Z tăng nhờ tìm kiếm cục bộ, cùng chỉ số
    hiệu năng thời gian (runtime_actual, total_evaluations, evals_per_sec, time_history).
    Ví dụ: run_ga(evaluate, "tabu", seed=1) cho kết quả GA-TS của seed 1."""
    state = None
    totals = dict(ls_runs=0, ls_steps=0, ls_worse=0, ls_aspiration=0, ls_blocked=0, ls_gain=0.0)
    for state in run_ga_iter(evaluate, ls_method, seed, budget, time_limit_sec, stopping_criterion, **kw):
        for event in state["ls_events"]:         # cộng dồn thống kê tìm kiếm cục bộ
            totals["ls_runs"] += 1
            totals["ls_steps"] += event["n_steps"]
            totals["ls_worse"] += event["n_worse"]
            totals["ls_aspiration"] += event["n_aspiration"]
            totals["ls_blocked"] += event["n_blocked"]
            totals["ls_gain"] += event["gain"]
    runtime = state["runtime"]
    run_ga.last_info = {
        "generations": state["gen"], "ls_evals": state["ls_calls"], "evals": state["evals"],
        "runtime_actual": runtime, "total_evaluations": state["evals"],
        "evals_per_sec": state["evals"] / runtime if runtime > 0 else 0.0,
        "time_history": state["time_history"], **totals}
    return state["best_x"], state["history"], runtime


# =====================================================================================
# 8. CÁC HÀM SOLVE
# Đây là "cửa chính" của file: gọi một hàm là chạy một thuật toán trọn vẹn với số liệu trong config.py.
# solve_spga, solve_ga_ts, solve_ga_sa là 3 thuật toán được so sánh với nhau.
# solve_milp là bài toán MILP giải bằng thư viện PuLP: CHỈ là đáp án chuẩn (cận tối ưu toàn cục)
# để xem 3 thuật toán ở trên còn cách tối ưu bao xa. KHÔNG phải là thuật toán thứ tư trong cuộc so sánh.
# =====================================================================================
def _solve_with_ga(ls_method, seed, scenarios, budget, time_limit_sec, stopping_criterion, ga_options):
    """Làm gì: phần chung của 3 hàm solve_*: dựng hàm evaluate từ config, chạy run_ga, gói kết quả.
    Nhận vào: ls_method kiểu tìm kiếm cục bộ; các tham số còn lại giống solve_spga.
    Trả ra: dict gồm best_x, z, mean_profit, mad, history, runtime."""
    if scenarios is None:
        scenarios = mdl.sample_scenarios(cfg.N_SCEN, cfg.SCEN_SEED)

    def evaluate(x):
        return mdl.evaluate(x, scenarios)

    best_x, history, runtime = run_ga(evaluate, ls_method, seed=seed, budget=budget,
                                      time_limit_sec=time_limit_sec, stopping_criterion=stopping_criterion,
                                      **ga_options)
    z, mean_profit, mad = evaluate(best_x)
    return {"best_x": best_x, "z": z, "mean_profit": mean_profit, "mad": mad,
            "history": history, "runtime": runtime}


def solve_spga(seed=1, scenarios=None, budget=cfg.BUDGET, time_limit_sec=None,
               stopping_criterion="evaluations", **ga_options):
    """Làm gì: chạy SPGA gốc (chỉ khung GA, không tìm kiếm cục bộ).
    Nhận vào: seed hạt giống; scenarios bộ kịch bản nhu cầu (bỏ trống thì sinh từ config: cfg.N_SCEN kịch bản,
        hạt giống cfg.SCEN_SEED); budget số lần evaluate; time_limit_sec và stopping_criterion nếu muốn dừng theo
        thời gian; ga_options tham số GA khác (xem run_ga_iter).
    Trả ra: dict gồm best_x (phương án tốt nhất, 135 gen), z (Z của nó, đồng), mean_profit (lợi nhuận trung bình),
        mad (độ dao động lợi nhuận), history (lịch sử [(số lần evaluate, Z tốt nhất)]), runtime (giây).
    Ví dụ: solve_spga(seed=1)["z"] cho điểm Z của SPGA gốc ở seed 1."""
    return _solve_with_ga("none", seed, scenarios, budget, time_limit_sec, stopping_criterion, ga_options)


def solve_ga_ts(seed=1, scenarios=None, budget=cfg.BUDGET, time_limit_sec=None,
                stopping_criterion="evaluations", **ga_options):
    """Làm gì: chạy GA-TS (khung GA + mài bằng Tabu Search).
    Nhận vào, Trả ra: giống solve_spga.
    Ví dụ: solve_ga_ts(seed=1)["z"] cho điểm Z của GA-TS ở seed 1."""
    return _solve_with_ga("tabu", seed, scenarios, budget, time_limit_sec, stopping_criterion, ga_options)


def solve_ga_sa(seed=1, scenarios=None, budget=cfg.BUDGET, time_limit_sec=None,
                stopping_criterion="evaluations", **ga_options):
    """Làm gì: chạy GA-SA (khung GA + mài bằng Simulated Annealing).
    Nhận vào, Trả ra: giống solve_spga.
    Ví dụ: solve_ga_sa(seed=1)["z"] cho điểm Z của GA-SA ở seed 1."""
    return _solve_with_ga("sa", seed, scenarios, budget, time_limit_sec, stopping_criterion, ga_options)


def solve_milp(k=None, sigma=None, n_scen=None, scen_seed=None, time_limit=None, msg=0, mu=None):
    """Làm gì: giải bài toán dưới dạng MILP (quy hoạch tuyến tính nguyên hỗn hợp) bằng PuLP + CBC để lấy
    CẬN TỐI ƯU TOÀN CỤC. Đây là BENCHMARK (đáp án chuẩn để so sánh), không thuộc 3 thuật toán được so sánh.
    Cùng bộ kịch bản (seed 2026) và cùng mục tiêu Z = (1 - k) x mean(h) - k x MAD(h) với 3 thuật toán.
    Nhận vào: k mức ngại rủi ro (mặc định cfg.K_RISK); sigma độ lệch chuẩn nhu cầu (mặc định cfg.SIGMA);
        n_scen số kịch bản (mặc định cfg.N_SCEN); scen_seed hạt giống kịch bản (mặc định cfg.SCEN_SEED);
        time_limit giây tối đa cho bộ giải (mặc định cfg.MILP_TIME_LIMIT_SEC); msg 1 để bộ giải in log;
        mu nhu cầu trung bình dạng bản chụp (để chạy nền an toàn khi demo đổi cfg.MU giữa chừng).
    Trả ra: dict gồm z_opt, mean_opt, mad_opt (đồng), status (tình trạng của bộ giải, "Optimal" là đã chứng minh
        tối ưu), solve_time_s, k, sigma, n_scen, scen_seed, solution (số máy mua, thuê, sản lượng của lời giải).
    Cần cài thư viện pulp; thư viện chỉ được import khi hàm này chạy.

    Tuyến tính hóa:
      MAD        : u_n >= h_n - mean, u_n >= mean - h_n  ->  MAD = (1/N) x tổng u_n (đúng khi k > 0)
      Tồn kho MTS: S = S+ - S-, chi phí = J_HOLD x S+ + L_BACK x S-
      Bán MTO    : sold <= sản lượng, sold <= nhu cầu (bán tối đa bằng nhu cầu)
      Vốn        : F_p = (1 + r) x F_(p-1) + doanh thu - chi phí thuê - chi phí tồn kho/thiếu
      Năng lực   : tổng Q/CAP <= máy sở hữu + máy thuê (tester theo từng loại, handler theo ghép cặp)
    Đơn vị nội bộ (để hệ số nhỏ, CBC giải nhanh và ổn định): tiền triệu đồng, sản lượng nghìn đơn vị.
    Kết quả trả về đã đổi ngược về đồng để so sánh trực tiếp với 3 thuật toán."""
    import pulp

    k = cfg.K_RISK if k is None else float(k)
    sigma = cfg.SIGMA if sigma is None else float(sigma)
    n_scen = cfg.N_SCEN if n_scen is None else n_scen
    scen_seed = cfg.SCEN_SEED if scen_seed is None else scen_seed
    time_limit = cfg.MILP_TIME_LIMIT_SEC if time_limit is None else time_limit
    mu_arr = cfg.MU if mu is None else np.asarray(mu, dtype=float)

    # CÙNG kịch bản với 3 thuật toán, đổi sang đơn vị nghìn
    scen = mdl.sample_scenarios(n_scen, scen_seed, mu=mu_arr, sigma=sigma) / 1000.0

    P, M, A, T = cfg.P, cfg.M, cfg.A, cfg.T      # số quý, tester, handler, sản phẩm
    # ---- hằng số theo đơn vị nội bộ (tiền: triệu; sản lượng: nghìn đơn vị)
    F0, RATE = float(cfg.F0) / 1e6, float(cfg.I_RATE)
    JH, LB = float(cfg.J_HOLD[0]) * 1000 / 1e6, float(cfg.L_BACK[0]) * 1000 / 1e6
    Bc = [float(v) * 1000 / 1e6 for v in cfg.B]
    UMc = [float(v) / 1e6 for v in cfg.U_M]
    UAc = [float(v) / 1e6 for v in cfg.U_A]
    EMc = [float(v) / 1e6 for v in cfg.E_M]
    DMc = [float(v) / 1e6 for v in cfg.D_M]
    EAc = [float(v) / 1e6 for v in cfg.E_A]
    DAc = [float(v) / 1e6 for v in cfg.D_A]
    CAPMc = [[float(v) / 1000.0 for v in row] for row in mdl.CAP_M]
    CAPHc = [float(v) / 1000.0 for v in mdl.CAP_H]
    QMAXc = [cfg.Q_MAX_FACTOR * float(v) / 1000.0 for v in mu_arr]
    K0Mc = [int(v) for v in cfg.K0_M]
    K0Ac = [int(v) for v in cfg.K0_A]
    KEXT_M, KEXT_A = int(cfg.K_EXTRA_M), int(cfg.K_EXTRA_A)
    XMAX_M, XMAX_A = int(cfg.X_MAX_M), int(cfg.X_MAX_A)

    prob = pulp.LpProblem("SPGA_MILP", pulp.LpMaximize)

    # ---- biến quyết định: máy mua (nguyên), máy thuê (nguyên), sản lượng (liên tục)
    Km = [pulp.LpVariable(f"Km_{m}", K0Mc[m], K0Mc[m] + KEXT_M, cat="Integer") for m in range(M)]
    Ka = [pulp.LpVariable(f"Ka_{a}", K0Ac[a], K0Ac[a] + KEXT_A, cat="Integer") for a in range(A)]
    Xm = [[pulp.LpVariable(f"Xm_{p}_{m}", 0, XMAX_M, cat="Integer") for m in range(M)] for p in range(P)]
    Xa = [[pulp.LpVariable(f"Xa_{p}_{a}", 0, XMAX_A, cat="Integer") for a in range(A)] for p in range(P)]
    Q = [[[pulp.LpVariable(f"Q_{p}_{m}_{t}", 0, QMAXc[t]) if cfg.C_MT[m, t] else None
           for t in range(T)] for m in range(M)] for p in range(P)]

    def prod(p, t):
        return pulp.lpSum(Q[p][m][t] for m in range(M) if Q[p][m][t] is not None)

    out_cost = [pulp.lpSum(Xm[p][m] * UMc[m] for m in range(M)) +
                pulp.lpSum(Xa[p][a] * UAc[a] for a in range(A)) for p in range(P)]
    invest = (pulp.lpSum((EMc[m] - DMc[m]) * (Km[m] - K0Mc[m]) for m in range(M)) +
              pulp.lpSum((EAc[a] - DAc[a]) * (Ka[a] - K0Ac[a]) for a in range(A)))

    # ---- ràng buộc năng lực
    for p in range(P):
        for m in range(M):
            prob += (pulp.lpSum(Q[p][m][t] * (1.0 / CAPMc[m][t]) for t in range(T) if Q[p][m][t] is not None)
                     <= Km[m] + Xm[p][m])
        for a in range(A):
            load = [Q[p][m][t] * (1.0 / CAPHc[a])
                    for m in range(M) for t in range(T)
                    if Q[p][m][t] is not None and cfg.H_OF[m, t] == a]
            prob += pulp.lpSum(load) <= Ka[a] + Xa[p][a]

    # ---- kịch bản nhu cầu
    h_exprs, u = [], []
    for n in range(n_scen):
        F = [None] * P
        S = [None] * P
        Sp = [pulp.LpVariable(f"Sp_{n}_{p}", 0) for p in range(P)]
        Sm = [pulp.LpVariable(f"Sm_{n}_{p}", 0) for p in range(P)]
        sold = {(p, t): pulp.LpVariable(f"sold_{n}_{p}_{t}", 0) for p in range(P) for t in range(1, T)}
        for p in range(P):
            # (5) cân bằng tồn kho MTS (SP1)
            S[p] = (prod(p, 0) - float(scen[n, p, 0])) if p == 0 else (S[p - 1] + prod(p, 0) - float(scen[n, p, 0]))
            prob += Sp[p] - Sm[p] == S[p]
            V = JH * Sp[p] + LB * Sm[p]
            # (6) MTO: chỉ bán được tối đa bằng nhu cầu
            income = Bc[0] * float(scen[n, p, 0])
            for t in range(1, T):
                prob += sold[(p, t)] <= prod(p, t)
                prob += sold[(p, t)] <= float(scen[n, p, t])
                income = income + Bc[t] * sold[(p, t)]
            # (8) cân bằng vốn
            F[p] = ((F0 * (1 + RATE) - out_cost[p] - V + income) if p == 0
                    else (F[p - 1] * (1 + RATE) - out_cost[p] - V + income))
        # (9) lợi nhuận = giá trị hiện tại vốn cuối kỳ - đầu tư ròng
        h = F[P - 1] * (1.0 / (1 + RATE) ** P) - invest
        h_exprs.append(h)
        u.append(pulp.LpVariable(f"u_{n}", 0))

    # MAD tuyến tính hóa + mục tiêu max (1-k) x tổng h - k x tổng u  (tương đương Z x N)
    mean = pulp.lpSum(h_exprs) * (1.0 / n_scen)
    for n in range(n_scen):
        prob += u[n] >= h_exprs[n] - mean
        prob += u[n] >= mean - h_exprs[n]
    prob += (1 - k) * pulp.lpSum(h_exprs) - k * pulp.lpSum(u)

    t_start = time.perf_counter()
    # CBC đi kèm pulp <= 2.9; nếu bản pulp mới không có thì dùng bộ giải mặc định
    solver_class = pulp.PULP_CBC_CMD if hasattr(pulp, "PULP_CBC_CMD") else pulp.LpSolverDefault
    status = prob.solve(solver_class(msg=msg, timeLimit=time_limit))
    solve_time = time.perf_counter() - t_start

    profits = [pulp.value(h_exprs[n]) * 1e6 for n in range(n_scen)]   # đổi ngược về đồng
    mean_opt = float(np.mean(profits))
    mad_opt = float(np.abs(np.array(profits) - mean_opt).mean())
    z_opt = (1 - k) * mean_opt - k * mad_opt

    solution = {
        "Km": [int(round(pulp.value(v))) for v in Km],
        "Ka": [int(round(pulp.value(v))) for v in Ka],
        "Xm": [[int(round(pulp.value(v))) for v in row] for row in Xm],
        "Xa": [[int(round(pulp.value(v))) for v in row] for row in Xa],
        "Q": [[[float(pulp.value(v)) * 1000.0 if v is not None else 0.0 for v in row] for row in mat] for mat in Q],
    }
    return {"z_opt": z_opt, "mean_opt": mean_opt, "mad_opt": mad_opt,
            "status": pulp.LpStatus[status], "solve_time_s": solve_time,
            "k": k, "sigma": sigma, "n_scen": n_scen, "scen_seed": scen_seed,
            "solution": solution}


# =====================================================================================
# CHẠY THỬ: python solver.py -> chạy 3 thuật toán cho 1 seed và in bảng so sánh
# =====================================================================================
if __name__ == "__main__":
    for stream in (sys.stdout, sys.stderr):      # in tiếng Việt được cả khi console là cp1252
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")

    demo_seed = cfg.SEEDS[0]
    demo_scenarios = mdl.sample_scenarios(cfg.N_SCEN, cfg.SCEN_SEED)
    print(f"So sánh 3 thuật toán: seed {demo_seed}, ngân sách {cfg.BUDGET} lần evaluate, "
          f"{cfg.N_SCEN} kịch bản (seed {cfg.SCEN_SEED})")
    print(f"{'Thuật toán':<10} {'Z (triệu)':>12} {'Lợi nhuận TB (triệu)':>22} {'MAD (triệu)':>13} {'Thời gian (s)':>14}")
    for name, solve in (("SPGA gốc", solve_spga), ("GA-TS", solve_ga_ts), ("GA-SA", solve_ga_sa)):
        result = solve(seed=demo_seed, scenarios=demo_scenarios)
        print(f"{name:<10} {result['z'] / 1e6:>12.3f} {result['mean_profit'] / 1e6:>22.3f} "
              f"{result['mad'] / 1e6:>13.3f} {result['runtime']:>14.1f}")
