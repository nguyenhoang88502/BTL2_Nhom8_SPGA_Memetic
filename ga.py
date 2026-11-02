"""
ga.py - SPGA gốc, GA + Tabu Search (GA-TS) và GA + Simulated Annealing (GA-SA).

TỔNG QUAN
---------
Cả 3 thuật toán dùng CHUNG một khung GA (giống SPGA của Wang, Wang & Chen 2008):
  quần thể 30, xác suất lai ghép 0.9, chọn lọc tournament 2, giữ 1 cá thể tốt nhất (elitism),
  lai ghép hybrid (ngẫu nhiên 1 trong 4: 1 điểm, 2 điểm, uniform, arithmetic),
  đột biến uniform pm = 0.05, ngân sách 6000 lần gọi evaluate (TÍNH CẢ lần gọi trong tìm kiếm cục bộ).

GA-TS và GA-SA chỉ thêm một bước "mài" (tìm kiếm cục bộ) vào khung đó, với cùng điểm kích hoạt:
  cứ 10 thế hệ, lấy 3 cá thể tốt nhất, mỗi cá thể được tối đa 20 lần gọi evaluate,
  dùng cùng tập 174 bước lân cận, và ghi ngược nghiệm TỐT NHẤT gặp được vào quần thể (Lamarckian).
=> Khác nhau DUY NHẤT giữa GA-TS và GA-SA là CHIẾN LƯỢC đi trong lân cận. So sánh vì vậy công bằng.

MÃ GIẢ CHUNG (xem chi tiết từng hàm bên dưới)
---------------------------------------------
  Khởi tạo quần thể ngẫu nhiên, sửa cho hợp lệ (repair), chấm điểm Z
  nếu GA-SA: ước lượng nhiệt độ ban đầu T0 (tốn 30 lần evaluate)
  lặp cho đến khi hết 6000 lần evaluate:
      thế hệ += 1
      giữ lại 1 cá thể tốt nhất
      lặp cho đến khi đủ 30 cá thể:
          chọn 2 cha mẹ bằng tournament
          lai ghép (xác suất 0.9), đột biến, repair, chấm điểm
      nếu có tìm kiếm cục bộ và thế hệ chia hết cho 10:
          với 3 cá thể tốt nhất:
              chạy Tabu Search / Simulated Annealing (tối đa 20 lần evaluate)
              ghi nghiệm tốt nhất tìm được đè lên cá thể đó
  trả về nghiệm tốt nhất từng gặp

Giao diện chính:
  run_ga(evaluate, ls_method="none"|"tabu"|"sa"|"hc", seed=...) -> (best_x, history, runtime)
  history: list các cặp (số lần gọi evaluate, fitness tốt nhất tới thời điểm đó)
  "hc" là bản hill climbing (first improvement) cũ, chỉ giữ để làm ablation.

THUẬT NGỮ dùng trong chú thích
  cá thể / nghiệm x : vector 135 gen trong [0,1], mã hóa 1 phương án mua/thuê máy và phân bổ sản lượng
  Z / fitness       : (1-k)·lợi nhuận trung bình - k·MAD trên 30 kịch bản nhu cầu (càng lớn càng tốt)
  evaluate          : 1 lần tính Z cho 1 nghiệm = đơn vị "chi phí" để so sánh công bằng
  bước lân cận      : 1 thay đổi nhỏ trên phương án (thêm/bớt 1 máy, chuyển 10% sản lượng...)
"""
import math
import time
import numpy as np

import model as mdl

# ============================================================ THAM SỐ
# --- tham số chung của khung GA (giống nhau ở cả 3 thuật toán)
POP_SIZE = 30            # kích thước quần thể
PC = 0.9                 # xác suất lai ghép
PM = 0.05                # xác suất đột biến mỗi gen
TOURNAMENT = 2           # số cá thể tham gia mỗi lần chọn lọc tournament
ELITE = 1                # số cá thể tốt nhất được giữ nguyên sang thế hệ sau
BUDGET = 6000            # tổng số lần gọi evaluate cho 1 lần chạy (điều kiện dừng)
# --- điểm kích hoạt tìm kiếm cục bộ (dùng chung cho GA-TS và GA-SA)
LS_FREQ = 10             # cứ 10 thế hệ chạy tìm kiếm cục bộ 1 lần
LS_TOP = 3               # chỉ mài 3 cá thể tốt nhất
LS_MAX_EVALS = 20        # mỗi cá thể được dùng tối đa 20 lần evaluate
LS_SHIFT = 0.10          # lân cận kiểu (3): chuyển 10% sản lượng giữa 2 tester
LS_METHODS = ("none", "tabu", "sa", "hc")
# --- Tabu Search
TS_N_CAND = 5            # mỗi vòng lặp xem thử 5 bước ứng viên (bốc ngẫu nhiên trong 174 bước)
TS_TENURE = 7            # bước ngược của bước vừa đi bị cấm trong 7 vòng lặp
TS_REDRAWS = 3           # nếu cả 5 ứng viên đều bị cấm thì bốc lại, tối đa 3 lần
# --- Simulated Annealing
SA_P0 = 0.5              # lúc đầu, 1 bước tệ "điển hình" được nhận với xác suất 50%
SA_T0_SAMPLES = 30       # số lần evaluate để ước lượng T0 (CÓ tính vào ngân sách 6000)
SA_FINAL_RATIO = 0.01    # đến lần evaluate cuối của mỗi lần SA thì T = 1% của T0


# ============================================================ BỘ ĐẾM ĐÁNH GIÁ
class _Counter:
    """Bọc hàm evaluate để đếm số lần gọi và ghi lịch sử best fitness.

    Mọi lần tính Z (kể cả trong Tabu/SA và lúc ước lượng T0) đều đi qua đây,
    nên ngân sách 6000 được kiểm soát ở MỘT chỗ duy nhất -> so sánh công bằng.
    Đồng thời nó tự nhớ nghiệm tốt nhất từng gặp (best_x, best_f) và lịch sử hội tụ."""

    def __init__(self, evaluate, budget):
        self.evaluate = evaluate     # hàm gốc: x -> (fitness, mean, mad)
        self.budget = budget         # tổng số lần được phép gọi
        self.n = 0                   # số lần đã gọi
        self.best_f = -np.inf        # Z tốt nhất từ đầu lần chạy
        self.best_x = None           # nghiệm tương ứng
        self.history = []            # [(n, best_f)] để vẽ đường hội tụ

    def left(self):
        """Số lần evaluate còn lại."""
        return self.budget - self.n

    def __call__(self, x):
        f = self.evaluate(x)[0]      # chỉ lấy fitness Z
        self.n += 1
        if f > self.best_f:          # cập nhật kỷ lục
            self.best_f, self.best_x = f, x.copy()
        self.history.append((self.n, self.best_f))
        return f


# ============================================================ TOÁN TỬ GA
def tournament(fit, rng, k=TOURNAMENT):
    """Chọn lọc tournament: bốc ngẫu nhiên k cá thể, lấy cá thể có Z cao nhất.
    k = 2 -> áp lực chọn lọc vừa phải, cá thể kém vẫn có cơ hội làm cha mẹ (giữ đa dạng)."""
    idx = rng.integers(0, len(fit), size=k)
    return idx[np.argmax(fit[idx])]


def crossover(p1, p2, rng):
    """Hybrid crossover: mỗi lần lai ghép chọn ngẫu nhiên 1 trong 4 kiểu.

    Mã giả:
        kind = ngẫu nhiên {0,1,2,3}
        0: 1 điểm   - cắt tại vị trí c, ghép đầu của cha với đuôi của mẹ
        1: 2 điểm   - đổi đoạn gen [a, b) giữa cha và mẹ
        2: uniform  - từng gen tung đồng xu xem lấy của cha hay mẹ
        3: arithmetic - con = λ·cha + (1-λ)·mẹ (trộn giá trị, hợp với gen liên tục)
    Trộn nhiều kiểu giúp vừa giữ được khối gen tốt (1, 2 điểm) vừa khám phá giá trị mới (arithmetic)."""
    n = len(p1)
    kind = rng.integers(4)
    if kind == 0:                                   # 1 điểm
        c = rng.integers(1, n)
        return np.r_[p1[:c], p2[c:]], np.r_[p2[:c], p1[c:]]
    if kind == 1:                                   # 2 điểm
        a, b = sorted(rng.choice(np.arange(1, n), 2, replace=False))
        c1, c2 = p1.copy(), p2.copy()
        c1[a:b], c2[a:b] = p2[a:b], p1[a:b]
        return c1, c2
    if kind == 2:                                   # uniform
        mask = rng.random(n) < 0.5
        return np.where(mask, p1, p2), np.where(mask, p2, p1)
    lam = rng.random()                              # arithmetic
    return lam * p1 + (1 - lam) * p2, lam * p2 + (1 - lam) * p1


def mutate(x, rng, pm=PM):
    """Uniform mutation: mỗi gen có xác suất pm được thay bằng giá trị U(0,1) mới.
    Với 135 gen và pm = 0.05, trung bình khoảng 7 gen bị đổi mỗi cá thể con."""
    mask = rng.random(len(x)) < pm
    x = x.copy()
    x[mask] = rng.random(mask.sum())
    return x


# ============================================================ BƯỚC LÂN CẬN
# Một "bước lân cận" là một thay đổi nhỏ, có ý nghĩa nghiệp vụ, trên phương án đã giải mã.
# Mỗi bước là tuple (kind, idx, step):
#   ("K_m", m, ±1)              : mua thêm / bớt 1 tester loại m           ->  3 x 2  =  6 bước
#   ("K_a", a, ±1)              : mua thêm / bớt 1 handler loại a          ->  4 x 2  =  8 bước
#   ("X_m", (p, m), ±1)         : thuê thêm / bớt 1 tester m ở quý p        ->  8x3x2  = 48 bước
#   ("X_a", (p, a), ±1)         : thuê thêm / bớt 1 handler a ở quý p       ->  8x4x2  = 64 bước
#   ("shift", (p,t,src,dst), .1): chuyển 10% sản lượng SP t quý p từ tester src sang dst
#                                  (chỉ giữa các tester làm được SP t)       ->        48 bước
#                                                                              Tổng = 174 bước
def _neighbor_moves():
    """Liệt kê toàn bộ 174 bước lân cận theo 3 kiểu đã chốt (xem bảng ở trên)."""
    moves = []
    for m in range(mdl.M):                           # (1) K_m ± 1
        moves += [("K_m", m, +1), ("K_m", m, -1)]
    for a in range(mdl.A):                           # (1) K_a ± 1
        moves += [("K_a", a, +1), ("K_a", a, -1)]
    for p in range(mdl.P):                           # (2) X ± 1 trong 1 quý
        for m in range(mdl.M):
            moves += [("X_m", (p, m), +1), ("X_m", (p, m), -1)]
        for a in range(mdl.A):
            moves += [("X_a", (p, a), +1), ("X_a", (p, a), -1)]
    for p in range(mdl.P):                           # (3) chuyển 10% sản lượng
        for t in range(mdl.T):
            ms = [m for m in range(mdl.M) if mdl.C_MT[m, t]]   # các tester làm được SP t
            for src in ms:
                for dst in ms:
                    if src != dst:
                        moves.append(("shift", (p, t, src, dst), LS_SHIFT))
    return moves


MOVES = _neighbor_moves()


def apply_move(d, move):
    """Áp 1 bước lân cận lên phương án đã giải mã d (dict số máy + sản lượng).
    Trả về dict MỚI (không sửa d), hoặc None nếu bước vô nghĩa / vượt giới hạn,
    ví dụ bớt máy khi đã về mức ban đầu, thuê quá X_MAX, hoặc chuyển sản lượng khi tester nguồn = 0."""
    kind, idx, step = move
    d = {k: np.array(v, copy=True) for k, v in d.items()}        # sao chép để không sửa bản gốc
    if kind == "K_m":                                           # số tester sở hữu
        new = d["K_m"][idx] + step
        if not (mdl.K0_M[idx] <= new <= mdl.K0_M[idx] + mdl.K_EXTRA_M):
            return None                                         # không bán dưới mức ban đầu
        d["K_m"][idx] = new
    elif kind == "K_a":                                         # số handler sở hữu
        new = d["K_a"][idx] + step
        if not (mdl.K0_A[idx] <= new <= mdl.K0_A[idx] + mdl.K_EXTRA_A):
            return None
        d["K_a"][idx] = new
    elif kind == "X_m":                                         # số tester thuê ở 1 quý
        new = d["X_m"][idx] + step
        if not (0 <= new <= mdl.X_MAX_M):
            return None
        d["X_m"][idx] = new
    elif kind == "X_a":                                         # số handler thuê ở 1 quý
        new = d["X_a"][idx] + step
        if not (0 <= new <= mdl.X_MAX_A):
            return None
        d["X_a"][idx] = new
    else:                                                       # shift: chuyển sản lượng
        p, t, src, dst = idx
        amount = step * d["Q"][p, src, t]                       # 10% lượng tester nguồn đang làm
        if amount <= 0:
            return None
        d["Q"][p, src, t] -= amount
        d["Q"][p, dst, t] = min(d["Q"][p, dst, t] + amount, mdl.Q_MAX[t])
    return d


def reverse_move(move):
    """Bước ngược của 1 bước (dùng cho danh sách tabu):
       K/X: đổi dấu (+1 <-> -1);  shift (p,t,src,dst) -> (p,t,dst,src)."""
    kind, idx, step = move
    if kind == "shift":
        p, t, src, dst = idx
        return (kind, (p, t, dst, src), step)
    return (kind, idx, -step)


def _sample_valid(d, moves, rng, n):
    """Bốc ngẫu nhiên tối đa n bước HỢP LỆ khác nhau từ danh sách moves.
    Trả về list (move, dict sau khi áp). Duyệt theo hoán vị ngẫu nhiên, bỏ qua bước trả về None."""
    out = []
    for i in rng.permutation(len(moves)):
        if len(out) >= n:
            break
        nd = apply_move(d, moves[i])
        if nd is not None:
            out.append((moves[i], nd))
    return out


def _step(kind, move, f_from, f_to, **extra):
    """Ghi 1 bước vào trace (để demo/log hiển thị). kind: better / worse / equal / aspiration / blocked."""
    return dict(kind=kind, move=move, f_from=f_from, f_to=f_to, **extra)


def _direction(fy, fx):
    """Phân loại bước: Z mới so với Z cũ là tốt hơn, tệ hơn hay bằng."""
    return "worse" if fy < fx else ("better" if fy > fx else "equal")


# ============================================================ TÌM KIẾM CỤC BỘ
def local_search(x, fx, f_eval, rng, max_evals=LS_MAX_EVALS, moves=None, trace=None):
    """Hill climbing (first improvement) - bản Memetic cũ, CHỈ dùng cho ablation.

    Mã giả:
        lặp khi còn cải thiện và còn ngân sách:
            duyệt các bước theo thứ tự ngẫu nhiên:
                y = áp bước lên x; tính Z(y)
                nếu Z(y) > Z(x): x = y; bắt đầu lại vòng duyệt   (nhận ngay, không xem tiếp)
        trả về x
    Nhược điểm: chỉ nhận bước tốt hơn -> dừng ngay ở cực trị địa phương đầu tiên gặp."""
    moves = MOVES if moves is None else moves
    used = 0
    improved = True
    while improved and used < max_evals and f_eval.left() > 0:
        improved = False
        d = mdl.decode(x)
        for i in rng.permutation(len(moves)):
            if used >= max_evals or f_eval.left() <= 0:
                break
            nd = apply_move(d, moves[i])
            if nd is None:
                continue
            y = mdl.repair(mdl.encode(nd))       # mã hóa lại thành gen và sửa cho khả thi
            fy = f_eval(y)
            used += 1
            if fy > fx:                              # first improvement
                if trace is not None:
                    trace.append(_step("better", moves[i], fx, fy))
                x, fx, improved = y, fy, True
                break
    return x, fx, used


class TabuList:
    """Danh sách tabu (bộ nhớ ngắn hạn của Tabu Search).

    Mỗi lần đi 1 bước, BƯỚC NGƯỢC của nó bị cấm trong `tenure` vòng lặp,
    ví dụ vừa "bớt 1 tester M2 thuê ở quý 4" thì cấm "thêm 1 tester M2 thuê ở quý 4".
    Mục đích: không quay lại nghiệm vừa rời đi -> không đi vòng tròn, buộc khám phá chỗ mới.

    Cài đặt: until[move] = số vòng lặp mà tới đó move còn bị cấm; it = đồng hồ vòng lặp hiện tại."""

    def __init__(self, tenure=TS_TENURE):
        self.tenure = tenure
        self.it = 0
        self.until = {}

    def tick(self):
        """Sang vòng lặp mới (đồng hồ +1)."""
        self.it += 1

    def forbid(self, move):
        """Vừa đi `move` -> cấm bước ngược của nó thêm `tenure` vòng."""
        self.until[reverse_move(move)] = self.it + self.tenure

    def is_tabu(self, move):
        """move còn bị cấm ở vòng hiện tại không?"""
        return self.until.get(move, -1) >= self.it


def tabu_search(x, fx, f_eval, rng, max_evals=LS_MAX_EVALS, moves=None, tabu=None, trace=None,
                n_cand=TS_N_CAND):
    """Tabu Search có lấy mẫu lân cận (sampled neighbourhood).

    Mã giả:
        best = x
        lặp khi còn ngân sách (tối đa 20 lần evaluate):
            đồng hồ tabu += 1
            lặp tối đa 1 + 3 lần:
                bốc ngẫu nhiên 5 bước hợp lệ, tính Z của cả 5
                được phép = các bước KHÔNG bị cấm
                          + các bước bị cấm nhưng Z vượt kỷ lục toàn cục (aspiration)
                nếu có bước được phép: chọn bước có Z lớn nhất; thoát
            nếu không chọn được: dừng
            đi tới bước đã chọn (KỂ CẢ khi Z giảm)    <- khác hill climbing: thoát được cực trị địa phương
            cấm bước ngược của nó trong 7 vòng
            nếu Z(x) > Z(best): best = x
        trả về best                                     <- Lamarckian ghi best, nên không bao giờ làm tệ cá thể

    ĐIỂM MỚI (đóng góp chính): danh sách tabu dùng CHUNG cho 3 cá thể của cùng một thế hệ. Truyền
    cùng một đối tượng TabuList cho cả 3 lần gọi; đồng hồ vòng lặp chạy nối tiếp qua các cá thể và
    danh sách được tạo mới ở đầu mỗi lần kích hoạt tìm kiếm cục bộ. Nhờ vậy cá thể sau không đảo
    ngược những bước mà cá thể trước vừa đi. Tenure tính trên đồng hồ chung này.
    Lý do: log của bản Memetic cũ cho thấy 3 cá thể tốt nhất thường rất giống nhau và lặp lại
    cùng một chuỗi bước -> tabu chung buộc 3 lần mài đi 3 hướng khác nhau.

    Trả về (nghiệm tốt nhất gặp được, Z của nghiệm đó, số lần evaluate đã dùng)."""
    moves = MOVES if moves is None else moves
    tabu = TabuList() if tabu is None else tabu
    best_x, best_f = x, fx                      # nghiệm tốt nhất trong lần tìm kiếm này
    used = 0                                    # số lần evaluate đã dùng
    while used < max_evals and f_eval.left() > 0:
        tabu.tick()
        d = mdl.decode(x)                       # giải mã nghiệm hiện tại để áp bước
        chosen = None
        for _ in range(1 + TS_REDRAWS):         # bốc lần đầu + tối đa 3 lần bốc lại
            n = min(n_cand, max_evals - used, f_eval.left())   # không vượt ngân sách
            if n <= 0:
                break
            cands = _sample_valid(d, moves, rng, n)
            if not cands:
                break
            # phải chụp trước khi đánh giá, vì f_eval tự cập nhật best_f khi được gọi
            record = f_eval.best_f              # kỷ lục toàn cục (dùng cho aspiration)
            allowed = []                        # (Z, bước, nghiệm, có phải aspiration không)
            for move, nd in cands:
                y = mdl.repair(mdl.encode(nd))
                fy = f_eval(y)
                used += 1
                if not tabu.is_tabu(move):      # bước bình thường
                    allowed.append((fy, move, y, False))
                elif fy > record:               # bị cấm nhưng phá kỷ lục -> vẫn cho đi (aspiration)
                    allowed.append((fy, move, y, True))
                elif trace is not None:         # bị cấm và không đủ tốt -> ghi lại để log
                    trace.append(_step("blocked", move, fx, fy))
            if allowed:
                chosen = max(allowed, key=lambda c: c[0])   # best-of-candidates, kể cả khi tệ hơn x
                break
        if chosen is None:                      # bốc 4 lần đều bị cấm hết -> dừng
            break
        fy, move, y, aspired = chosen
        if trace is not None:
            trace.append(_step("aspiration" if aspired else _direction(fy, fx), move, fx, fy))
        tabu.forbid(move)                       # cấm bước ngược
        x, fx = y, fy                           # ĐI TỚI bước đã chọn
        if fx > best_f:                         # nhớ nghiệm tốt nhất
            best_x, best_f = x, fx
    return best_x, best_f, used


def estimate_t0(pop, fit, f_eval, rng, moves=None, p0=SA_P0, samples=SA_T0_SAMPLES):
    """Ước lượng nhiệt độ ban đầu T0 cho SA (1 lần ở đầu mỗi lần chạy GA).

    Mã giả:
        lặp 30 lần:
            chọn ngẫu nhiên 1 cá thể, áp 1 bước ngẫu nhiên, tính Z
            nếu Z giảm: ghi lại mức giảm |Δ|
        T0 = -median(|Δ|) / ln(p0)

    Ý nghĩa: với T0 này, exp(-median|Δ| / T0) = p0 = 0.5, tức 1 bước tệ "điển hình"
    được nhận với xác suất 50% lúc bắt đầu. Cách này tự co giãn theo thang đo của Z (hàng triệu)
    nên không phải chỉnh tay. 30 lần evaluate ở đây được f_eval đếm, nên tính vào ngân sách 6000."""
    moves = MOVES if moves is None else moves
    deltas = []
    for _ in range(min(samples, f_eval.left())):
        i = rng.integers(len(pop))
        cands = _sample_valid(mdl.decode(pop[i]), moves, rng, 1)
        if not cands:
            continue
        fy = f_eval(mdl.repair(mdl.encode(cands[0][1])))
        if fy < fit[i]:
            deltas.append(fit[i] - fy)          # chỉ quan tâm các bước làm tệ
    return -float(np.median(deltas)) / math.log(p0) if deltas else 1.0


def simulated_annealing(x, fx, f_eval, rng, max_evals=LS_MAX_EVALS, moves=None, T0=1.0, trace=None):
    """Simulated Annealing (luật Metropolis + làm nguội cấp số nhân).

    Mã giả:
        α = 0.01^(1/20)                  # sau 20 lần evaluate, T giảm còn 1% T0
        best = x
        lặp tối đa 20 lần:
            y = áp 1 bước ngẫu nhiên lên x; tính Z(y)
            T = T0 · α^i
            nếu Z(y) ≥ Z(x): nhận
            ngược lại: nhận với xác suất exp((Z(y) - Z(x)) / T)   # T cao -> dễ nhận bước tệ
            nếu nhận: x = y; nếu Z(x) > Z(best): best = x
        trả về best

    Khác Tabu: không có bộ nhớ, mỗi vòng chỉ xem 1 bước (thay vì 5), và việc nhận bước tệ là
    NGẪU NHIÊN theo nhiệt độ. Với chỉ 20 lần evaluate, lịch làm nguội rất ngắn -> đây là giả thuyết
    cần kiểm chứng: SA có thể không kịp phát huy so với Tabu."""
    moves = MOVES if moves is None else moves
    alpha = SA_FINAL_RATIO ** (1 / max_evals)   # hệ số làm nguội
    best_x, best_f = x, fx
    used = 0
    while used < max_evals and f_eval.left() > 0:
        cands = _sample_valid(mdl.decode(x), moves, rng, 1)   # 1 bước hợp lệ ngẫu nhiên
        if not cands:
            break
        move, nd = cands[0]
        y = mdl.repair(mdl.encode(nd))
        fy = f_eval(y)
        used += 1
        T = T0 * alpha ** used                  # nhiệt độ hiện tại
        if fy >= fx:                            # tốt hơn hoặc bằng: luôn nhận
            accept, p = True, 1.0
        else:                                   # tệ hơn: nhận với xác suất p
            p = math.exp((fy - fx) / T)
            accept = rng.random() < p
        if accept:
            if trace is not None:
                trace.append(_step(_direction(fy, fx), move, fx, fy, T=T, p=p))
            x, fx = y, fy
            if fx > best_f:
                best_x, best_f = x, fx
    return best_x, best_f, used


def _ls_event(i, rank, f_before, f_after, used, trace, **extra):
    """Đóng gói kết quả 1 lần tìm kiếm cục bộ (cho demo/log): số bước đã đi, số bước tệ hơn đã đi,
    số lần aspiration, số bước bị cấm, mức Z tăng thêm."""
    count = lambda kind: sum(s["kind"] == kind for s in trace)
    return dict(idx=int(i), rank=rank, f_before=f_before, f_after=f_after, used=used, steps=trace,
                n_steps=len(trace) - count("blocked"), n_worse=count("worse"),
                n_aspiration=count("aspiration"), n_blocked=count("blocked"),
                gain=f_after - f_before, **extra)


# ============================================================ VÒNG LẶP CHÍNH
def run_ga_iter(evaluate, ls_method="none", seed=0, budget=BUDGET,
                ls_freq=LS_FREQ, ls_top=LS_TOP, ls_max_evals=LS_MAX_EVALS,
                pop_size=POP_SIZE, pc=PC, pm=PM, tour=TOURNAMENT, elite=ELITE,
                ls_shift=LS_SHIFT, lamarck=True,
                ts_n_cand=TS_N_CAND, ts_tenure=TS_TENURE, ts_shared=True, sa_p0=SA_P0):
    """Vòng lặp GA dạng generator: trả về trạng thái sau MỖI thế hệ để demo vẽ trực tiếp.

    ls_method: "none" = SPGA gốc, "tabu" = GA-TS, "sa" = GA-SA, "hc" = hill climbing (ablation).
    Mỗi lần yield một dict: gen, pop, fit, evals, best_x, best_f, history, ls_events, ls_calls,
    runtime, done. Mỗi phần tử của ls_events có thêm: n_steps (số bước đã đi), n_worse (số bước tệ
    hơn đã đi/được nhận), n_aspiration, n_blocked, gain (mức Z tăng thêm).
    ts_shared=False: mỗi cá thể một danh sách tabu riêng (chỉ để làm ablation).
    lamarck=False: Baldwinian - chỉ cập nhật điểm, không ghi nghiệm (ablation).

    Mã giả: xem đầu file."""
    if ls_method not in LS_METHODS:
        raise ValueError(f"ls_method phải thuộc {LS_METHODS}")
    rng = np.random.default_rng(seed)            # cùng seed -> cùng kết quả (tái lập được)
    f_eval = _Counter(evaluate, budget)          # mọi lần tính Z đều qua bộ đếm này
    # nếu đổi % chuyển sản lượng thì tạo lại danh sách bước với tỉ lệ mới
    moves = MOVES if ls_shift == LS_SHIFT else [
        (k, i, ls_shift) if k == "shift" else (k, i, st) for (k, i, st) in MOVES]
    t0 = time.perf_counter()

    # ---- (1) Khởi tạo: 30 vector gen ngẫu nhiên, repair cho khả thi, chấm điểm
    pop = np.array([mdl.repair(rng.random(mdl.N_GENES)) for _ in range(pop_size)])
    fit = np.array([f_eval(x) for x in pop])
    # ---- (1b) Chỉ GA-SA: ước lượng T0 từ quần thể ban đầu (tốn 30 lần evaluate)
    sa_t0 = estimate_t0(pop, fit, f_eval, rng, moves, sa_p0) if ls_method == "sa" else None
    gen = 0
    ls_calls = 0                                 # tổng số lần evaluate dùng cho tìm kiếm cục bộ
    yield dict(gen=0, pop=pop, fit=fit, evals=f_eval.n, best_x=f_eval.best_x, best_f=f_eval.best_f,
               history=f_eval.history, ls_events=[], ls_calls=0, runtime=time.perf_counter() - t0, done=False)

    while f_eval.left() > 0:                     # ---- dừng khi hết 6000 lần evaluate
        gen += 1
        # ---- (2) Elitism: giữ nguyên cá thể tốt nhất sang thế hệ mới
        order = np.argsort(-fit)
        new_pop = [pop[i].copy() for i in order[:elite]]
        new_fit = [fit[i] for i in order[:elite]]
        # ---- (3) Sinh con cho đến khi đủ quần thể
        while len(new_pop) < pop_size and f_eval.left() > 0:
            p1 = pop[tournament(fit, rng, tour)]                 # chọn lọc
            p2 = pop[tournament(fit, rng, tour)]
            if rng.random() < pc:                                # lai ghép với xác suất pc
                c1, c2 = crossover(p1, p2, rng)
            else:
                c1, c2 = p1.copy(), p2.copy()
            for c in (c1, c2):
                if len(new_pop) >= pop_size or f_eval.left() <= 0:
                    break
                c = mdl.repair(mutate(c, rng, pm))               # đột biến + sửa khả thi
                new_pop.append(c)
                new_fit.append(f_eval(c))                        # chấm điểm (tốn 1 lần evaluate)
        pop, fit = np.array(new_pop), np.array(new_fit)

        # ---- (4) Tìm kiếm cục bộ: cứ ls_freq thế hệ, mài ls_top cá thể tốt nhất
        events = []
        if ls_method != "none" and gen % ls_freq == 0:
            tabu = TabuList(ts_tenure)                   # reset ở đầu mỗi lần kích hoạt
            for rank, i in enumerate(np.argsort(-fit)[:ls_top]):
                if f_eval.left() <= 0:
                    break
                trace = []
                f_before = fit[i]
                extra = {}
                if ls_method == "tabu":
                    if not ts_shared:                    # ablation: tabu riêng từng cá thể
                        tabu = TabuList(ts_tenure)
                    # mặc định: CÙNG đối tượng tabu cho cả 3 cá thể (tabu chung)
                    y, fy, used = tabu_search(pop[i], fit[i], f_eval, rng, ls_max_evals, moves, tabu, trace, ts_n_cand)
                elif ls_method == "sa":
                    y, fy, used = simulated_annealing(pop[i], fit[i], f_eval, rng, ls_max_evals, moves, sa_t0, trace)
                    extra["sa_t0"] = sa_t0
                else:
                    y, fy, used = local_search(pop[i], fit[i], f_eval, rng, ls_max_evals, moves, trace)
                ls_calls += used
                events.append(_ls_event(i, rank, f_before, fy, used, trace, **extra))
                if lamarck:
                    pop[i], fit[i] = y, fy           # Lamarckian: ghi nghiệm tốt nhất gặp được
                else:
                    fit[i] = fy                      # Baldwinian: chỉ cập nhật fitness
        yield dict(gen=gen, pop=pop, fit=fit, evals=f_eval.n, best_x=f_eval.best_x, best_f=f_eval.best_f,
                   history=f_eval.history, ls_events=events, ls_calls=ls_calls,
                   runtime=time.perf_counter() - t0, done=f_eval.left() <= 0)


def run_ga(evaluate, ls_method="none", seed=0, budget=BUDGET, **kw):
    """Chạy trọn 1 lần GA (không cần xem từng thế hệ), dùng trong main.py và tests.py.

    evaluate(x) -> (fitness, mean_profit, mad) (đã gắn sẵn bộ kịch bản).
    ls_method: "none" SPGA gốc, "tabu" GA-TS, "sa" GA-SA, "hc" hill climbing (ablation).
    Tham số khác (ls_freq, ts_n_cand, ...) truyền thẳng cho run_ga_iter.
    Sau khi chạy, run_ga.last_info chứa thống kê tổng: số thế hệ, số lần evaluate cho tìm kiếm
    cục bộ, tổng số bước đã đi / bước tệ hơn / aspiration / bị cấm, tổng mức Z tăng nhờ tìm kiếm cục bộ."""
    st = None
    totals = dict(ls_runs=0, ls_steps=0, ls_worse=0, ls_aspiration=0, ls_blocked=0, ls_gain=0.0)
    for st in run_ga_iter(evaluate, ls_method, seed, budget, **kw):
        for e in st["ls_events"]:                # cộng dồn thống kê tìm kiếm cục bộ
            totals["ls_runs"] += 1
            totals["ls_steps"] += e["n_steps"]
            totals["ls_worse"] += e["n_worse"]
            totals["ls_aspiration"] += e["n_aspiration"]
            totals["ls_blocked"] += e["n_blocked"]
            totals["ls_gain"] += e["gain"]
    run_ga.last_info = {"generations": st["gen"], "ls_evals": st["ls_calls"], "evals": st["evals"], **totals}
    return st["best_x"], st["history"], st["runtime"]
