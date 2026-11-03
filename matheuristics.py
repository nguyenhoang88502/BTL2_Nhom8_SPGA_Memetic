"""
matheuristics.py - Matheuristics (GA-LP): GA trên không gian máy móc + LP tối ưu sản lượng.

CHIẾN LƯỢC 1 (master-subproblem):
  Master (GA)     : cá thể chỉ gồm 63 gen: mua tester K_m (3), mua handler K_a (4),
                    thuê tester theo quý X_m (24), thuê handler theo quý X_a (32).
                    KHÔNG chứa gen sản lượng -> không gian tìm kiếm gọn hơn nhiều.
  Subproblem (LP) : với phương án máy CỐ ĐỊNH, giải LP phân bổ sản lượng Q*[p,m,t]
                    tối ưu tuyệt đối, trả về Z = (1-k)·E[h] − k·MAD chính xác.

SOLVER LP: scipy.optimize.linprog (HiGHS) thay vì PuLP/CBC cho subproblem, vì GA cần
giải hàng nghìn LP liên tiếp: HiGHS giải trong bộ nhớ (~5-15 ms/LP) còn PuLP phải ghi
file .mps/.lp tạm cho CBC mỗi lần giải (~50-100 ms/LP). Mô hình LP được build MỘT LẦN,
mỗi lần đánh giá chỉ cập nhật vế phải ràng buộc năng lực (K + X) rồi giải lại.

LP (đơn vị nội bộ: tiền triệu, sản lượng nghìn đơn vị):
  Biến:  Q[p,m,t] ≥ 0 (72) · Sp[n,p], Sm[n,p] ≥ 0 (tồn kho dư/thiếu SP1) ·
         sold[n,p,t] ≥ 0 (bán MTO SP2/SP3) · u[n] ≥ 0 (độ lệch cho MAD)
  Lợi nhuận giản ước về hiện giá (KHÔNG cần biến dòng tiền F):
      h_n = F0 + Σ_p β_p·(income − V − out_cost) − invest ,  β_p = (1+r)^(−p−1)
  MAD tuyến tính hóa: u[n] ≥ h_n − mean(h), u[n] ≥ mean(h) − h_n (đúng khi k > 0).
  Mục tiêu: max (1−k)·Σ h_n − k·Σ u_n.

CHẤM ĐIỂM (điểm quan trọng để so sánh công bằng): LP cho phép "bán dưới nhu cầu"
(sold ≤ nhu cầu) — một mức linh hoạt theo kịch bản mà model.py (bán hết min(prod,
demand)) không có, và với k ≥ 0.5 LP có thể dùng nó để giảm MAD. Vì vậy fitness
KHÔNG lấy từ hàm mục tiêu LP mà được tính LẠI bằng model.scenario_profits trên
phương án "sạch" (MTO sản xuất đúng mức bán lớn nhất qua các kịch bản) -> khớp
tuyệt đối với cách chấm điểm của SPGA/GA-TS/GA-SA.

Hàm chính:
  LpEvaluator(k, sigma, scenarios).evaluate(x63) -> (fitness, mean, mad)  [VNĐ]
  LpEvaluator.plan_dict(x63)                 -> dict máy + sản lượng tối ưu (đơn vị gốc)
  decode_ml / encode_ml / repair_ml          -> mã hóa 63 gen máy móc
  run_matheur(evaluate, seed, budget)        -> chạy GA-LP trọn vẹn (qua ga.run_ga)
"""
import numpy as np
from scipy.optimize import linprog
from scipy.sparse import coo_matrix

import ga
import model as mdl

N_GENES_ML = mdl.M + mdl.A + mdl.P * mdl.M + mdl.P * mdl.A          # 3+4+24+32 = 63
SL_KM = slice(0, 3)
SL_KA = slice(3, 7)
SL_XM = slice(7, 31)
SL_XA = slice(31, 63)


def decode_ml(x):
    """63 gen [0,1] -> dict số máy nguyên (K_m, K_a, X_m, X_a). Không có sản lượng."""
    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)
    return {"K_m": mdl.K0_M + np.rint(x[SL_KM] * mdl.K_EXTRA_M).astype(int),
            "K_a": mdl.K0_A + np.rint(x[SL_KA] * mdl.K_EXTRA_A).astype(int),
            "X_m": np.rint(x[SL_XM].reshape(mdl.P, mdl.M) * mdl.X_MAX_M).astype(int),
            "X_a": np.rint(x[SL_XA].reshape(mdl.P, mdl.A) * mdl.X_MAX_A).astype(int)}


def encode_ml(d):
    """dict máy -> 63 gen [0,1] (ngược của decode_ml)."""
    x = np.empty(N_GENES_ML)
    x[SL_KM] = (np.asarray(d["K_m"]) - mdl.K0_M) / mdl.K_EXTRA_M
    x[SL_KA] = (np.asarray(d["K_a"]) - mdl.K0_A) / mdl.K_EXTRA_A
    x[SL_XM] = (np.asarray(d["X_m"]) / mdl.X_MAX_M).ravel()
    x[SL_XA] = (np.asarray(d["X_a"]) / mdl.X_MAX_A).ravel()
    return np.clip(x, 0.0, 1.0)


def repair_ml(x):
    """Mọi gen trong [0,1] đều là phương án máy hợp lệ -> chỉ cần clip."""
    return np.clip(np.asarray(x, dtype=float), 0.0, 1.0)


def run_matheur(evaluate, seed=0, budget=None, **kw):
    """Chạy trọn 1 lần GA-LP qua khung GA chung (không tìm kiếm cục bộ).

    evaluate = LpEvaluator.evaluate (63 gen -> (fitness, mean, mad)).
    Trả về (best_x, history, runtime) giống ga.run_ga."""
    return ga.run_ga(evaluate, "none", seed,
                     ga.BUDGET if budget is None else int(budget),
                     repair_fn=repair_ml, n_genes=N_GENES_ML, **kw)


class LpEvaluator:
    """Fitness chính xác của phương án máy: giải LP phân bổ sản lượng bằng HiGHS."""

    def __init__(self, k=None, sigma=None, scenarios=None, n_scen=30, scen_seed=2026):
        self.k = mdl.K_RISK if k is None else float(k)
        self.sigma = mdl.SIGMA if sigma is None else float(sigma)
        if scenarios is None:
            scenarios = mdl.sample_scenarios(n_scen, scen_seed, sigma=self.sigma)
        self.scen_orig = np.asarray(scenarios, dtype=float)          # đơn vị gốc
        self.scen = self.scen_orig / 1000.0                          # nghìn đơn vị
        self.N = self.scen.shape[0]
        self._last_q = None
        self._build()

    # ------------------------------------------------ build LP một lần
    def _build(self):
        P, M, A, T = mdl.P, mdl.M, mdl.A, mdl.T
        N = self.N
        rate = float(mdl.I_RATE)
        beta = [(1.0 + rate) ** -(p + 1) for p in range(P)]          # hệ số hiện giá
        Bc = [float(v) * 1000.0 / 1e6 for v in mdl.B]                # lợi nhuận/1k đv (triệu)
        JH = float(mdl.J_HOLD[0]) * 1000.0 / 1e6
        LB = float(mdl.L_BACK[0]) * 1000.0 / 1e6
        CAPMc = [[float(v) / 1000.0 for v in row] for row in mdl.CAP_M]
        CAPHc = [float(v) / 1000.0 for v in mdl.CAP_H]
        QMAXc = [float(v) / 1000.0 for v in mdl.Q_MAX]
        self._beta, self._Bc, self._JH, self._LB = beta, Bc, JH, LB
        # phần không phụ thuộc biến: doanh thu bán SP1 (MTS, bán hết nhu cầu)
        const = np.array([sum(beta[p] * Bc[0] * self.scen[n, p, 0] for p in range(P))
                          for n in range(N)])

        # ---- chỉ số biến (+1 biến mean(lin) để ràng buộc MAD thưa)
        nQ = P * M * T
        off_Sp, off_Sm = nQ, nQ + N * P
        off_Sold = off_Sm + N * P
        off_U = off_Sold + N * P * 2
        mean_idx = off_U + N
        nvar = mean_idx + 1
        self._nQ = nQ

        def qi(p, m, t): return p * M * T + m * T + t
        def spi(n, p): return off_Sp + n * P + p
        def smi(n, p): return off_Sm + n * P + p
        def soi(n, p, j): return off_Sold + (n * P + p) * 2 + j
        def ui(n): return off_U + n
        self._soi = soi

        # ---- bounds + mục tiêu (min c^T x = −(1−k)Σlin + kΣu)
        c = np.zeros(nvar)
        bnd = [(0.0, None)] * nvar
        bnd[mean_idx] = (None, None)                    # mean(lin) tự do
        for p in range(P):
            for m in range(M):
                for t in range(T):
                    i = qi(p, m, t)
                    bnd[i] = (0.0, QMAXc[t]) if mdl.C_MT[m, t] else (0.0, 0.0)
                    c[i] = 1e-9        # đẩy sản lượng dư thừa về 0 khi lợi ích hòa
        for n in range(N):
            for p in range(P):
                c[spi(n, p)] = (1.0 - self.k) * JH * beta[p]
                c[smi(n, p)] = (1.0 - self.k) * LB * beta[p]
                c[soi(n, p, 0)] = -(1.0 - self.k) * Bc[1] * beta[p] - 1e-9
                c[soi(n, p, 1)] = -(1.0 - self.k) * Bc[2] * beta[p] - 1e-9
        for n in range(N):
            c[ui(n)] = self.k

        # ---- ràng buộc (sparse COO -> CSC)
        rows, cols, vals = [], [], []
        n_cap = P * M + P * A
        n_sold = N * P * 2
        n_dem = n_sold
        n_mad = 2 * N
        n_ub = n_cap + n_sold + n_dem + n_mad
        bub = np.zeros(n_ub)

        self._cap_m_rows, self._cap_a_rows = [], []
        for p in range(P):                                   # (R1) năng lực tester
            for m in range(M):
                self._cap_m_rows.append(p * M + m)
                for t in range(T):
                    if mdl.C_MT[m, t]:
                        rows.append(p * M + m)
                        cols.append(qi(p, m, t))
                        vals.append(1.0 / CAPMc[m][t])
        for p in range(P):                                   # (R2) năng lực handler
            for a in range(A):
                self._cap_a_rows.append(P * M + p * A + a)
                for m in range(M):
                    for t in range(T):
                        if mdl.C_MT[m, t] and mdl.H_OF[m, t] == a:
                            rows.append(P * M + p * A + a)
                            cols.append(qi(p, m, t))
                            vals.append(1.0 / CAPHc[a])
        for n in range(N):                                   # (R4) bán MTO <= sản lượng
            for p in range(P):
                for j, t in enumerate((1, 2)):
                    r = n_cap + (n * P + p) * 2 + j
                    rows.append(r); cols.append(soi(n, p, j)); vals.append(1.0)
                    for m in range(M):
                        if mdl.C_MT[m, t]:
                            rows.append(r); cols.append(qi(p, m, t)); vals.append(-1.0)
        for n in range(N):                                   # (R5) bán MTO <= nhu cầu
            for p in range(P):
                for j, t in enumerate((1, 2)):
                    r = n_cap + n_sold + (n * P + p) * 2 + j
                    rows.append(r); cols.append(soi(n, p, j)); vals.append(1.0)
                    bub[r] = float(self.scen[n, p, t])
        for n in range(N):                                   # (R6) MAD với biến mean (thưa)
            an = float(const[n] - const.mean())
            for s_idx, sgn in enumerate((1.0, -1.0)):
                r = n_cap + n_sold + n_dem + 2 * n + s_idx
                rows.append(r); cols.append(ui(n)); vals.append(-1.0)
                rows.append(r); cols.append(mean_idx); vals.append(-sgn)
                for p in range(P):
                    rows.append(r); cols.append(spi(n, p)); vals.append(sgn * (-JH * beta[p]))
                    rows.append(r); cols.append(smi(n, p)); vals.append(sgn * (-LB * beta[p]))
                    rows.append(r); cols.append(soi(n, p, 0)); vals.append(sgn * Bc[1] * beta[p])
                    rows.append(r); cols.append(soi(n, p, 1)); vals.append(sgn * Bc[2] * beta[p])
                bub[r] = -sgn * an

        # (R3) cân bằng tồn kho SP1 (đẳng thức): Sp − Sm − Σ_{q≤p} prod(q,0) = −Σ scen
        # (R3b) đẳng thức mean: mean(lin) − (1/N)·Σ lin_n = 0
        erows, ecols, evals, beq = [], [], [], []
        for n in range(N):
            for p in range(P):
                r = n * P + p
                erows.append(r); ecols.append(spi(n, p)); evals.append(1.0)
                erows.append(r); ecols.append(smi(n, p)); evals.append(-1.0)
                for q in range(p + 1):
                    for m in range(M):
                        if mdl.C_MT[m, 0]:
                            erows.append(r); ecols.append(qi(q, m, 0)); evals.append(-1.0)
                beq.append(-float(self.scen[n, :p + 1, 0].sum()))
        erows.append(N * P); ecols.append(mean_idx); evals.append(1.0)
        for n in range(N):
            for p in range(P):
                erows.append(N * P); ecols.append(spi(n, p)); evals.append((-1.0 / N) * (-JH * beta[p]))
                erows.append(N * P); ecols.append(smi(n, p)); evals.append((-1.0 / N) * (-LB * beta[p]))
                erows.append(N * P); ecols.append(soi(n, p, 0)); evals.append((-1.0 / N) * Bc[1] * beta[p])
                erows.append(N * P); ecols.append(soi(n, p, 1)); evals.append((-1.0 / N) * Bc[2] * beta[p])
        beq.append(0.0)

        self._c = c
        self._bnd = bnd
        self._bub = bub
        self._Aub = coo_matrix((vals, (rows, cols)), shape=(n_ub, nvar)).tocsc()
        self._Aeq = coo_matrix((evals, (erows, ecols)), shape=(N * P + 1, nvar)).tocsc()
        self._beq = np.array(beq)

    # ------------------------------------------------ giải LP với RHS mới
    def _solve_lp(self, b):
        """Giải LP với vế phải b. Thử HiGHS trước, lùi về interior-point (dense)."""
        last = None
        for meth in ("highs", "highs-ds", "highs-ipm"):
            try:
                res = linprog(self._c, A_ub=self._Aub, b_ub=b, A_eq=self._Aeq,
                              b_eq=self._beq, bounds=self._bnd, method=meth)
                if res.status == 0:
                    return res
                last = res
            except Exception:
                continue
        try:
            res = linprog(self._c, A_ub=self._Aub.toarray(), b_ub=b,
                          A_eq=self._Aeq.toarray(), b_eq=self._beq,
                          bounds=self._bnd, method="interior-point")
            if res.status == 0:
                return res
            last = res
        except Exception:
            pass
        return last

    def _fallback(self, d):
        """Phương án Q=0 (luôn khả thi) khi LP gặp sự cố số học."""
        d0 = dict(d, Q=np.zeros((mdl.P, mdl.M, mdl.T)))
        h = mdl.scenario_profits(mdl.encode(d0), self.scen_orig)
        mean = float(h.mean())
        mad = float(np.abs(h - mean).mean())
        return (1.0 - self.k) * mean - self.k * mad, mean, mad

    # ------------------------------------------------ đánh giá 1 phương án máy
    def evaluate(self, x):
        """x (63 gen) -> (fitness Z, mean_profit, mad) [VNĐ]. 1 lần gọi = 1 lần giải LP.

        Fitness tính lại bằng mô hình dùng chung (bán hết nhu cầu MTO) trên phương án
        sạch để khớp tuyệt đối với cách chấm điểm của SPGA/GA-TS/GA-SA."""
        P, M, A, T = mdl.P, mdl.M, mdl.A, mdl.T
        d = decode_ml(x)
        Km, Ka = d["K_m"], d["K_a"]
        Xm, Xa = d["X_m"], d["X_a"]
        b = self._bub.copy()
        for p in range(P):                                   # cập nhật RHS năng lực
            for m in range(M):
                b[self._cap_m_rows[p * M + m]] = Km[m] + Xm[p, m]
            for a in range(A):
                b[self._cap_a_rows[p * A + a]] = Ka[a] + Xa[p, a]
        res = self._solve_lp(b)
        if res is None or res.status != 0:
            return self._fallback(d)
        xv = res.x
        Q = np.clip(xv[:self._nQ], 0.0, None).reshape(P, M, T) * 1000.0   # đơn vị gốc
        # phương án sạch: MTO sản xuất đúng mức bán lớn nhất qua các kịch bản
        # (loại bỏ quyền "sản xuất nhưng không bán" mà LP được phép dùng để giảm MAD)
        sold_k = np.zeros((P, 2))
        for p in range(P):
            for j in range(2):
                sold_k[p, j] = max(xv[self._soi(n, p, j)] for n in range(self.N)) * 1000.0
        prod_k = Q.sum(axis=1)                                # [P, T] đơn vị gốc
        Qc = Q.copy()
        for t in (1, 2):
            j = t - 1
            for p in range(P):
                if prod_k[p, t] > 1e-9 and sold_k[p, j] < prod_k[p, t]:
                    Qc[p, :, t] *= sold_k[p, j] / prod_k[p, t]
        self._last_q = Qc
        d0 = dict(d, Q=Qc)
        h = mdl.scenario_profits(mdl.encode(d0), self.scen_orig)
        mean = float(h.mean())
        mad = float(np.abs(h - mean).mean())
        z = (1.0 - self.k) * mean - self.k * mad
        return z, mean, mad

    def plan_dict(self, x):
        """dict phương án máy + sản lượng tối ưu (đơn vị gốc) cho phương án x.
        Gọi evaluate để lấy Q* của đúng phương án này."""
        d = decode_ml(x)
        self.evaluate(x)
        Q = self._last_q if self._last_q is not None else np.zeros((mdl.P, mdl.M, mdl.T))
        return dict(d, Q=Q)
