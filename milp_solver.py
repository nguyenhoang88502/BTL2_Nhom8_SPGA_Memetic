"""
milp_solver.py - Mô hình MILP (pulp + CBC) cho bài toán hoạch định danh mục nguồn lực.

Dùng làm CẬN TỐI ƯU TOÀN CỤC (Global Optimum Benchmark Bound) để đối chiếu với GA:
cùng bộ kịch bản nhu cầu (seed 2026), cùng mục tiêu Z = (1-k)·mean(h) − k·MAD(h).

Tuyến tính hóa:
  MAD        : u_n >= h_n − mean, u_n >= mean − h_n  ->  MAD = (1/N)·Σ u_n (đúng khi k > 0)
  Tồn kho MTS: S = S⁺ − S⁻, chi phí = J_HOLD·S⁺ + L_BACK·S⁻
  Bán MTO    : sold <= sản lượng, sold <= nhu cầu (bán tối đa bằng nhu cầu)
  Vốn        : F_p = (1+r)·F_{p-1} + doanh thu − chi phí thuê − chi phí tồn kho/thiếu
  Năng lực   : Σ Q/CAP <= máy sở hữu + máy thuê (tester theo từng loại, handler theo ghép cặp)

ĐƠN VỊ NỘI BỘ (để hệ số nhỏ, CBC giải nhanh và ổn định):
  tiền: triệu VNĐ · sản lượng/nhu cầu: nghìn đơn vị
Kết quả trả về đổi ngược về đơn vị gốc (VNĐ) để so sánh trực tiếp với GA.

Hàm chính:
  solve_milp_exact(k, sigma, n_scen, scen_seed, time_limit) -> dict
      {"z_opt", "mean_opt", "mad_opt", "status", "solve_time_s", "solution", ...}
"""
import time

import numpy as np

import model as mdl


def _get_solver():
    """Trả về CBC nếu có (pulp <= 2.9 đi kèm CBC), ngược lại solver mặc định."""
    import pulp
    if hasattr(pulp, "PULP_CBC_CMD"):
        return pulp.PULP_CBC_CMD
    return pulp.LpSolverDefault


def solve_milp_exact(k=None, sigma=None, n_scen=30, scen_seed=2026, time_limit=120, msg=0, mu=None):
    """Giải MILP trả về nghiệm tối ưu toàn cục (hoặc tốt nhất tìm được khi hết giờ).

    mu tùy chọn: snapshot nhu cầu trung bình để chạy nền song song an toàn với các
    lần chạy GA (không đọc/ghi global MU/SIGMA/Q_MAX trong lúc giải)."""
    import pulp

    k = mdl.K_RISK if k is None else float(k)
    sigma = mdl.SIGMA if sigma is None else float(sigma)
    mu_arr = mdl.MU if mu is None else np.asarray(mu, dtype=float)

    scen = mdl.sample_scenarios(n_scen, scen_seed, mu=mu_arr, sigma=sigma) / 1000.0   # CÙNG kịch bản với GA, đơn vị nghìn

    P, M, A, T = mdl.P, mdl.M, mdl.A, mdl.T
    # ---- hằng số theo đơn vị nội bộ (tiền: triệu; sản lượng: nghìn đơn vị)
    F0, RATE = float(mdl.F0) / 1e6, float(mdl.I_RATE)
    JH, LB = float(mdl.J_HOLD[0]) * 1000 / 1e6, float(mdl.L_BACK[0]) * 1000 / 1e6
    Bc = [float(v) * 1000 / 1e6 for v in mdl.B]
    UMc = [float(v) / 1e6 for v in mdl.U_M]
    UAc = [float(v) / 1e6 for v in mdl.U_A]
    EMc = [float(v) / 1e6 for v in mdl.E_M]
    DMc = [float(v) / 1e6 for v in mdl.D_M]
    EAc = [float(v) / 1e6 for v in mdl.E_A]
    DAc = [float(v) / 1e6 for v in mdl.D_A]
    CAPMc = [[float(v) / 1000.0 for v in row] for row in mdl.CAP_M]
    CAPHc = [float(v) / 1000.0 for v in mdl.CAP_H]
    QMAXc = [1.5 * float(v) / 1000.0 for v in mu_arr]
    K0Mc = [int(v) for v in mdl.K0_M]
    K0Ac = [int(v) for v in mdl.K0_A]
    KEXT_M, KEXT_A = int(mdl.K_EXTRA_M), int(mdl.K_EXTRA_A)
    XMAX_M, XMAX_A = int(mdl.X_MAX_M), int(mdl.X_MAX_A)

    prob = pulp.LpProblem("SPGA_MILP", pulp.LpMaximize)

    # ---- biến quyết định: máy mua (nguyên), máy thuê (nguyên), sản lượng (liên tục)
    Km = [pulp.LpVariable(f"Km_{m}", K0Mc[m], K0Mc[m] + KEXT_M, cat="Integer") for m in range(M)]
    Ka = [pulp.LpVariable(f"Ka_{a}", K0Ac[a], K0Ac[a] + KEXT_A, cat="Integer") for a in range(A)]
    Xm = [[pulp.LpVariable(f"Xm_{p}_{m}", 0, XMAX_M, cat="Integer") for m in range(M)] for p in range(P)]
    Xa = [[pulp.LpVariable(f"Xa_{p}_{a}", 0, XMAX_A, cat="Integer") for a in range(A)] for p in range(P)]
    Q = [[[pulp.LpVariable(f"Q_{p}_{m}_{t}", 0, QMAXc[t]) if mdl.C_MT[m, t] else None
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
                    if Q[p][m][t] is not None and mdl.H_OF[m, t] == a]
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
        # (9) lợi nhuận = giá trị hiện tại vốn cuối kỳ − đầu tư ròng
        h = F[P - 1] * (1.0 / (1 + RATE) ** P) - invest
        h_exprs.append(h)
        u.append(pulp.LpVariable(f"u_{n}", 0))

    # MAD tuyến tính hóa + mục tiêu max (1-k)·Σh − k·Σu  (tương đương Z·N)
    mean = pulp.lpSum(h_exprs) * (1.0 / n_scen)
    for n in range(n_scen):
        prob += u[n] >= h_exprs[n] - mean
        prob += u[n] >= mean - h_exprs[n]
    prob += (1 - k) * pulp.lpSum(h_exprs) - k * pulp.lpSum(u)

    t0 = time.perf_counter()
    solver = _get_solver()
    status = prob.solve(solver(msg=msg, timeLimit=time_limit))
    dt = time.perf_counter() - t0

    hv = [pulp.value(h_exprs[n]) * 1e6 for n in range(n_scen)]   # đổi ngược về VNĐ
    mean_opt = float(np.mean(hv))
    mad_opt = float(np.abs(np.array(hv) - mean_opt).mean())
    z_opt = (1 - k) * mean_opt - k * mad_opt

    sol = {
        "Km": [int(round(pulp.value(v))) for v in Km],
        "Ka": [int(round(pulp.value(v))) for v in Ka],
        "Xm": [[int(round(pulp.value(v))) for v in row] for row in Xm],
        "Xa": [[int(round(pulp.value(v))) for v in row] for row in Xa],
        "Q": [[[float(pulp.value(v)) * 1000.0 if v is not None else 0.0 for v in row] for row in mat] for mat in Q],
    }
    return {"z_opt": z_opt, "mean_opt": mean_opt, "mad_opt": mad_opt,
            "status": pulp.LpStatus[status], "solve_time_s": dt,
            "k": k, "sigma": sigma, "n_scen": n_scen, "scen_seed": scen_seed,
            "solution": sol}


if __name__ == "__main__":
    import json
    res = solve_milp_exact(time_limit=60)
    print(json.dumps({kk: res[kk] for kk in ("z_opt", "mean_opt", "mad_opt", "status", "solve_time_s")}, indent=2))
