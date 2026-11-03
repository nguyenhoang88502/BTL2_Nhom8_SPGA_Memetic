"""
exporter.py - Thống kê kiểm định, phân tích độ nhạy và xuất báo cáo Excel (1 sheet).

Dùng cho demo_server.py. Mọi import matplotlib / scipy / openpyxl đều nằm TRONG hàm
để server vẫn khởi động được khi thiếu thư viện và trả lỗi rõ ràng.

Hàm chính:
  compute_benchmark_stats(benchmark)            -> mean/std/median, Friedman, Wilcoxon, t-test, effect size
  generate_hypothesis_conclusion(stats, hyp)    -> văn bản kết luận kiểm định theo alpha
  compute_sensitivity_metrics / build_sensitivity_impact / generate_sensitivity_narrative
  overlay_distribution_png(runs)                -> PNG phân bố tần suất 3 thuật toán
  sensitivity_curves_png(sensitivity, metrics)  -> PNG đường cong độ nhạy theo k và σ
  export_single_tab_report(benchmark, sensitivity, hypothesis, path, milp=None)
                                                -> ghi file .xlsx DUY NHẤT 1 sheet "Bao_Cao_Kiem_Dinh"
"""
import io

import numpy as np

ALG_ORDER = ["SPGA gốc", "GA-TS", "GA-SA", "Matheuristics (GA-LP)"]
ALG_COLORS = {"SPGA gốc": "#1F3A93", "GA-TS": "#E07B39", "GA-SA": "#2E8B57",
              "Matheuristics (GA-LP)": "#7B2FBE"}


# ============================================================ thống kê
def _holm(pvalues):
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def _cohens_d(a, b):
    """Cohen's d cho mẫu cặp (paired): mean(diff) / std(diff)."""
    diff = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    sd = float(diff.std(ddof=1))
    return float(diff.mean() / sd) if sd > 0 else 0.0


def compute_benchmark_stats(benchmark):
    """Thống kê so sánh từ kết quả benchmark (dict fitness theo thuật toán).

    Trả về: per_alg, friedman, wilcoxon (kèm t-test và effect size)."""
    from scipy import stats

    algs = benchmark["algorithms"]
    base = benchmark["base"]
    F = benchmark["fitness"]
    per = {}
    for a in algs:
        f = np.array(F[a], dtype=float)
        per[a] = {
            "n": int(len(f)),
            "mean": float(f.mean()),
            "std": float(f.std(ddof=1)) if len(f) > 1 else 0.0,
            "median": float(np.median(f)),
            "min": float(f.min()),
            "max": float(f.max()),
        }
    fb = np.array(F[base], dtype=float)
    for a in algs:
        if a == base:
            per[a]["win_rate"] = None
            per[a]["improvement_pct"] = 0.0
        else:
            fa = np.array(F[a], dtype=float)
            per[a]["win_rate"] = float((fa > fb).sum()) / len(fa) if len(fa) else 0.0
            per[a]["improvement_pct"] = (fa.mean() / fb.mean() - 1) * 100 if fb.mean() != 0 else 0.0

    friedman = None
    if len(algs) >= 3 and all(len(F[a]) >= 2 for a in algs):
        fr = stats.friedmanchisquare(*[F[a] for a in algs])
        friedman = {"stat": float(fr.statistic), "p": float(fr.pvalue)}

    wilcoxon = []
    for a in algs:
        if a == base or len(F[a]) < 3:
            continue
        w = stats.wilcoxon(F[a], F[base], alternative="two-sided", zero_method="wilcox")
        t = stats.ttest_rel(F[a], F[base])
        wilcoxon.append({"a": a, "b": base, "stat": float(w.statistic), "p": float(w.pvalue),
                         "t": float(t.statistic), "t_p": float(t.pvalue),
                         "effect_size": _cohens_d(F[a], F[base])})
    if wilcoxon:
        adj = _holm([w["p"] for w in wilcoxon])
        for i, w in enumerate(wilcoxon):
            w["p_holm"] = adj[i]
    return {"per_alg": per, "friedman": friedman, "wilcoxon": wilcoxon}


def generate_hypothesis_conclusion(stats, hypothesis_info):
    """Sinh văn bản kết luận kiểm định dựa trên alpha và kết quả thống kê."""
    alpha = float(hypothesis_info.get("alpha", 0.05))
    h0 = hypothesis_info.get("h0", "")
    h1 = hypothesis_info.get("h1", "")
    lines = [f"Mức ý nghĩa α = {alpha:g}",
             f"H0: {h0}", f"H1: {h1}", ""]
    if stats.get("friedman"):
        p = stats["friedman"]["p"]
        verdict = "bác bỏ H0" if p < alpha else "không đủ bằng chứng để bác bỏ H0"
        lines.append(f"Friedman (so sánh đồng thời): chi2 = {stats['friedman']['stat']:.3f}, "
                     f"p = {p:.4f} → {verdict}.")
    for w in stats.get("wilcoxon", []):
        p = w.get("p_holm", w["p"])
        verdict = "bác bỏ H0 (có khác biệt có ý nghĩa)" if p < alpha else "không đủ bằng chứng để bác bỏ H0"
        es = f", effect size d = {w['effect_size']:.3f}" if "effect_size" in w else ""
        lines.append(f"Wilcoxon {w['a']} vs {w['b']}: p (Holm) = {p:.4f}{es} → {verdict}.")
        if "t" in w:
            lines.append(f"  t-test ghép cặp: t = {w['t']:.3f}, p = {w['t_p']:.4f}.")
    return "\n".join(lines)


# ============================================================ độ nhạy
def compute_sensitivity_metrics(sensitivity):
    algs = sensitivity["algorithms"]
    out = {}
    for a in algs:
        F = np.array(sensitivity["fitness"][a], dtype=float)
        mean_by_k = F.mean(axis=1)
        mean_by_sigma = F.mean(axis=0)
        total_var = float(F.var())
        contrib_k = float(mean_by_k.var()) / total_var if total_var > 0 else 0.0
        contrib_sigma = float(mean_by_sigma.var()) / total_var if total_var > 0 else 0.0
        robustness = float(F.mean() / F.std()) if F.std() > 0 else 0.0
        out[a] = {"mean_by_k": mean_by_k.tolist(), "mean_by_sigma": mean_by_sigma.tolist(),
                  "contribution_k": contrib_k, "contribution_sigma": contrib_sigma,
                  "robustness": robustness}
    return out


def build_sensitivity_impact(sensitivity):
    """Bảng tác động tham số: mỗi dòng = (tham số, giá trị, thuật toán, Z TB, LN TB, MAD, Δ%)."""
    algs = sensitivity["algorithms"]
    k_grid = sensitivity.get("k_grid", [])
    sigma_grid = sensitivity.get("sigma_grid", [])
    nk = sensitivity.get("nominal_k")
    ns = sensitivity.get("nominal_sigma")
    k_arr = np.array(k_grid, dtype=float)
    s_arr = np.array(sigma_grid, dtype=float)
    ki_nom = int(np.argmin(np.abs(k_arr - nk))) if nk is not None and len(k_arr) else 0
    si_nom = int(np.argmin(np.abs(s_arr - ns))) if ns is not None and len(s_arr) else 0
    rows = []
    for a in algs:
        F = np.array(sensitivity["fitness"][a], dtype=float)
        prof = sensitivity.get("profit", {}).get(a)
        mad = sensitivity.get("mad", {}).get(a)
        P = np.array(prof, dtype=float) if prof is not None else np.zeros_like(F)
        M = np.array(mad, dtype=float) if mad is not None else np.zeros_like(F)
        z_by_k, z_by_s = F.mean(axis=1), F.mean(axis=0)
        p_by_k, p_by_s = P.mean(axis=1), P.mean(axis=0)
        m_by_k, m_by_s = M.mean(axis=1), M.mean(axis=0)
        z_nom_k = z_by_k[ki_nom]
        z_nom_s = z_by_s[si_nom]
        for i, k in enumerate(k_grid):
            dz = (z_by_k[i] - z_nom_k) / abs(z_nom_k) * 100 if z_nom_k else 0.0
            rows.append({"param": "k", "value": float(k), "algorithm": a, "z_mean": float(z_by_k[i]),
                         "profit_mean": float(p_by_k[i]), "mad_mean": float(m_by_k[i]), "delta_pct": float(dz)})
        for j, s in enumerate(sigma_grid):
            dz = (z_by_s[j] - z_nom_s) / abs(z_nom_s) * 100 if z_nom_s else 0.0
            rows.append({"param": "sigma", "value": float(s), "algorithm": a, "z_mean": float(z_by_s[j]),
                         "profit_mean": float(p_by_s[j]), "mad_mean": float(m_by_s[j]), "delta_pct": float(dz)})
    return rows


def generate_sensitivity_narrative(sensitivity, metrics):
    algs = sensitivity["algorithms"]
    out = []
    for a in algs:
        m = metrics.get(a, {})
        ck, cs = m.get("contribution_k", 0), m.get("contribution_sigma", 0)
        dominant = "k" if ck >= cs else "σ"
        mbk = np.array(m.get("mean_by_k", []), dtype=float)
        mbs = np.array(m.get("mean_by_sigma", []), dtype=float)
        trend_k = "giảm" if len(mbk) > 1 and mbk[-1] < mbk[0] else ("tăng" if len(mbk) > 1 and mbk[-1] > mbk[0] else "gần như không đổi")
        trend_s = "giảm" if len(mbs) > 1 and mbs[-1] < mbs[0] else ("tăng" if len(mbs) > 1 and mbs[-1] > mbs[0] else "gần như không đổi")
        out.append(f"{a}: tham số chi phối là {dominant} (đóng góp k = {ck:.0%}, σ = {cs:.0%}). "
                   f"Khi k tăng, Z {trend_k}; khi σ tăng, Z {trend_s}. Độ ổn định (mean/std) = {m.get('robustness', 0):.2f}.")
    return "\n".join(out)


# ============================================================ vẽ biểu đồ
def _plt():
    """Trả về pyplot với backend Agg (an toàn khi vẽ trong thread của server)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    return plt


def _clean_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8bfcc")
    ax.spines["bottom"].set_color("#b8bfcc")
    ax.tick_params(colors="#5a6270")


def _fig_to_png(fig, dpi=140):
    plt = _plt()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def overlay_distribution_png(runs, bins=18):
    """Biểu đồ phân bố tần suất (Overlay Distribution) của 3 thuật toán."""
    plt = _plt()
    algs = runs["algorithms"]
    arrays = [np.array(runs["fitness"][a], dtype=float) / 1e6 for a in algs]
    lo, hi = min(a.min() for a in arrays), max(a.max() for a in arrays)
    pad = (hi - lo) * 0.05 or 0.5
    lo, hi = lo - pad, hi + pad
    edges = np.histogram_bin_edges(np.concatenate(arrays), bins=bins, range=(lo, hi))
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for a, arr in zip(algs, arrays):
        ax.hist(arr, bins=edges, density=True, histtype="stepfilled", alpha=0.26,
                color=ALG_COLORS[a], edgecolor=ALG_COLORS[a], linewidth=1.5, label=a)
        ax.axvline(arr.mean(), color=ALG_COLORS[a], ls="--", lw=1.4)
    ax.set_title("Phân bố Tần suất (Frequency Distribution)")
    ax.set_xlabel("Fitness Z (triệu VNĐ)")
    ax.set_ylabel("Tần suất (chuẩn hóa)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25, axis="y")
    _clean_style(ax)
    return _fig_to_png(fig)


def sensitivity_curves_png(sensitivity, metrics):
    """Biểu đồ Phân tích Độ nhạy: đường cong phản ứng Z theo k và theo σ."""
    plt = _plt()
    algs = sensitivity["algorithms"]
    k_grid = np.array(sensitivity.get("k_grid", []), dtype=float)
    s_grid = np.array(sensitivity.get("sigma_grid", []), dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for a in algs:
        m = metrics.get(a, {})
        mbk = np.array(m.get("mean_by_k", []) or [], dtype=float)
        mbs = np.array(m.get("mean_by_sigma", []) or [], dtype=float)
        if mbk.size:
            axes[0].plot(k_grid, mbk / 1e6, color=ALG_COLORS[a], lw=2, marker="o", ms=4, label=a)
        if mbs.size:
            axes[1].plot(s_grid, mbs / 1e6, color=ALG_COLORS[a], lw=2, marker="o", ms=4, label=a)
    axes[0].set_title("Z theo k (mức ngại rủi ro)")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel("Z trung bình (triệu VNĐ)")
    axes[1].set_title("Z theo σ (biến động nhu cầu)")
    axes[1].set_xlabel("σ")
    axes[1].set_ylabel("Z trung bình (triệu VNĐ)")
    for ax in axes:
        ax.legend(loc="best", frameon=False, fontsize=9)
        ax.grid(alpha=0.25)
        _clean_style(ax)
    fig.tight_layout()
    return _fig_to_png(fig)


# ============================================================ báo cáo Excel (1 sheet)
def _cell(ws, row, col, value, fmt=None, bold=False, fill=None, align=None, font_color=None):
    c = ws.cell(row=row, column=col, value=value)
    if fmt:
        c.number_format = fmt
    if bold or font_color:
        c.font = Font(bold=bold, color=font_color)
    if fill:
        c.fill = fill
    if align:
        c.alignment = Alignment(**align)
    return c


def export_single_tab_report(benchmark, sensitivity, hypothesis_info, path, milp=None):
    """Ghi báo cáo .xlsx DUY NHẤT 1 sheet 'Bao_Cao_Kiem_Dinh'.

    Bảng 1: thống kê (Z TB, Std, Min, Max, p-value Wilcoxon/t-test, kết luận H0/H1).
    Hình 1: Phân bố Tần suất (matplotlib PNG) + Hình 2: Đường cong Độ nhạy (matplotlib PNG),
    cả hai chèn trực tiếp vào sheet bằng openpyxl.drawing.image.Image."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    stats = compute_benchmark_stats(benchmark)
    algs = benchmark["algorithms"]
    params = benchmark.get("params", {})
    hypothesis_info = hypothesis_info or {}
    conclusion = generate_hypothesis_conclusion(stats, hypothesis_info)

    HEADER_FILL = PatternFill("solid", fgColor="1F3A93")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    ZEBRA_FILL = PatternFill("solid", fgColor="EEF2FB")
    THIN = Side(style="thin", color="D9DEE7")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    NUM = "#,##0.00"
    PCT = "0.00%"

    wb = Workbook()
    ws = wb.active
    ws.title = "Bao_Cao_Kiem_Dinh"

    def style_header(row, ncols):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=row, column=c)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = BORDER

    # ---- Khối 1: tiêu đề + giả thuyết + kết luận
    ws["A1"] = "Báo cáo Kiểm định Lặp — " + ", ".join(algs)
    ws["A1"].font = Font(bold=True, size=14, color="1F3A93")
    ws["A2"] = (f"Ngân sách {params.get('budget', '—')} lần đánh giá · k = {params.get('k', '—')} · "
                f"σ = {params.get('sigma', '—')} · {benchmark['n_runs']} lần lặp")
    ws["A2"].font = Font(italic=True, color="667085")
    if milp:
        ws["A3"] = (f"Cận tối ưu toàn cục (MILP): Z = {milp['z_opt'] / 1e6:,.3f} triệu VNĐ "
                    f"(status {milp['status']}, {milp['solve_time_s']:.1f}s)")
        ws["A3"].font = Font(bold=True, color="2E8B57")
    r = 5
    for line in conclusion.split("\n"):
        ws.cell(row=r, column=1, value=line)
        if line.startswith("H0:") or line.startswith("H1:"):
            ws.cell(row=r, column=1).font = Font(bold=True)
        r += 1
    r += 1

    # ---- Bảng 1: thống kê + p-value
    ws.cell(row=r, column=1, value="Bảng 1 — Thống kê & Kiểm định").font = Font(bold=True, color="1F3A93")
    r += 1
    headers = ["Chỉ số"] + algs
    for j, h in enumerate(headers, 1):
        ws.cell(row=r, column=j, value=h)
    style_header(r, len(headers))
    r += 1

    def add_row(label, getter, fmt=NUM):
        nonlocal r
        ws.cell(row=r, column=1, value=label).border = BORDER
        for j, a in enumerate(algs, 2):
            c = ws.cell(row=r, column=j, value=getter(a))
            c.number_format = fmt
            c.border = BORDER
        if r % 2 == 0:
            for j in range(1, len(headers) + 1):
                ws.cell(row=r, column=j).fill = ZEBRA_FILL
        r += 1

    def add_kv(label, value, fmt=None):
        nonlocal r
        ws.cell(row=r, column=1, value=label).border = BORDER
        c = ws.cell(row=r, column=2, value=value)
        if fmt:
            c.number_format = fmt
        c.border = BORDER
        r += 1

    add_row("Z trung bình (triệu)", lambda a: stats["per_alg"][a]["mean"] / 1e6)
    add_row("Độ lệch chuẩn (triệu)", lambda a: stats["per_alg"][a]["std"] / 1e6)
    add_row("Trung vị (triệu)", lambda a: stats["per_alg"][a]["median"] / 1e6)
    add_row("Nhỏ nhất (triệu)", lambda a: stats["per_alg"][a]["min"] / 1e6)
    add_row("Lớn nhất (triệu)", lambda a: stats["per_alg"][a]["max"] / 1e6)
    add_row("Win-rate so với gốc", lambda a: stats["per_alg"][a]["win_rate"], PCT)
    if stats["friedman"]:
        add_kv("Friedman p-value", stats["friedman"]["p"], "0.0000")
    for w in stats["wilcoxon"]:
        add_kv(f"Wilcoxon {w['a']} vs {w['b']} p (Holm)", w["p_holm"], "0.0000")
        add_kv(f"t-test {w['a']} vs {w['b']} p", w["t_p"], "0.0000")
        add_kv(f"Effect size d ({w['a']} vs {w['b']})", w["effect_size"], NUM)
    r += 1

    # ---- Top 3 + dữ liệu N lần lặp
    leaderboard = benchmark.get("leaderboard", [])
    if leaderboard:
        ws.cell(row=r, column=1, value="Top 3 lần chạy có Z cao nhất").font = Font(bold=True, color="1F3A93")
        r += 1
        for j, h in enumerate(["Hạng", "Seed", "Thuật toán", "Z (triệu)"], 1):
            ws.cell(row=r, column=j, value=h)
        style_header(r, 4)
        r += 1
        for rank, item in enumerate(leaderboard[:3], 1):
            ws.cell(row=r, column=1, value=rank).border = BORDER
            ws.cell(row=r, column=2, value=item["seed"]).border = BORDER
            ws.cell(row=r, column=3, value=item["algorithm"]).border = BORDER
            c = ws.cell(row=r, column=4, value=item["z"] / 1e6)
            c.number_format = NUM
            c.border = BORDER
            r += 1
    r += 1

    ws.cell(row=r, column=1, value="Dữ liệu chi tiết N lần lặp").font = Font(bold=True, color="1F3A93")
    r += 1
    hdr = ["Seed", "Thuật toán", "Z", "Lợi nhuận TB", "MAD", "Runtime (s)"]
    for j, h in enumerate(hdr, 1):
        ws.cell(row=r, column=j, value=h)
    style_header(r, len(hdr))
    r += 1
    seeds = benchmark.get("seeds", list(range(1, benchmark["n_runs"] + 1)))
    for i, s in enumerate(seeds):
        for a in algs:
            vals = [s, a, benchmark["fitness"][a][i],
                    benchmark.get("mean_profit", {}).get(a, [0] * len(seeds))[i],
                    benchmark.get("mad", {}).get(a, [0] * len(seeds))[i],
                    benchmark.get("runtime", {}).get(a, [0] * len(seeds))[i]]
            for j, v in enumerate(vals, 1):
                c = ws.cell(row=r, column=j, value=v)
                c.border = BORDER
                if j >= 3:
                    c.number_format = NUM if j == 3 else ("#,##0.00" if j < 6 else "0.00")
                if r % 2 == 0:
                    c.fill = ZEBRA_FILL
            r += 1
    r += 1

    # ---- bảng độ nhạy
    impact = build_sensitivity_impact(sensitivity) if sensitivity else []
    if impact:
        ws.cell(row=r, column=1, value="Phân tích độ nhạy (tác động tham số)").font = Font(bold=True, color="1F3A93")
        r += 1
        hdr = ["Tham số", "Giá trị", "Thuật toán", "Z TB (triệu)", "Lợi nhuận TB (triệu)", "MAD (triệu)", "Δ%"]
        for j, h in enumerate(hdr, 1):
            ws.cell(row=r, column=j, value=h)
        style_header(r, len(hdr))
        r += 1
        for row_ in impact:
            vals = [row_["param"], row_["value"], row_["algorithm"], row_["z_mean"] / 1e6,
                    row_["profit_mean"] / 1e6, row_["mad_mean"] / 1e6, row_["delta_pct"] / 100]
            for j, v in enumerate(vals, 1):
                c = ws.cell(row=r, column=j, value=v)
                c.border = BORDER
                if j >= 4:
                    c.number_format = PCT if j == 7 else NUM
                if r % 2 == 0:
                    c.fill = ZEBRA_FILL
            r += 1
    r += 2

    # ---- Hình 1 + Hình 2: chèn trực tiếp vào sheet
    img1 = XLImage(io.BytesIO(overlay_distribution_png(benchmark)))
    img1.width, img1.height = 740, 410
    ws.add_image(img1, f"A{r}")
    if sensitivity:
        img2 = XLImage(io.BytesIO(sensitivity_curves_png(sensitivity, compute_sensitivity_metrics(sensitivity))))
        img2.width, img2.height = 980, 400
        ws.add_image(img2, f"A{r + 26}")

    for j, w in enumerate([22, 16, 14, 14, 16, 12], 1):
        ws.column_dimensions[get_column_letter(j)].width = w

    wb.save(path)
    return path
