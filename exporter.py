"""
exporter.py - Xuất báo cáo Excel (.xlsx) và biểu đồ phong cách Oracle Crystal Ball.

Dùng cho demo_server.py (kiểm định lặp, phân tích độ nhạy, xuất báo cáo). Không được
import ở mức module của demo_server nếu thư viện chưa cài -> mọi import matplotlib /
scipy / openpyxl đều nằm TRONG hàm để server vẫn khởi động được và trả lỗi rõ ràng.

Hàm chính:
  compute_benchmark_stats(benchmark)         -> thống kê (mean/std/median, Friedman, Wilcoxon, win-rate)
  compute_sensitivity_metrics(sensitivity)   -> mean theo k/sigma, mức đóng góp, độ ổn định
  overlay_distribution_png(runs)             -> PNG biểu đồ phủ phân bố tần suất
  cumulative_probability_png(runs)           -> PNG CDF (xác suất tích lũy)
  convergence_band_png(convergence)          -> PNG dải hội tụ (TB ± std, 10-90%)
  sensitivity_tornado_png(sensitivity)       -> PNG biểu đồ Tornado
  data_url(png)                              -> chuỗi data:image/png;base64 cho <img>
  build_excel_report(benchmark, sensitivity, path) -> ghi file .xlsx 4 tab
"""
import base64
import io

import numpy as np

ALG_ORDER = ["SPGA gốc", "GA-TS", "GA-SA"]
ALG_COLORS = {"SPGA gốc": "#1F3A93", "GA-TS": "#E07B39", "GA-SA": "#2E8B57"}
CB_GREEN = "#2E8B57"
CB_ORANGE = "#E07B39"
CB_BLUE = "#1F3A93"


def _plt():
    """Trả về pyplot với backend Agg (an toàn khi vẽ trong thread của server)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt
    return plt


# ============================================================ thống kê
def _holm(pvalues):
    """Hiệu chỉnh Holm cho một nhóm p-value, giữ nguyên thứ tự đầu vào."""
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def compute_benchmark_stats(benchmark):
    """Từ kết quả benchmark (dict fitness theo thuật toán) tính thống kê so sánh.

    benchmark = {"algorithms": [...], "base": "SPGA gốc", "fitness": {alg: [..]}, ...}
    Trả về dict: per_alg, friedman, wilcoxon."""
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
        wilcoxon.append({"a": a, "b": base, "stat": float(w.statistic), "p": float(w.pvalue)})
    if wilcoxon:
        adj = _holm([w["p"] for w in wilcoxon])
        for i, w in enumerate(wilcoxon):
            w["p_holm"] = adj[i]
    return {"per_alg": per, "friedman": friedman, "wilcoxon": wilcoxon}


def compute_sensitivity_metrics(sensitivity):
    """Tính mean theo k/sigma, mức đóng góp vào phương sai và chỉ số ổn định.

    sensitivity["fitness"][alg] là ma trận [len(k_grid), len(sigma_grid)].
    Đóng góp (contribution) ước lượng phần phương sai của Z do riêng k (hoặc sigma)
    gây ra, chuẩn hóa theo tổng phương sai (first-order variance decomposition)."""
    algs = sensitivity["algorithms"]
    out = {}
    for a in algs:
        F = np.array(sensitivity["fitness"][a], dtype=float)
        mean_by_k = F.mean(axis=1)       # trung bình theo sigma -> ảnh hưởng của k
        mean_by_sigma = F.mean(axis=0)   # trung bình theo k -> ảnh hưởng của sigma
        total_var = float(F.var())
        contrib_k = float(mean_by_k.var()) / total_var if total_var > 0 else 0.0
        contrib_sigma = float(mean_by_sigma.var()) / total_var if total_var > 0 else 0.0
        robustness = float(F.mean() / F.std()) if F.std() > 0 else 0.0
        out[a] = {
            "mean_by_k": mean_by_k.tolist(),
            "mean_by_sigma": mean_by_sigma.tolist(),
            "contribution_k": contrib_k,
            "contribution_sigma": contrib_sigma,
            "robustness": robustness,
        }
    return out


# ============================================================ vẽ biểu đồ
def _crystal_style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b8bfcc")
    ax.spines["bottom"].set_color("#b8bfcc")
    ax.tick_params(colors="#5a6270")


def _fig_to_png(fig):
    plt = _plt()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def data_url(png):
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def overlay_distribution_png(runs, bins=18):
    """Biểu đồ phủ phân bố tần suất fitness của các thuật toán (Crystal Ball style)."""
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
    ax.set_title("Forecast Probability Distribution (Crystal Ball style)", fontsize=11)
    ax.set_xlabel("Fitness Z (triệu)")
    ax.set_ylabel("Tần suất (chuẩn hóa)")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(alpha=0.25, axis="y")
    _crystal_style(ax)
    return _fig_to_png(fig)


def cumulative_probability_png(runs):
    """Biểu đồ xác suất tích lũy (CDF) fitness của các thuật toán."""
    plt = _plt()

    algs = runs["algorithms"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for a in algs:
        arr = np.sort(np.array(runs["fitness"][a], dtype=float) / 1e6)
        cdf = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, cdf, color=ALG_COLORS[a], lw=2, label=a)
    ax.set_title("Cumulative Probability (CDF)", fontsize=11)
    ax.set_xlabel("Fitness Z (triệu)")
    ax.set_ylabel("Xác suất tích lũy")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    _crystal_style(ax)
    return _fig_to_png(fig)


def convergence_band_png(convergence):
    """Dải hội tụ: đường trung bình, vùng ±1 độ lệch chuẩn, phân vị 10-90%.

    convergence = {"grid": [...], alg: [[...seed1...], [...seed2...], ...]} (cùng độ dài grid)."""
    plt = _plt()

    algs = [a for a in ALG_ORDER if a in convergence and len(convergence[a])]
    grid = np.array(convergence["grid"], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4.4))
    for a in algs:
        curves = np.array(convergence[a], dtype=float) / 1e6
        mean = curves.mean(0)
        std = curves.std(0, ddof=1)
        p10 = np.percentile(curves, 10, axis=0)
        p90 = np.percentile(curves, 90, axis=0)
        ax.plot(grid, mean, color=ALG_COLORS[a], lw=2, label=f"{a} (TB)")
        ax.fill_between(grid, mean - std, mean + std, color=ALG_COLORS[a], alpha=0.16)
        ax.fill_between(grid, p10, p90, color=ALG_COLORS[a], alpha=0.08)
    ax.set_title("Convergence Band (Mean ± Std & 10–90%)", fontsize=11)
    ax.set_xlabel("Số lần gọi evaluate")
    ax.set_ylabel("Fitness tốt nhất (triệu)")
    ax.legend(loc="lower right", frameon=False)
    ax.grid(alpha=0.25)
    _crystal_style(ax)
    return _fig_to_png(fig)


def sensitivity_tornado_png(sensitivity, metrics=None):
    """Biểu đồ Tornado: độ nhạy của Z với k và sigma (thay đổi min -> max)."""
    plt = _plt()

    algs = sensitivity["algorithms"]
    metrics = metrics or {a: {} for a in algs}
    rows, vals, colors = [], [], []
    for a in algs:
        m = metrics.get(a, {})
        mbk = np.array(m.get("mean_by_k", sensitivity.get("mean_by_k", {}).get(a, [])) or [], dtype=float)
        mbs = np.array(m.get("mean_by_sigma", sensitivity.get("mean_by_sigma", {}).get(a, [])) or [], dtype=float)
        if mbk.size:
            vals.append(float(mbk.max() - mbk.min()) / 1e6)
            rows.append(f"{a} · k")
            colors.append(CB_GREEN)
        if mbs.size:
            vals.append(float(mbs.max() - mbs.min()) / 1e6)
            rows.append(f"{a} · σ")
            colors.append(CB_ORANGE)
    fig, ax = plt.subplots(figsize=(8, 0.6 + 0.85 * len(rows)))
    y = np.arange(len(rows))
    ax.barh(y, vals, color=colors, alpha=0.85)
    ax.axvline(0, color="#333", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels(rows)
    ax.invert_yaxis()
    ax.set_title("Sensitivity Tornado: ΔZ khi k và σ quét từ min → max", fontsize=11)
    ax.set_xlabel("Δ Z (triệu)")
    ax.grid(alpha=0.25, axis="x")
    _crystal_style(ax)
    return _fig_to_png(fig)


# ============================================================ báo cáo Excel
def _write_cell(ws, row, col, value, fmt=None, bold=False, fill=None, align=None):
    c = ws.cell(row=row, column=col, value=value)
    if fmt:
        c.number_format = fmt
    if bold:
        c.font = Font(bold=True)
    if fill:
        c.fill = fill
    if align:
        c.alignment = Alignment(**align)
    return c


def build_excel_report(benchmark, sensitivity, path):
    """Ghi báo cáo .xlsx 4 tab: Executive Summary, Benchmark Runs, Crystal Ball Visuals,
    Sensitivity Analysis. Trả về đường dẫn file."""
    from openpyxl import Workbook
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.utils import get_column_letter

    stats = compute_benchmark_stats(benchmark)
    sens_metrics = compute_sensitivity_metrics(sensitivity) if sensitivity else {}
    algs = benchmark["algorithms"]
    params = benchmark.get("params", {})

    HEADER_FILL = PatternFill("solid", fgColor="1F3A93")
    HEADER_FONT = Font(color="FFFFFF", bold=True)
    ZEBRA_FILL = PatternFill("solid", fgColor="EEF2FB")
    THIN = Side(style="thin", color="D9DEE7")
    BORDER = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
    NUM = "#,##0.00"

    wb = Workbook()

    def style_header(ws, row, ncols):
        for c in range(1, ncols + 1):
            cell = ws.cell(row=row, column=c)
            cell.fill = HEADER_FILL
            cell.font = HEADER_FONT
            cell.alignment = Alignment(horizontal="center", vertical="center")
            cell.border = BORDER

    # ================= Tab 1: Executive Summary =================
    ws = wb.active
    ws.title = "Executive Summary"
    ws["A1"] = "Báo cáo kiểm định lặp — SPGA gốc, GA-TS, GA-SA"
    ws["A1"].font = Font(bold=True, size=14, color="1F3A93")
    ws["A2"] = (f"Ngân sách {params.get('budget', '—')} lần đánh giá · k = {params.get('k', '—')} · "
                f"σ = {params.get('sigma', '—')} · {benchmark['n_runs']} lần lặp")
    ws["A2"].font = Font(italic=True, color="667085")

    r = 4
    headers = ["Chỉ số"] + algs
    for j, h in enumerate(headers, 1):
        ws.cell(row=r, column=j, value=h)
    style_header(ws, r, len(headers))
    r += 1
    mean_row = r  # dòng "Fitness trung bình" để vẽ bar chart

    def add_row(label, getter, fmt=NUM, val_fmt=None):
        nonlocal r
        ws.cell(row=r, column=1, value=label).border = BORDER
        for j, a in enumerate(algs, 2):
            c = ws.cell(row=r, column=j, value=getter(a))
            c.number_format = val_fmt or fmt
            c.border = BORDER
        if r % 2 == 0:
            for j in range(1, len(headers) + 1):
                ws.cell(row=r, column=j).fill = ZEBRA_FILL
        r += 1

    add_row("Fitness trung bình (triệu)", lambda a: stats["per_alg"][a]["mean"] / 1e6)
    add_row("Độ lệch chuẩn (triệu)", lambda a: stats["per_alg"][a]["std"] / 1e6)
    add_row("Trung vị (triệu)", lambda a: stats["per_alg"][a]["median"] / 1e6)
    add_row("Nhỏ nhất (triệu)", lambda a: stats["per_alg"][a]["min"] / 1e6)
    add_row("Lớn nhất (triệu)", lambda a: stats["per_alg"][a]["max"] / 1e6)
    add_row("Cải thiện so với gốc", lambda a: stats["per_alg"][a]["improvement_pct"], val_fmt="0.00%")
    add_row("Win-rate so với gốc", lambda a: (stats["per_alg"][a]["win_rate"]
             if stats["per_alg"][a].get("win_rate") is not None else None), val_fmt="0.0%")

    r += 1
    ws.cell(row=r, column=1, value="Kiểm định thống kê").font = Font(bold=True, color="1F3A93")
    r += 1
    if stats["friedman"]:
        ws.cell(row=r, column=1, value="Friedman p-value").border = BORDER
        c = ws.cell(row=r, column=2, value=stats["friedman"]["p"])
        c.number_format = "0.0000"
        c.border = BORDER
        r += 1
    for w in stats["wilcoxon"]:
        ws.cell(row=r, column=1, value=f"Wilcoxon {w['a']} vs {w['b']} p (Holm)").border = BORDER
        c = ws.cell(row=r, column=2, value=w["p_holm"])
        c.number_format = "0.0000"
        c.border = BORDER
        r += 1

    # Bar chart so sánh Z trung bình (native Excel chart)
    from openpyxl.chart import BarChart, Reference
    chart = BarChart()
    chart.type = "col"
    chart.style = 10
    chart.title = "Z trung bình (triệu)"
    data_ref = Reference(ws, min_col=2, min_row=mean_row, max_col=1 + len(algs), max_row=mean_row)
    cats_ref = Reference(ws, min_col=2, min_row=3, max_col=1 + len(algs), max_row=3)
    chart.add_data(data_ref, titles_from_data=False)
    chart.set_categories(cats_ref)
    chart.legend = None
    ws.add_chart(chart, f"A{r + 2}")

    # ================= Tab 2: Benchmark Runs =================
    ws2 = wb.create_sheet("Benchmark Runs")
    hdr = ["Algorithm", "Seed", "Fitness", "Mean Profit", "MAD", "Runtime (s)"]
    for j, h in enumerate(hdr, 1):
        ws2.cell(row=1, column=j, value=h)
    style_header(ws2, 1, len(hdr))
    row = 2
    seeds = benchmark.get("seeds", list(range(1, benchmark["n_runs"] + 1)))
    for a in algs:
        for i, s in enumerate(seeds):
            vals = [
                a, s,
                benchmark["fitness"][a][i],
                benchmark.get("mean_profit", {}).get(a, [0] * len(seeds))[i],
                benchmark.get("mad", {}).get(a, [0] * len(seeds))[i],
                benchmark.get("runtime", {}).get(a, [0] * len(seeds))[i],
            ]
            for j, v in enumerate(vals, 1):
                c = ws2.cell(row=row, column=j, value=v)
                c.border = BORDER
                if j >= 3:
                    c.number_format = NUM if j == 3 else ("#,##0.00" if j < 6 else "0.00")
                if row % 2 == 0:
                    c.fill = ZEBRA_FILL
            row += 1
    for j, w in enumerate([16, 10, 14, 14, 12, 12], 1):
        ws2.column_dimensions[get_column_letter(j)].width = w

    # ================= Tab 3: Crystal Ball Visuals =================
    ws3 = wb.create_sheet("Crystal Ball Visuals")
    ws3["A1"] = "Biểu đồ phong cách Oracle Crystal Ball"
    ws3["A1"].font = Font(bold=True, size=14, color="1F3A93")
    png1 = overlay_distribution_png(benchmark)
    png2 = cumulative_probability_png(benchmark)
    img1 = XLImage(io.BytesIO(png1))
    img1.width, img1.height = 760, 420
    ws3.add_image(img1, "A3")
    img2 = XLImage(io.BytesIO(png2))
    img2.width, img2.height = 760, 420
    ws3.add_image(img2, "A28")

    # ================= Tab 4: Sensitivity Analysis =================
    ws4 = wb.create_sheet("Sensitivity Analysis")
    ws4["A1"] = "Phân tích độ nhạy theo k và σ"
    ws4["A1"].font = Font(bold=True, size=14, color="1F3A93")

    k_grid = sensitivity.get("k_grid", []) if sensitivity else []
    sigma_grid = sensitivity.get("sigma_grid", []) if sensitivity else []
    r = 3
    if not sensitivity:
        ws4.cell(row=r, column=1, value="Chưa chạy phân tích độ nhạy. Hãy mở tab 'Phân tích độ nhạy' và bấm Chạy.").font = Font(italic=True, color="667085")
    for a in (algs if sensitivity else []):
        ws4.cell(row=r, column=1, value=f"{a} — fitness trung bình (triệu)").font = Font(bold=True, color=ALG_COLORS[a].lstrip("#"))
        r += 1
        ws4.cell(row=r, column=1, value="k \\ σ")
        for j, s in enumerate(sigma_grid, 2):
            ws4.cell(row=r, column=j, value=f"σ={s}")
        style_header(ws4, r, 1 + len(sigma_grid))
        r += 1
        F = np.array(sensitivity["fitness"][a], dtype=float)
        for i, k in enumerate(k_grid):
            ws4.cell(row=r, column=1, value=f"k={k}")
            for j in range(len(sigma_grid)):
                c = ws4.cell(row=r, column=j + 2, value=float(F[i, j]) / 1e6)
                c.number_format = NUM
            r += 1
        m = sens_metrics.get(a, {})
        ws4.cell(row=r, column=1, value="Đóng góp k").border = BORDER
        c = ws4.cell(row=r, column=2, value=m.get("contribution_k"))
        c.number_format = "0.00%"
        c.border = BORDER
        r += 1
        ws4.cell(row=r, column=1, value="Đóng góp σ").border = BORDER
        c = ws4.cell(row=r, column=2, value=m.get("contribution_sigma"))
        c.number_format = "0.00%"
        c.border = BORDER
        r += 1
        ws4.cell(row=r, column=1, value="Chỉ số ổn định (mean/std)").border = BORDER
        c = ws4.cell(row=r, column=2, value=m.get("robustness"))
        c.number_format = NUM
        c.border = BORDER
        r += 2

    if sensitivity:
        png3 = sensitivity_tornado_png(sensitivity, sens_metrics)
        img3 = XLImage(io.BytesIO(png3))
        img3.width, img3.height = 700, 260 + 70 * len(algs)
        ws4.add_image(img3, "G3")

    wb.save(path)
    return path
