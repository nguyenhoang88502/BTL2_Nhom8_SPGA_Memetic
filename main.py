"""
main.py - chạy thực nghiệm so sánh SPGA gốc, GA-TS và GA-SA (Phần 4).

    python main.py            # thí nghiệm chính + ablation
    python main.py --quick    # chỉ thí nghiệm chính

Đầu ra (thư mục output/):
    results.csv            mỗi dòng = 1 lần chạy (thuật toán, seed, chỉ số)
    history.csv            fitness tốt nhất theo số lần gọi evaluate
    summary.csv            trung bình / độ lệch chuẩn / min / max theo thuật toán
    stats.txt              Friedman, Wilcoxon theo cặp hiệu chỉnh Holm, số seed thắng
    best_portfolio.csv     danh mục máy của nghiệm tốt nhất (giống Bảng 1 bài báo)
    fig_convergence.png    biểu đồ hội tụ trung bình, dải ±1 độ lệch chuẩn
    fig_boxplot.png        boxplot fitness 10 lần chạy
    ablation.csv           các biến thể của TS và SA, cùng hill climbing cũ
    fig_ablation.png       biểu đồ ablation
"""
import argparse
import functools
import os
import platform
import time

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

import model as mdl
import ga

N_SCEN = 30
SCEN_SEED = 2026          # cùng 1 bộ kịch bản cho mọi lần chạy
TEST_SCEN = 1000          # bộ kịch bản độc lập để kiểm tra ngoài mẫu
TEST_SEED = 9999
SEEDS = range(1, 11)
OUT = "output"
LABEL = {"none": "SPGA gốc", "tabu": "GA-TS", "sa": "GA-SA"}
COLORS = {"SPGA gốc": "#1F3A93", "GA-TS": "#E07B39", "GA-SA": "#2E8B57"}
BASE, TS, SA = LABEL["none"], LABEL["tabu"], LABEL["sa"]
PAIRS = [(TS, BASE), (SA, BASE), (TS, SA)]


def first_reach(hist, target):
    for n, f in hist:
        if f >= target:
            return n
    return np.nan


def run_set(evaluate, method, seeds, **kw):
    rows, hists = [], {}
    for s in seeds:
        bx, hist, rt = ga.run_ga(evaluate, method, seed=s, **kw)
        info = ga.run_ga.last_info
        f, mean, mad = evaluate(bx)
        rows.append({"seed": s, "best_x": bx, "fitness": f, "mean_profit": mean,
                     "mad": mad, "runtime_s": rt, **info})
        hists[s] = hist
    return rows, hists


def holm(pvalues):
    """Hiệu chỉnh Holm cho một nhóm p-value, giữ nguyên thứ tự đầu vào."""
    m = len(pvalues)
    adjusted, running = [0.0] * m, 0.0
    for rank, i in enumerate(np.argsort(pvalues)):
        running = max(running, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running
    return adjusted


def write_stats(df):
    """Friedman cho cả 3 thuật toán, Wilcoxon hai phía theo cặp (cùng seed) hiệu chỉnh Holm."""
    z = {alg: df[df.algorithm == alg].sort_values("seed").fitness.values for alg in LABEL.values()}
    fr = stats.friedmanchisquare(*z.values())
    wil = [stats.wilcoxon(z[a], z[b], alternative="two-sided") for a, b in PAIRS]
    adjusted = holm([w.pvalue for w in wil])
    best_each_seed = np.argmax(np.array(list(z.values())), axis=0)
    with open(f"{OUT}/stats.txt", "w", encoding="utf-8") as fh:
        fh.write(f"Friedman (3 thuat toan, 10 seed): chi2={fr.statistic:.3f}, p={fr.pvalue:.5f}\n")
        for (a, b), w, p in zip(PAIRS, wil, adjusted):
            fh.write(f"Wilcoxon {a} vs {b}: stat={w.statistic:.1f}, p={w.pvalue:.5f}, p_Holm={p:.5f}, "
                     f"{a} thang {int((z[a] > z[b]).sum())}/10 seed\n")
        for i, alg in enumerate(z):
            fh.write(f"So seed co Z cao nhat ca 3: {alg} = {int((best_each_seed == i).sum())}/10\n")
        for alg in (TS, SA):
            fh.write(f"Cai thien Z trung binh so voi {BASE}: {alg} = {(z[alg].mean() / z[BASE].mean() - 1) * 100:+.3f}%\n")
    print(open(f"{OUT}/stats.txt", encoding="utf-8").read())


def plot_main(df, all_hist):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    grid = np.arange(ga.POP_SIZE, ga.BUDGET + 1, 30)
    fig, ax = plt.subplots(figsize=(7.5, 4.2), dpi=200)
    for alg, hs in all_hist.items():
        curves = []
        for h in hs.values():
            n = np.array([x[0] for x in h]); f = np.array([x[1] for x in h])
            curves.append(f[np.searchsorted(n, grid, side="right") - 1])
        curves = np.array(curves) / 1e6
        mean, std = curves.mean(0), curves.std(0, ddof=1)
        ax.plot(grid, mean, color=COLORS[alg], lw=2, label=f"{alg} (trung bình 10 lần)")
        ax.fill_between(grid, mean - std, mean + std, color=COLORS[alg], alpha=0.15)
    ax.set_xlabel("Số lần gọi evaluate")
    ax.set_ylabel("Fitness tốt nhất (triệu), dải ±1 độ lệch chuẩn")
    ax.set_ylim(bottom=max(ax.get_ylim()[0], 44.5))
    ax.grid(alpha=0.3); ax.legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_convergence.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.2), dpi=200)
    algs = list(LABEL.values())
    data = [df[df.algorithm == alg].fitness.values / 1e6 for alg in algs]
    bp = ax.boxplot(data, tick_labels=algs, patch_artist=True, widths=0.5, showfliers=False)
    for patch, alg in zip(bp["boxes"], algs):
        patch.set_facecolor(COLORS[alg]); patch.set_alpha(0.35)
    for i, d in enumerate(data, 1):
        ax.scatter(np.full(len(d), i) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)),
                   d, s=14, color="black", zorder=3)
    ax.set_ylabel("Fitness tốt nhất (triệu)"); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_boxplot.png"); plt.close(fig)


def run_ablation(df, evaluate, evaluate_test):
    """So các biến thể trên 10 seed. p_vs_base: Wilcoxon hai phía so với cấu hình chuẩn của thuật
    toán đó, chưa hiệu chỉnh đa so sánh."""
    z_ts = df[df.algorithm == TS].sort_values("seed")
    z_sa = df[df.algorithm == SA].sort_values("seed")
    configs = []
    for n_cand in (3, 5, 10):
        for tenure in (3, 7, 15):
            configs.append(("TS: n_cand × tenure", f"n={n_cand}, t={tenure}", "tabu",
                            dict(ts_n_cand=n_cand, ts_tenure=tenure), z_ts))
    configs.append(("TS: tabu chung vs riêng", "tabu chung", "tabu", {}, z_ts))
    configs.append(("TS: tabu chung vs riêng", "tabu riêng", "tabu", dict(ts_shared=False), z_ts))
    for p0 in (0.2, 0.5, 0.8):
        configs.append(("SA: xác suất nhận ban đầu", f"p0={p0}", "sa", dict(sa_p0=p0), z_sa))
    configs.append(("Hill climbing cũ", "first improvement", "hc", {}, None))

    rows = []
    for group, name, method, kw, base in configs:
        is_base = base is not None and kw in ({}, dict(ts_n_cand=5, ts_tenure=7), dict(sa_p0=0.5))
        runs = base.to_dict("records") if is_base else run_set(evaluate, method, SEEDS, **kw)[0]
        fits = np.array([r["fitness"] for r in runs])
        tests = np.array([evaluate_test(r["best_x"])[0] for r in runs])
        p = np.nan
        if base is not None and not is_base:
            p = stats.wilcoxon(fits, base.fitness.values, alternative="two-sided").pvalue
        rows.append({"group": group, "config": name, "method": method, "is_base": is_base,
                     "fitness_mean": fits.mean(), "fitness_std": fits.std(ddof=1),
                     "fitness_min": fits.min(), "fitness_max": fits.max(),
                     "test_fitness_mean": tests.mean(), "p_vs_base": p,
                     "ls_steps_mean": np.mean([r["ls_steps"] for r in runs]),
                     "ls_worse_mean": np.mean([r["ls_worse"] for r in runs]),
                     "ls_aspiration_mean": np.mean([r["ls_aspiration"] for r in runs]),
                     "ls_blocked_mean": np.mean([r["ls_blocked"] for r in runs])})
        print(f"{group:28s} {name:20s} {fits.mean():,.0f} ± {fits.std(ddof=1):,.0f}", flush=True)
    adf = pd.DataFrame(rows)
    adf.to_csv(f"{OUT}/ablation.csv", index=False, float_format="%.4f")

    groups = list(dict.fromkeys(g for g in adf.group if g != "Hill climbing cũ"))
    baseline = df[df.algorithm == BASE].fitness.mean() / 1e6
    fig, axes = plt.subplots(1, len(groups), figsize=(11, 3.9), dpi=200, sharey=True,
                             gridspec_kw={"width_ratios": [3, 1.1, 1.3]})
    for ax, g in zip(axes, groups):
        sub = adf[adf.group == g]
        color = COLORS[SA] if g.startswith("SA") else COLORS[TS]
        for i, r in enumerate(sub.itertuples()):
            ax.errorbar(i, r.fitness_mean / 1e6, yerr=r.fitness_std / 1e6, fmt="o", color=color,
                        alpha=1.0 if r.is_base else 0.55, ms=8 if r.is_base else 6, capsize=4, lw=1.4)
        ax.axhline(baseline, color=COLORS[BASE], ls="--", lw=1.2, label=f"{BASE} (trung bình)")
        ax.set_xticks(range(len(sub))); ax.set_xticklabels(sub.config, rotation=40, ha="right")
        ax.set_xlim(-0.5, len(sub) - 0.5); ax.set_title(g, fontsize=10); ax.grid(alpha=0.3, axis="y")
    axes[0].set_ylabel("Fitness: trung bình ± độ lệch chuẩn (triệu)")
    axes[0].legend(loc="lower right")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_ablation.png"); plt.close(fig)


def main(quick=False):
    started = time.perf_counter()
    os.makedirs(OUT, exist_ok=True)
    scen = mdl.sample_scenarios(N_SCEN, SCEN_SEED)
    test = mdl.sample_scenarios(TEST_SCEN, TEST_SEED)
    evaluate = functools.partial(mdl.evaluate, scenarios=scen)
    evaluate_test = functools.partial(mdl.evaluate, scenarios=test)

    print(f"Python {platform.python_version()}, numpy {np.__version__}, "
          f"{platform.processor() or platform.machine()}")

    # ---------------------------------------------------- thí nghiệm chính
    all_rows, all_hist = [], {}
    for method, label in LABEL.items():
        rows, hists = run_set(evaluate, method, SEEDS)
        for r in rows:
            r["algorithm"] = label
        all_rows += rows
        all_hist[label] = hists
        print(f"{label:10s} xong: fitness TB = {np.mean([r['fitness'] for r in rows]):,.0f}", flush=True)

    df = pd.DataFrame(all_rows)
    # chỉ số hội tụ
    target = df[df.algorithm == BASE].fitness.median()
    for i, r in df.iterrows():
        h = all_hist[r.algorithm][r.seed]
        f_init, f_fin = h[ga.POP_SIZE - 1][1], h[-1][1]
        df.loc[i, "evals_to_95pct_gain"] = first_reach(h, f_init + 0.95 * (f_fin - f_init))
        df.loc[i, "evals_to_spga_median"] = first_reach(h, target)
        tf, tm, td = evaluate_test(r.best_x)
        df.loc[i, "test_fitness"], df.loc[i, "test_mean_profit"], df.loc[i, "test_mad"] = tf, tm, td

    cols = ["algorithm", "seed", "fitness", "mean_profit", "mad", "runtime_s",
            "evals_to_95pct_gain", "evals_to_spga_median", "test_fitness",
            "test_mean_profit", "test_mad", "generations", "ls_evals", "evals",
            "ls_runs", "ls_steps", "ls_worse", "ls_aspiration", "ls_blocked", "ls_gain"]
    df[cols].to_csv(f"{OUT}/results.csv", index=False, float_format="%.4f")

    # lịch sử hội tụ
    hrows = []
    for alg, hs in all_hist.items():
        for s, h in hs.items():
            hrows += [(alg, s, n, f) for n, f in h]
    pd.DataFrame(hrows, columns=["algorithm", "seed", "evals", "best_fitness"]).to_csv(
        f"{OUT}/history.csv", index=False, float_format="%.2f")

    # tổng hợp
    metrics = ["fitness", "mean_profit", "mad", "runtime_s", "evals_to_95pct_gain",
               "evals_to_spga_median", "test_fitness", "ls_steps", "ls_worse", "ls_aspiration",
               "ls_blocked"]
    summary = df.groupby("algorithm", sort=False)[metrics].agg(["mean", "std", "min", "max"])
    summary.to_csv(f"{OUT}/summary.csv", float_format="%.2f")

    write_stats(df)
    plot_main(df, all_hist)

    # ---------------------------------------------------- danh mục máy nghiệm tốt nhất
    best = df.loc[df.fitness.idxmax()]
    d = mdl.decode(best.best_x)
    rows = []
    for m in range(mdl.M):
        rows.append([f"Tester M{m+1}", "Đầu tư (in-house)", mdl.K0_M[m], d["K_m"][m]] + [""] * (mdl.P - 1))
        rows.append([f"Tester M{m+1}", "Chuyển/thuê", 0] + list(d["X_m"][:, m]))
        rows.append([f"Tester M{m+1}", "Tổng", mdl.K0_M[m]] + list(d["K_m"][m] + d["X_m"][:, m]))
    for a_ in range(mdl.A):
        rows.append([f"Handler H{a_+1}", "Đầu tư (in-house)", mdl.K0_A[a_], d["K_a"][a_]] + [""] * (mdl.P - 1))
        rows.append([f"Handler H{a_+1}", "Chuyển/thuê", 0] + list(d["X_a"][:, a_]))
        rows.append([f"Handler H{a_+1}", "Tổng", mdl.K0_A[a_]] + list(d["K_a"][a_] + d["X_a"][:, a_]))
    pd.DataFrame(rows, columns=["Nguồn lực", "Loại", "Quý 0"] + [f"Quý {p}" for p in range(1, mdl.P + 1)]
                 ).to_csv(f"{OUT}/best_portfolio.csv", index=False, encoding="utf-8-sig")
    qrows = []
    for p in range(mdl.P):
        qrows.append([p + 1] + [round(d["Q"][p, m, t]) for m in range(mdl.M) for t in range(mdl.T)])
    pd.DataFrame(qrows, columns=["Quý"] + [f"M{m+1}-SP{t+1}" for m in range(mdl.M) for t in range(mdl.T)]
                 ).to_csv(f"{OUT}/best_allocation.csv", index=False, encoding="utf-8-sig")
    print(f"Nghiệm tốt nhất: {best.algorithm} seed {best.seed}, fitness {best.fitness:,.0f}")
    print(f"Thí nghiệm chính: {time.perf_counter() - started:.1f} giây", flush=True)

    if not quick:
        run_ablation(df, evaluate, evaluate_test)
    print(f"Tổng thời gian: {time.perf_counter() - started:.1f} giây")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    main(ap.parse_args().quick)
