"""
demo_server.py - server nhỏ cho bản demo HTML (Nhóm 8 - SPGA gốc, GA-TS, GA-SA, GA-LP).

Chạy:  python demo_server.py [--port 8765]
Mở:    http://localhost:8765

API:
  GET /                  -> demo.html
  GET /api/defaults      -> tham số mặc định
  GET /api/run?...       -> chạy thuật toán, gửi từng thế hệ (Server-Sent Events)
  GET /api/results       -> kết quả thực nghiệm đã lưu trong output/
  GET /api/rerun         -> chạy lại main.py --quick, gửi log (Server-Sent Events)
  GET/POST /api/run-benchmark -> kiểm định lặp nhiều seed (SSE), tự động chạy MILP nền
  GET /api/sensitivity   -> phân tích độ nhạy k × σ (SSE)
  GET/POST /api/export-excel -> tải báo cáo Excel 1 sheet
  GET/POST /api/milp, GET /api/milp/status -> cận tối ưu toàn cục MILP (chạy nền)
  GET /output/<file>     -> hình trong output/
Server chuẩn + numpy; exporter cần scipy/pandas/matplotlib/openpyxl; MILP cần pulp.
"""
import argparse
import csv
import functools
import json
import os
import subprocess
import sys
import threading
import time
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model as mdl  # noqa: E402
import ga  # noqa: E402
import exporter  # noqa: E402
import milp_solver  # noqa: E402
import matheuristics  # noqa: E402

DEFAULT_MODEL = dict(K_RISK=mdl.K_RISK, SIGMA=mdl.SIGMA, MU=mdl.MU.copy())
RUN_LOCK = threading.Lock()
# Kết quả kiểm định lặp / độ nhạy / MILP được cache lại để export và UI đọc lại không cần chạy lại.
BENCHMARK_CACHE = {}
SENSITIVITY_CACHE = {}
SENSITIVITY_METRICS = {}
MILP_CACHE = {}
MILP_THREAD = None

# ---------------------------------------------------------------- log ra terminal (cho người xem demo)
if os.name == "nt":
    os.system("")  # bật màu ANSI trên cửa sổ PowerShell/cmd của Windows 10/11
COL = {"spga": "\033[94m", "ts": "\033[93m", "sa": "\033[92m", "ml": "\033[95m",
       "ok": "\033[92m", "dim": "\033[90m", "head": "\033[96m", "bold": "\033[1m", "end": "\033[0m"}
TAG = {"spga": "SPGA ", "ts": "GA-TS", "sa": "GA-SA", "ml": "GA-LP"}
LS_METHOD = {"spga": "none", "ts": "tabu", "sa": "sa", "ml": "none"}
MODES = {"all": ("spga", "ts", "sa", "ml"), "spga": ("spga",), "ts": ("ts",),
         "sa": ("sa",), "ml": ("ml",)}
FULL_LABEL = {"spga": "SPGA gốc", "ts": "GA-TS", "sa": "GA-SA", "ml": "Matheuristics (GA-LP)"}
ALL_CODES = ("spga", "ts", "sa", "ml")


def selected_algos(q):
    """Đọc danh sách thuật toán từ checkbox (algos=spga,ts,sa) hoặc mode cũ."""
    raw = q.get("algos", [""])[0].strip()
    if raw:
        seen, codes = set(), []
        for c in raw.replace(";", ",").split(","):
            c = c.strip()
            if c in ALL_CODES and c not in seen:
                seen.add(c)
                codes.append(c)
        if codes:
            return tuple(codes)
    mode = q.get("mode", ["all"])[0]
    return MODES.get(mode, MODES["all"])


def log(msg, col=None):
    t = time.strftime("%H:%M:%S")
    c, e = (COL.get(col, ""), COL["end"]) if col else ("", "")
    try:
        print(f"{COL['dim']}{t}{COL['end']} {c}{msg}{e}", flush=True)
    except UnicodeEncodeError:
        print(f"{t} {msg}".encode("ascii", "replace").decode(), flush=True)

DEFAULTS = {
    "k": 0.5, "sigma": 7000, "n_scen": 30, "scen_seed": 2026,
    "mu1": 25000, "mu2": 22000, "mu3": 15000,
    "pop": 30, "pc": 0.9, "pm": 0.05, "tour": 2, "budget": 6000,
    "ls_freq": 10, "ls_top": 3, "ls_max": 20, "ls_shift": 10, "lamarck": 1,
    "ts_n_cand": ga.TS_N_CAND, "ts_tenure": ga.TS_TENURE, "sa_p0": ga.SA_P0,
    "seed": 1, "mode": "all", "delay": 30,
    "n_runs": 10, "sens_k_min": 0.0, "sens_k_max": 1.0, "sens_k_steps": 5,
    "sens_sigma_min": 1000, "sens_sigma_max": 15000, "sens_sigma_steps": 5,
    "sens_budget": 2000, "sens_runs": 1, "alpha": 0.05, "time_limit": 120,
    "stopping": "evaluations", "time_limit_sec": 180.0,
}


def fnum(q, key, cast=float):
    v = q.get(key, [DEFAULTS[key]])[0]
    try:
        return cast(v)
    except (TypeError, ValueError):
        return cast(DEFAULTS[key])


def describe(move):
    kind, idx, step = move
    sign = "Thêm" if (step > 0 and kind != "shift") else "Bớt"
    if kind == "K_m":
        return f"{sign} 1 tester M{idx + 1} mua"
    if kind == "K_a":
        return f"{sign} 1 handler H{idx + 1} mua"
    if kind == "X_m":
        p, m = idx
        return f"{sign} 1 tester M{m + 1} thuê ở quý {p + 1}"
    if kind == "X_a":
        p, a = idx
        return f"{sign} 1 handler H{a + 1} thuê ở quý {p + 1}"
    p, t, src, dst = idx
    return f"Chuyển {round(step * 100)}% việc SP{t + 1} quý {p + 1} từ M{src + 1} sang M{dst + 1}"


def step_message(s):
    """Câu mô tả một bước tìm kiếm cục bộ, dùng chung cho log terminal và log trên trang."""
    text = describe(s["move"])
    gain = (s["f_to"] - s["f_from"]) / 1e6
    if s["kind"] == "blocked":
        return f"bước {text} bị cấm ({gain:+.3f})"
    if s["kind"] == "aspiration":
        return f"aspiration: {text} ({gain:+.3f}, vượt Z tốt nhất nên được đi dù bị cấm)"
    if s["kind"] == "worse":
        if "T" in s:
            return f"nhận bước tệ hơn, T = {s['T'] / 1e6:.4f} tr, p = {s['p']:.2f}: {text} ({gain:+.3f})"
        return f"đi bước tệ hơn (tabu): {text} ({gain:+.3f})"
    return f"{text} ({gain:+.3f})"


def plan_summary(x, scen):
    d = mdl.decode(x)
    need_m, _ = mdl._loads(d)
    h = mdl.scenario_profits(x, scen)
    f, mean, mad = mdl.evaluate(x, scen)
    return {
        "owned": [int(d["K_m"].sum())] * mdl.P,
        "rented": [int(v) for v in d["X_m"].sum(axis=1)],
        "need": [round(float(v), 2) for v in need_m.sum(axis=1)],
        "K_m": [int(v) for v in d["K_m"]], "K_a": [int(v) for v in d["K_a"]],
        "h": [round(float(v) / 1e6, 4) for v in h],
        "z": f / 1e6, "mean": mean / 1e6, "mad": mad / 1e6,
    }


def _resample(hist, grid):
    """Nội suy lịch sử hội tụ [(evals, best)] về lưới evals cố định (cho biểu đồ dải hội tụ)."""
    n = np.array([p[0] for p in hist])
    f = np.array([p[1] for p in hist])
    idx = np.clip(np.searchsorted(n, grid, side="right") - 1, 0, len(f) - 1)
    return f[idx].tolist()


def _plan_for(algo, best_x, scen, ev_ml):
    """Phương án tốt nhất dạng 135 gen để vẽ biểu đồ. GA-LP (63 gen) được chuyển
    qua dict máy + sản lượng tối ưu từ LP rồi encode lại."""
    if algo == "ml" and ev_ml is not None:
        return plan_summary(mdl.encode(ev_ml.plan_dict(best_x)), scen)
    return plan_summary(best_x, scen)


def milp_launch(k, sigma, mu, n_scen, scen_seed, time_limit=None):
    """Khởi động MILP nền (KHÔNG giữ RUN_LOCK): dùng snapshot tham số riêng nên chạy
    song song an toàn với các lần chạy GA. Bỏ qua nếu đã có kết quả cùng tham số."""
    global MILP_THREAD
    if MILP_THREAD is not None and MILP_THREAD.is_alive():
        return
    tl = max(int(time_limit or 120), 20)
    m = MILP_CACHE
    if (m and "error" not in m and abs(float(m.get("k", -1)) - k) < 1e-9
            and abs(float(m.get("sigma", -1)) - sigma) < 1e-6
            and m.get("n_scen") == n_scen and m.get("scen_seed") == scen_seed):
        return
    mu_snap = [float(v) for v in mu]

    def worker():
        global MILP_THREAD
        try:
            log(f"TÍNH CẬN TỐI ƯU MILP (CBC, tối đa {tl}s, {n_scen} kịch bản)...", "head")
            res = milp_solver.solve_milp_exact(k=k, sigma=sigma, n_scen=n_scen,
                                               scen_seed=scen_seed, time_limit=tl, mu=mu_snap)
            MILP_CACHE.clear()
            MILP_CACHE.update(res)
            log(f"  MILP: Z_opt = {res['z_opt'] / 1e6:.3f} tr (status {res['status']}, {res['solve_time_s']:.1f}s)", "ok")
        except Exception as e:
            log(f"  MILP lỗi: {e}", "bold")
            MILP_CACHE.clear()
            MILP_CACHE.update({"error": str(e), "k": k, "sigma": sigma,
                               "n_scen": n_scen, "scen_seed": scen_seed})
        finally:
            MILP_THREAD = None

    MILP_THREAD = threading.Thread(target=worker, daemon=True)
    MILP_THREAD.start()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):  # yên lặng
        pass

    # ------------------------------------------------------------ tiện ích
    def send_json(self, obj, code=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype, download=None):
        if not os.path.isfile(path):
            self.send_error(404)
            return
        with open(path, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        if download:
            self.send_header("Content-Disposition", f'attachment; filename="{download}"')
        else:
            self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def start_sse(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

    def sse(self, event, data):
        msg = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
        self.wfile.write(msg.encode("utf-8"))
        self.wfile.flush()

    # ------------------------------------------------------------ định tuyến
    def do_GET(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/", "/index.html", "/demo.html"):
            return self.send_file(os.path.join(HERE, "demo.html"), "text/html; charset=utf-8")
        if u.path == "/api/defaults":
            return self.send_json(DEFAULTS)
        if u.path == "/api/run":
            return self.run_algo(q)
        if u.path == "/api/results":
            return self.send_json(load_results())
        if u.path == "/api/rerun":
            return self.rerun()
        if u.path in ("/api/benchmark", "/api/run-benchmark"):
            return self.benchmark(q)
        if u.path == "/api/benchmark/run":
            return self.benchmark_run(q)
        if u.path == "/api/sensitivity":
            return self.sensitivity(q)
        if u.path in ("/api/export_excel", "/api/export-excel"):
            return self.export_excel(q)
        if u.path == "/api/milp":
            return self.milp_start(q)
        if u.path == "/api/milp/status":
            return self.send_json({"running": MILP_THREAD is not None and MILP_THREAD.is_alive(),
                                   "result": MILP_CACHE or None})
        if u.path.startswith("/output/"):
            name = os.path.basename(u.path)
            ctype = "image/png" if name.endswith(".png") else "text/plain; charset=utf-8"
            return self.send_file(os.path.join(HERE, "output", name), ctype)
        self.send_error(404)

    def do_POST(self):
        u = urlparse(self.path)
        q = parse_qs(u.query)
        if u.path in ("/api/run-benchmark", "/api/benchmark"):
            return self.benchmark(q)
        if u.path in ("/api/export-excel", "/api/export_excel"):
            return self.export_excel(q)
        if u.path == "/api/milp":
            return self.milp_start(q)
        self.send_error(404)

    # ------------------------------------------------------------ chạy thuật toán
    def run_algo(self, q):
        if not RUN_LOCK.acquire(blocking=False):
            return self.send_json({"error": "Đang có một lần chạy khác"}, 409)
        try:
            self.start_sse()
            # tham số bài toán
            mdl.K_RISK = min(max(fnum(q, "k"), 0.0), 1.0)
            mdl.SIGMA = max(fnum(q, "sigma"), 0.0)
            mdl.MU = np.array([fnum(q, "mu1"), fnum(q, "mu2"), fnum(q, "mu3")])
            mdl.Q_MAX = 1.5 * mdl.MU
            scen = mdl.sample_scenarios(max(fnum(q, "n_scen", int), 2), fnum(q, "scen_seed", int))
            evaluate = functools.partial(mdl.evaluate, scenarios=scen)
            kw = dict(budget=max(fnum(q, "budget", int), 100), pop_size=max(fnum(q, "pop", int), 4),
                      pc=fnum(q, "pc"), pm=fnum(q, "pm"), tour=max(fnum(q, "tour", int), 1),
                      ls_freq=max(fnum(q, "ls_freq", int), 1), ls_top=max(fnum(q, "ls_top", int), 1),
                      ls_max_evals=max(fnum(q, "ls_max", int), 1), ls_shift=fnum(q, "ls_shift") / 100,
                      lamarck=bool(fnum(q, "lamarck", int)),
                      ts_n_cand=max(fnum(q, "ts_n_cand", int), 1), ts_tenure=max(fnum(q, "ts_tenure", int), 0),
                      sa_p0=min(max(fnum(q, "sa_p0"), 0.01), 0.99))
            stopping = q.get("stopping", ["evaluations"])[0]
            stopping = stopping if stopping in ("evaluations", "time") else "evaluations"
            kw["stopping_criterion"] = stopping
            kw["time_limit_sec"] = max(fnum(q, "time_limit_sec"), 1.0)
            seed = fnum(q, "seed", int)
            names = selected_algos(q)
            delay = max(fnum(q, "delay"), 0) / 1000

            ev_ml = matheuristics.LpEvaluator(k=mdl.K_RISK, sigma=mdl.SIGMA, scenarios=scen) if "ml" in names else None
            algos = []
            for a in names:
                if a == "ml":
                    algos.append((a, ga.run_ga_iter(ev_ml.evaluate, "none", seed,
                                                    repair_fn=matheuristics.repair_ml,
                                                    n_genes=matheuristics.N_GENES_ML, **kw)))
                else:
                    algos.append((a, ga.run_ga_iter(evaluate, LS_METHOD[a], seed, **kw)))
            milp_launch(k=mdl.K_RISK, sigma=mdl.SIGMA, mu=mdl.MU, n_scen=len(scen),
                        scen_seed=fnum(q, "scen_seed", int), time_limit=fnum(q, "time_limit", int))
            log("=" * 72, "head")
            log(f"BẮT ĐẦU CHẠY: {' + '.join(TAG[a].strip() for a, _ in algos)}   (seed {seed})", "head")
            log(f"  Bài toán : k={mdl.K_RISK}  μ={[int(v) for v in mdl.MU]}  σ={mdl.SIGMA:g}  "
                f"N={len(scen)} kịch bản (seed {fnum(q, 'scen_seed', int)})", "head")
            stop_desc = (f"ngân sách {kw['budget']} lần đánh giá" if kw["stopping_criterion"] == "evaluations"
                         else f"giới hạn {kw['time_limit_sec']:g} giây")
            log(f"  GA       : quần thể={kw['pop_size']}  pc={kw['pc']}  pm={kw['pm']}  tournament={kw['tour']}  "
                f"dừng theo {stop_desc}", "head")
            if "ts" in names or "sa" in names:
                log(f"  Cục bộ   : mỗi {kw['ls_freq']} thế hệ, top {kw['ls_top']}, tối đa {kw['ls_max_evals']} "
                    f"lần/cá thể, chuyển {kw['ls_shift']:.0%}, {'Lamarckian' if kw['lamarck'] else 'Baldwinian'}", "head")
            if "ts" in names:
                log(f"  GA-TS    : {kw['ts_n_cand']} ứng viên/vòng, tenure {kw['ts_tenure']}, tabu chung cho các cá thể", "head")
            if "sa" in names:
                log(f"  GA-SA    : xác suất nhận ban đầu {kw['sa_p0']:g}, T0 ước lượng từ 30 lần evaluate", "head")
            if "ml" in names:
                log(f"  GA-LP    : 63 gen máy móc + LP (HiGHS) tối ưu sản lượng cho từng phương án", "head")
            log("=" * 72, "head")
            self.sse("start", {"algos": [a for a, _ in algos], "budget": kw["budget"],
                               "stopping": kw["stopping_criterion"], "time_limit_sec": kw["time_limit_sec"],
                               "n_scen": len(scen), "pop": kw["pop_size"]})

            best_seen = {a: -np.inf for a, _ in algos}
            finished = {a: None for a, _ in algos}
            while any(v is None for v in finished.values()):
                for name, it in algos:
                    if finished[name] is not None:
                        continue
                    try:
                        st = next(it)
                    except StopIteration:
                        finished[name] = finished[name] or {}
                        continue
                    ev = []
                    for e in st["ls_events"]:
                        ev.append({"rank": e["rank"] + 1, "before": e["f_before"] / 1e6, "after": e["f_after"] / 1e6,
                                   "used": e["used"], "n_steps": e["n_steps"], "n_worse": e["n_worse"],
                                   "n_aspiration": e["n_aspiration"], "n_blocked": e["n_blocked"],
                                   "steps": [{"kind": s["kind"], "msg": step_message(s)} for s in e["steps"]]})
                    payload = {"algo": name, "gen": st["gen"], "evals": st["evals"], "best": st["best_f"] / 1e6,
                               "fit": sorted([float(v) / 1e6 for v in st["fit"]], reverse=True),
                               "ls": ev, "ls_calls": st["ls_calls"], "runtime": st["runtime"]}
                    improved = st["best_f"] > best_seen[name] + 1e-9
                    if improved:
                        best_seen[name] = st["best_f"]
                        payload["plan"] = _plan_for(name, st["best_x"], scen, ev_ml)
                    self.sse("gen", payload)
                    head = (f"[{TAG[name]}] thế hệ {st['gen']:>3} | {st['runtime']:>5.1f}/{kw['time_limit_sec']:g} giây | "
                            if kw["stopping_criterion"] == "time" else
                            f"[{TAG[name]}] thế hệ {st['gen']:>3} | {st['evals']:>5}/{kw['budget']} đánh giá | ")
                    if improved:
                        pl = payload["plan"]
                        log(head + f"Z tốt nhất = {payload['best']:.3f} tr  ▲ "
                            f"(LN TB {pl['mean']:.2f}, MAD {pl['mad']:.2f})", name)
                    elif st["gen"] % 25 == 0:
                        log(head + f"Z tốt nhất = {payload['best']:.3f} tr", "dim")
                    for e in ev:
                        g = e["after"] - e["before"]
                        log(f"    ↳ [{TAG[name]}] tìm kiếm cục bộ, cá thể hạng {e['rank']}: {e['before']:.3f} → {e['after']:.3f} tr"
                            + (f"  (+{g:.3f})" if g > 1e-9 else "  (không cải thiện)")
                            + f", dùng {e['used']} lần, đi {e['n_steps']} bước, {e['n_worse']} bước tệ hơn",
                            name if g > 1e-9 else "dim")
                        for stp in e["steps"]:
                            log(f"         • {stp['msg']}", "dim" if stp["kind"] == "blocked" else name)
                    if st["done"]:
                        finished[name] = {"best": st["best_f"] / 1e6, "gen": st["gen"], "evals": st["evals"],
                                          "runtime": st["runtime"], "ls_calls": st["ls_calls"]}
                if delay:
                    time.sleep(delay)
            self.sse("done", finished)
            log("-" * 72, "head")
            for a, r in finished.items():
                if r:
                    log(f"KẾT QUẢ [{TAG[a]}] Z = {r['best']:.3f} triệu | {r['gen']} thế hệ | {r['evals']} đánh giá "
                        f"(tìm kiếm cục bộ: {r['ls_calls']}) | {r['runtime']:.1f} s", a)
            if finished.get("spga"):
                for a in ("ts", "sa", "ml"):
                    if finished.get(a):
                        d = finished[a]["best"] - finished["spga"]["best"]
                        log(f"{TAG[a]} {'hơn' if d >= 0 else 'kém'} SPGA gốc: {d:+.3f} triệu "
                            f"({d / abs(finished['spga']['best']):+.2%})", "bold")
            log("-" * 72, "head")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            log("(trình duyệt đã dừng lần chạy)", "dim")  # trình duyệt đã dừng
        finally:
            mdl.K_RISK, mdl.SIGMA = DEFAULT_MODEL["K_RISK"], DEFAULT_MODEL["SIGMA"]
            mdl.MU = DEFAULT_MODEL["MU"].copy()
            mdl.Q_MAX = 1.5 * mdl.MU
            RUN_LOCK.release()

    # ------------------------------------------------------------ chạy lại main.py
    def rerun(self):
        if not RUN_LOCK.acquire(blocking=False):
            return self.send_json({"error": "Đang có một lần chạy khác"}, 409)
        try:
            self.start_sse()
            self.sse("log", {"line": "Đang chạy: python main.py --quick (khoảng 1 phút)..."})
            env = dict(os.environ, PYTHONIOENCODING="utf-8")
            p = subprocess.Popen([sys.executable, "-u", "main.py", "--quick"], cwd=HERE,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                                 text=True, encoding="utf-8", errors="replace")
            log("CHẠY LẠI THỰC NGHIỆM: python main.py --quick", "head")
            for line in p.stdout:
                print("   " + line.rstrip(), flush=True)
                self.sse("log", {"line": line.rstrip()})
            p.wait()
            self.sse("done", {"code": p.returncode})
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass
        finally:
            RUN_LOCK.release()


    # ------------------------------------------------------------ kiểm định lặp (multi-run benchmark)
    def benchmark(self, q):
        if not RUN_LOCK.acquire(blocking=False):
            return self.send_json({"error": "Đang có một lần chạy khác"}, 409)
        try:
            self.start_sse()
            n_runs = max(fnum(q, "n_runs", int), 1)
            budget = max(fnum(q, "budget", int), 100)
            stopping = q.get("stopping", ["evaluations"])[0]
            stopping = stopping if stopping in ("evaluations", "time") else "evaluations"
            time_limit_sec = max(fnum(q, "time_limit_sec"), 1.0)
            stop_kw = dict(stopping_criterion=stopping, time_limit_sec=time_limit_sec)
            k = min(max(fnum(q, "k"), 0.0), 1.0)
            sigma = max(fnum(q, "sigma"), 0.0)
            mu = np.array([fnum(q, "mu1"), fnum(q, "mu2"), fnum(q, "mu3")])
            n_scen = max(fnum(q, "n_scen", int), 2)
            scen_seed = fnum(q, "scen_seed", int)
            codes = selected_algos(q)
            labels = [FULL_LABEL[c] for c in codes]
            seeds = list(range(1, n_runs + 1))
            base = FULL_LABEL["spga"] if "spga" in codes else labels[0]
            h0 = q.get("h0", [""])[0]
            h1 = q.get("h1", [""])[0]
            alpha = min(max(fnum(q, "alpha"), 0.0), 1.0) or 0.05

            mdl.K_RISK, mdl.SIGMA = k, sigma
            mdl.MU = mu
            mdl.Q_MAX = 1.5 * mdl.MU
            scen = mdl.sample_scenarios(n_scen, scen_seed)
            evaluate = functools.partial(mdl.evaluate, scenarios=scen)
            ev_ml = matheuristics.LpEvaluator(k=k, sigma=sigma, scenarios=scen) if "ml" in codes else None
            hist_grid = np.linspace(0, budget, 101).astype(int)
            milp_launch(k=k, sigma=sigma, mu=mu, n_scen=n_scen, scen_seed=scen_seed,
                        time_limit=fnum(q, "time_limit", int))

            log("=" * 72, "head")
            stop_desc = (f"budget {budget}" if stopping == "evaluations" else f"thời gian {time_limit_sec:g} giây")
            log(f"KIỂM ĐỊNH LẶP: {n_runs} seed × {' + '.join(labels)}  (dừng theo {stop_desc}, k={k}, σ={sigma:g})", "head")
            log("=" * 72, "head")
            self.sse("start", {"algorithms": labels, "n_runs": n_runs, "seeds": seeds, "budget": budget,
                               "stopping": stopping, "time_limit_sec": time_limit_sec,
                               "hypothesis": {"h0": h0, "h1": h1, "alpha": alpha}})

            fitness = {lab: [] for lab in labels}
            mean_profit = {lab: [] for lab in labels}
            mad = {lab: [] for lab in labels}
            runtime = {lab: [] for lab in labels}
            total_evaluations = {lab: [] for lab in labels}
            evals_per_sec = {lab: [] for lab in labels}
            time_history = {lab: [] for lab in labels}
            runs = []

            for i, s in enumerate(seeds):
                self.sse("run_start", {"seed": s, "i": i, "n": n_runs})
                algos = []
                for c in codes:
                    if c == "ml":
                        algos.append((c, ga.run_ga_iter(ev_ml.evaluate, "none", s, budget,
                                                        repair_fn=matheuristics.repair_ml,
                                                        n_genes=matheuristics.N_GENES_ML, **stop_kw)))
                    else:
                        algos.append((c, ga.run_ga_iter(evaluate, LS_METHOD[c], s, budget, **stop_kw)))
                best_seen = {c: -np.inf for c, _ in algos}
                finished = {c: None for c, _ in algos}
                final_state = {c: None for c, _ in algos}
                run_plan = {lab: None for lab in labels}
                while any(v is None for v in finished.values()):
                    for c, it in algos:
                        if finished[c] is not None:
                            continue
                        try:
                            st = next(it)
                        except StopIteration:
                            finished[c] = finished[c] or {}
                            continue
                        lab = FULL_LABEL[c]
                        payload = {"seed": s, "algo": lab, "gen": st["gen"], "evals": st["evals"],
                                   "best": st["best_f"] / 1e6, "ls_calls": st["ls_calls"],
                                   "runtime": st["runtime"]}
                        if st["best_f"] > best_seen[c]:
                            best_seen[c] = st["best_f"]
                            payload["plan"] = _plan_for(c, st["best_x"], scen, ev_ml)
                            run_plan[lab] = payload["plan"]
                        self.sse("gen", payload)
                        if st["done"]:
                            finished[c] = {"best": st["best_f"], "gen": st["gen"], "evals": st["evals"],
                                           "runtime": st["runtime"], "ls_calls": st["ls_calls"]}
                            final_state[c] = st
                run_detail = {"seed": s, "algorithms": {}}
                for c in codes:
                    lab = FULL_LABEL[c]
                    st = final_state[c]
                    if c == "ml":
                        f, mp, md = ev_ml.evaluate(st["best_x"])
                    else:
                        f, mp, md = evaluate(st["best_x"])
                    fitness[lab].append(f)
                    mean_profit[lab].append(mp)
                    mad[lab].append(md)
                    rt_c = st["runtime"]
                    ev_c = st["evals"]
                    runtime[lab].append(rt_c)
                    total_evaluations[lab].append(ev_c)
                    evals_per_sec[lab].append(ev_c / rt_c if rt_c > 0 else 0.0)
                    time_history[lab].append(st["time_history"])
                    hist = [[int(hist_grid[j]), float(v)] for j, v in enumerate(_resample(st["history"], hist_grid))]
                    run_detail["algorithms"][lab] = {"fitness": f, "mean_profit": mp, "mad": md,
                                                     "runtime": rt_c, "history": hist,
                                                     "total_evaluations": ev_c,
                                                     "evals_per_sec": ev_c / rt_c if rt_c > 0 else 0.0,
                                                     "time_history": st["time_history"],
                                                     "plan": run_plan.get(lab)}
                runs.append(run_detail)
                best_lab = max(labels, key=lambda lb: run_detail["algorithms"][lb]["fitness"])
                # gửi kèm TOÀN BỘ lịch sử hội tụ + phương án của run này cho UI xem lại
                self.sse("run_done", {"seed": s, "i": i, "n": n_runs,
                                      "best_z": run_detail["algorithms"][best_lab]["fitness"] / 1e6,
                                      "best_algo": best_lab,
                                      "algorithms": {lb: {"fitness": run_detail["algorithms"][lb]["fitness"],
                                                          "mean_profit": run_detail["algorithms"][lb]["mean_profit"],
                                                          "mad": run_detail["algorithms"][lb]["mad"],
                                                          "runtime": run_detail["algorithms"][lb]["runtime"],
                                                          "history": run_detail["algorithms"][lb]["history"],
                                                          "plan": run_detail["algorithms"][lb]["plan"]}
                                                     for lb in labels}})
                log(f"  seed {s:>2}/{n_runs}: " + "   ".join(
                    f"{lb}: {run_detail['algorithms'][lb]['fitness'] / 1e6:.3f} tr" for lb in labels), "dim")

            leaderboard = sorted(
                [{"seed": rd["seed"],
                  "algorithm": max(labels, key=lambda lb: rd["algorithms"][lb]["fitness"]),
                  "z": max(rd["algorithms"][lb]["fitness"] for lb in labels)} for rd in runs],
                key=lambda x: -x["z"])[:3]

            benchmark = {"algorithms": labels, "base": base, "n_runs": n_runs, "seeds": seeds,
                         "fitness": fitness, "mean_profit": mean_profit, "mad": mad, "runtime": runtime,
                         "total_evaluations": total_evaluations, "evals_per_sec": evals_per_sec,
                         "time_history": time_history,
                         "leaderboard": leaderboard, "runs": runs,
                         "hypothesis": {"h0": h0, "h1": h1, "alpha": alpha},
                         "params": {"budget": budget, "k": k, "sigma": sigma, "n_scen": n_scen, "scen_seed": scen_seed,
                                    "stopping": stopping, "time_limit_sec": time_limit_sec}}
            BENCHMARK_CACHE.clear()
            BENCHMARK_CACHE.update(benchmark)
            stats = exporter.compute_benchmark_stats(benchmark)
            conclusion = exporter.generate_hypothesis_conclusion(stats, {"h0": h0, "h1": h1, "alpha": alpha})
            self.sse("done", {"summary": stats, "leaderboard": leaderboard, "conclusion": conclusion,
                              "hypothesis": {"h0": h0, "h1": h1, "alpha": alpha}})
            log("-" * 72, "head")
            for a, v in stats["per_alg"].items():
                log(f"  {a}: TB {v['mean'] / 1e6:.3f} tr ± {v['std'] / 1e6:.3f} (median {v['median'] / 1e6:.3f})", a)
            if stats["friedman"]:
                log(f"  Friedman p = {stats['friedman']['p']:.4f}", "bold")
            for w in stats["wilcoxon"]:
                log(f"  Wilcoxon {w['a']} vs {w['b']}: p_Holm = {w['p_holm']:.4f}", "bold")
            log("-" * 72, "head")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            log("(trình duyệt đã dừng kiểm định)", "dim")
        finally:
            mdl.K_RISK, mdl.SIGMA = DEFAULT_MODEL["K_RISK"], DEFAULT_MODEL["SIGMA"]
            mdl.MU = DEFAULT_MODEL["MU"].copy()
            mdl.Q_MAX = 1.5 * mdl.MU
            RUN_LOCK.release()

    # ------------------------------------------------------------ phân tích độ nhạy
    def sensitivity(self, q):
        if not RUN_LOCK.acquire(blocking=False):
            return self.send_json({"error": "Đang có một lần chạy khác"}, 409)
        try:
            self.start_sse()
            k_min = min(max(fnum(q, "sens_k_min"), 0.0), 1.0)
            k_max = min(max(fnum(q, "sens_k_max"), 0.0), 1.0)
            k_steps = max(fnum(q, "sens_k_steps", int), 2)
            s_min = max(fnum(q, "sens_sigma_min"), 100.0)
            s_max = max(fnum(q, "sens_sigma_max"), s_min)
            s_steps = max(fnum(q, "sens_sigma_steps", int), 2)
            budget = max(fnum(q, "sens_budget", int), 100)
            sens_runs = max(fnum(q, "sens_runs", int), 1)
            n_scen = max(fnum(q, "n_scen", int), 2)
            scen_seed = fnum(q, "scen_seed", int)
            codes = selected_algos(q)
            labels = [FULL_LABEL[c] for c in codes]
            mu = DEFAULT_MODEL["MU"].copy()

            if k_max < k_min:
                k_max, k_min = k_min, k_max
            k_grid = np.linspace(k_min, k_max, k_steps)
            sigma_grid = np.linspace(s_min, s_max, s_steps)
            nominal_k = float(k_grid[int(np.argmin(np.abs(k_grid - 0.5)))])
            nominal_sigma = float(sigma_grid[int(np.argmin(np.abs(sigma_grid - DEFAULT_MODEL["SIGMA"])))])

            total = len(k_grid) * len(sigma_grid) * len(labels)
            log("=" * 72, "head")
            log(f"PHÂN TÍCH ĐỘ NHẠY: k ∈ [{k_min},{k_max}] ({k_steps}) × σ ∈ [{s_min:g},{s_max:g}] ({s_steps}) "
                f"× {len(labels)} thuật toán = {total} ô", "head")
            self.sse("start", {"algorithms": labels, "k_grid": k_grid.tolist(),
                               "sigma_grid": sigma_grid.tolist(), "total": total})

            F = {lab: np.zeros((len(k_grid), len(sigma_grid))) for lab in labels}
            P = {lab: np.zeros((len(k_grid), len(sigma_grid))) for lab in labels}
            M = {lab: np.zeros((len(k_grid), len(sigma_grid))) for lab in labels}
            done = 0
            for ki, k in enumerate(k_grid):
                for si, sigma in enumerate(sigma_grid):
                    mdl.K_RISK = k
                    mdl.SIGMA = sigma
                    mdl.MU = mu
                    mdl.Q_MAX = 1.5 * mu
                    scen = mdl.sample_scenarios(n_scen, scen_seed)
                    evaluate = functools.partial(mdl.evaluate, scenarios=scen)
                    ev_ml = matheuristics.LpEvaluator(k=k, sigma=sigma, scenarios=scen) if "ml" in codes else None
                    for code, lab in zip(codes, labels):
                        best_f = -np.inf
                        best_mp = best_md = 0.0
                        for r in range(sens_runs):
                            if code == "ml":
                                bx, _, _ = matheuristics.run_matheur(ev_ml.evaluate, seed=1 + r, budget=budget)
                                f, mp, md = ev_ml.evaluate(bx)
                            else:
                                bx, _, _ = ga.run_ga(evaluate, LS_METHOD[code], seed=1 + r, budget=budget)
                                f, mp, md = evaluate(bx)
                            if f > best_f:
                                best_f, best_mp, best_md = f, mp, md
                        F[lab][ki, si] = best_f
                        P[lab][ki, si] = best_mp
                        M[lab][ki, si] = best_md
                    done += len(labels)
                    self.sse("cell", {"k": k, "sigma": sigma, "done": done, "total": total})
                log(f"  k={k:.2f}: xong {done}/{total} ô", "dim")

            sensitivity = {"algorithms": labels, "k_grid": k_grid.tolist(), "sigma_grid": sigma_grid.tolist(),
                           "nominal_k": nominal_k, "nominal_sigma": nominal_sigma,
                           "fitness": {lab: F[lab].tolist() for lab in labels},
                           "profit": {lab: P[lab].tolist() for lab in labels},
                           "mad": {lab: M[lab].tolist() for lab in labels}}
            metrics = exporter.compute_sensitivity_metrics(sensitivity)
            impact = exporter.build_sensitivity_impact(sensitivity)
            narrative = exporter.generate_sensitivity_narrative(sensitivity, metrics)
            sensitivity["metrics"] = metrics
            sensitivity["impact"] = impact
            sensitivity["narrative"] = narrative
            SENSITIVITY_CACHE.clear()
            SENSITIVITY_CACHE.update(sensitivity)
            SENSITIVITY_METRICS.clear()
            SENSITIVITY_METRICS.update(metrics)
            self.sse("done", {"metrics": metrics, "impact": impact, "narrative": narrative,
                              "k_grid": k_grid.tolist(), "sigma_grid": sigma_grid.tolist()})
            log("-" * 72, "head")
            for lab in labels:
                m = metrics[lab]
                log(f"  {lab}: đóng góp k = {m['contribution_k']:.1%}, σ = {m['contribution_sigma']:.1%}, "
                    f"ổn định = {m['robustness']:.2f}", lab)
            log("-" * 72, "head")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            log("(trình duyệt đã dừng phân tích)", "dim")
        finally:
            mdl.K_RISK, mdl.SIGMA = DEFAULT_MODEL["K_RISK"], DEFAULT_MODEL["SIGMA"]
            mdl.MU = DEFAULT_MODEL["MU"].copy()
            mdl.Q_MAX = 1.5 * mdl.MU
            RUN_LOCK.release()

    # ------------------------------------------------------------ xuất báo cáo Excel
    def export_excel(self, q):
        if not BENCHMARK_CACHE:
            return self.send_json({"error": "Chưa có kết quả kiểm định. Hãy chạy kiểm định trước."}, 400)
        os.makedirs(os.path.join(HERE, "output"), exist_ok=True)
        path = os.path.join(HERE, "output", "bao_cao_kiem_dinh.xlsx")
        try:
            exporter.export_single_tab_report(BENCHMARK_CACHE, SENSITIVITY_CACHE or None,
                                              BENCHMARK_CACHE.get("hypothesis", {}), path,
                                              milp=MILP_CACHE or None)
        except ImportError as e:
            return self.send_json({"error": f"Thiếu thư viện: {e}. Chạy: pip install openpyxl"}, 500)
        except Exception as e:
            return self.send_json({"error": f"Lỗi xuất Excel: {e}"}, 500)
        log(f"Đã xuất báo cáo Excel: {os.path.basename(path)}", "ok")
        return self.send_file(path, "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                              download="bao_cao_kiem_dinh.xlsx")

    # ------------------------------------------------------------ chi tiết 1 run (click-to-view)
    def benchmark_run(self, q):
        seed = fnum(q, "seed", int)
        for rd in BENCHMARK_CACHE.get("runs", []):
            if rd["seed"] == seed:
                return self.send_json(rd)
        return self.send_json({"error": "Không tìm thấy run này."}, 404)

    # ------------------------------------------------------------ cận tối ưu MILP (chạy nền)
    def milp_start(self, q):
        if MILP_THREAD is not None and MILP_THREAD.is_alive():
            return self.send_json({"running": True, "result": MILP_CACHE or None})
        k = min(max(fnum(q, "k"), 0.0), 1.0)
        sigma = max(fnum(q, "sigma"), 0.0)
        n_scen = max(fnum(q, "n_scen", int), 2)
        scen_seed = fnum(q, "scen_seed", int)
        time_limit = max(fnum(q, "time_limit", int), 20)
        mu = np.array([fnum(q, "mu1"), fnum(q, "mu2"), fnum(q, "mu3")])
        milp_launch(k, sigma, mu, n_scen, scen_seed, time_limit)
        return self.send_json({"running": True})


def load_results():
    out = os.path.join(HERE, "output")
    res = {"available": False}
    path = os.path.join(out, "results.csv")
    if not os.path.isfile(path):
        return res
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    res["available"] = True
    res["rows"] = [{k: r[k] for k in ("algorithm", "seed", "fitness", "mean_profit", "mad", "runtime_s",
                                      "test_fitness")} for r in rows]
    summ = {}
    for alg in sorted({r["algorithm"] for r in rows}):
        sub = [r for r in rows if r["algorithm"] == alg]
        f = np.array([float(r["fitness"]) for r in sub])
        summ[alg] = {"n": len(sub), "fitness_mean": f.mean() / 1e6, "fitness_std": f.std(ddof=1) / 1e6,
                     "fitness_min": f.min() / 1e6, "fitness_max": f.max() / 1e6,
                     "profit_mean": np.mean([float(r["mean_profit"]) for r in sub]) / 1e6,
                     "mad_mean": np.mean([float(r["mad"]) for r in sub]) / 1e6,
                     "runtime_mean": np.mean([float(r["runtime_s"]) for r in sub]),
                     "test_mean": np.mean([float(r["test_fitness"]) for r in sub]) / 1e6}
    res["summary"] = summ
    hpath = os.path.join(out, "history.csv")
    if os.path.isfile(hpath):
        grid = list(range(60, 6001, 60))
        curves = {}
        data = {}
        for r in csv.DictReader(open(hpath, encoding="utf-8")):
            data.setdefault((r["algorithm"], r["seed"]), []).append((int(r["evals"]), float(r["best_fitness"])))
        for (alg, _), pts in data.items():
            n = np.array([p[0] for p in pts]); f = np.array([p[1] for p in pts])
            idx = np.clip(np.searchsorted(n, grid, side="right") - 1, 0, len(f) - 1)
            curves.setdefault(alg, []).append(f[idx] / 1e6)
        res["curves"] = {"grid": grid, **{a: list(np.mean(c, axis=0)) for a, c in curves.items()}}
    spath = os.path.join(out, "ablation.csv")
    if os.path.isfile(spath):
        res["ablation"] = list(csv.DictReader(open(spath, encoding="utf-8")))
    tpath = os.path.join(out, "stats.txt")
    if os.path.isfile(tpath):
        res["stats"] = open(tpath, encoding="utf-8").read()
    res["images"] = sorted(n for n in os.listdir(out) if n.endswith(".png"))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    srv = ThreadingHTTPServer(("127.0.0.1", a.port), Handler)
    log(f"Server demo đang chạy tại http://localhost:{a.port}", "ok")
    log("Cửa sổ này sẽ hiện log thuật toán mỗi khi bấm ▶ Chạy trên trình duyệt.", "dim")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
