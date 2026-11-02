"""tests.py - kiểm thử nhanh (mục 4.5). Chạy: python tests.py"""
import functools
import numpy as np

import model as mdl
import ga

SPGA_SEED1 = 47236886.8875


def check(name, cond):
    print(f"[{'OK ' if cond else 'LỖI'}] {name}")
    assert cond, name


def same_dict(d1, d2):
    return all(np.array_equal(d1[k], d2[k]) for k in d1)


def main():
    rng = np.random.default_rng(0)
    scen = mdl.sample_scenarios(30, 2026)
    check("sample_scenarios trả về [30, 8, 3], không âm", scen.shape == (30, 8, 3) and (scen >= 0).all())
    check("cùng seed cho cùng kịch bản", np.allclose(scen, mdl.sample_scenarios(30, 2026)))

    xs = [mdl.repair(rng.random(mdl.N_GENES)) for _ in range(500)]
    check("nhiễm sắc thể 135 gen trong [0,1]", all(x.shape == (135,) and x.min() >= 0 and x.max() <= 1 for x in xs))

    ok_cap = True
    for x in xs:
        d = mdl.decode(x)
        need_m, need_a = mdl._loads(d)
        ok_cap &= (need_m <= d["K_m"][None] + d["X_m"] + 1e-6).all()
        ok_cap &= (need_a <= d["K_a"][None] + d["X_a"] + 1e-6).all()
        ok_cap &= np.allclose(d["Q"][mdl.C_MT[None].repeat(8, 0) == 0], 0)
    check("sau repair thỏa (2), (4) và sản lượng cặp không tương thích = 0", ok_cap)

    check("repair lũy đẳng: repair(repair(x)) = repair(x)",
          all(np.allclose(mdl.repair(x), x) for x in xs[:100]))

    rt = all(np.allclose(mdl.encode(mdl.decode(x)), x, atol=1e-9) or
             (lambda d1, d2: all(np.allclose(d1[k], d2[k]) for k in d1))(mdl.decode(x), mdl.decode(mdl.encode(mdl.decode(x))))
             for x in xs[:100])
    check("decode(encode(decode(x))) = decode(x)", rt)

    f, mean, mad = mdl.evaluate(xs[0], scen)
    check("evaluate: fitness = 0.5*mean - 0.5*MAD", abs(f - (0.5 * mean - 0.5 * mad)) < 1e-6 and mad >= 0)

    # nghiệm không đầu tư, không sản xuất: lợi nhuận MTO = 0, MTS bị phạt thiếu hàng
    zero = mdl.repair(np.zeros(mdl.N_GENES))
    h = mdl.scenario_profits(zero, scen)
    check("nghiệm rỗng có lợi nhuận thấp hơn nghiệm ngẫu nhiên",
          h.mean() < np.mean([mdl.evaluate(x, scen)[1] for x in xs[:50]]))

    # ------------------------------------------------------- ngân sách, lịch sử, tái lập
    ev = functools.partial(mdl.evaluate, scenarios=scen)
    for method in ga.LS_METHODS:
        bx, hist, _ = ga.run_ga(ev, method, seed=1, budget=600)
        check(f"run_ga({method}) dùng đúng ngân sách 600 lần evaluate", hist[-1][0] == 600)
        check(f"run_ga({method}) lịch sử best không giảm",
              all(hist[i][1] <= hist[i + 1][1] for i in range(len(hist) - 1)))
        check(f"run_ga({method}) best_x khớp fitness cuối", abs(ev(bx)[0] - hist[-1][1]) < 1e-6)
        a = ga.run_ga(ev, method, seed=3, budget=600)[1]
        b = ga.run_ga(ev, method, seed=3, budget=600)[1]
        check(f"run_ga({method}) cùng seed cho cùng kết quả", a == b)

    for method in ("tabu", "sa"):
        calls = {"n": 0}

        def counted(x):
            calls["n"] += 1
            return ev(x)

        ga.run_ga(counted, method, seed=2)
        check(f"{method}: số lần gọi evaluate thật đúng 6000 (kể cả 30 lần ước lượng T0 của SA)", calls["n"] == 6000)

    first = next(ga.run_ga_iter(ev, "sa", seed=1))
    check("SA: 30 lần ước lượng T0 đã tính vào ngân sách ngay từ thế hệ 0",
          first["evals"] == ga.POP_SIZE + ga.SA_T0_SAMPLES)

    # ------------------------------------------------------- Lamarckian
    for method in ("tabu", "sa", "hc"):
        ok_mono, ok_write, n_ev = True, True, 0
        for st in ga.run_ga_iter(ev, method, seed=4, budget=1500):
            for e in st["ls_events"]:
                n_ev += 1
                ok_mono &= e["f_after"] >= e["f_before"]
                ok_write &= abs(st["fit"][e["idx"]] - e["f_after"]) < 1e-6
                ok_write &= abs(ev(st["pop"][e["idx"]])[0] - e["f_after"]) < 1e-6
        check(f"{method}: Lamarckian không làm giảm fitness của cá thể được ghi đè ({n_ev} lần)", n_ev > 0 and ok_mono)
        check(f"{method}: nghiệm ghi vào quần thể đúng là nghiệm có Z = f_after", ok_write)

    # ------------------------------------------------------- bước ngược
    ok_exact, ok_shift, n_shift, ok_rev = True, True, 0, True
    d0 = mdl.decode(xs[0])
    for move in ga.MOVES:
        ok_rev &= ga.reverse_move(ga.reverse_move(move)) == move
        ok_rev &= ga.reverse_move(move) in ga.MOVES
        d1 = ga.apply_move(d0, move)
        if d1 is None:
            continue
        d2 = ga.apply_move(d1, ga.reverse_move(move))
        if move[0] == "shift":
            # bước ngược chuyển 10% của ô đích, nên chỉ quay về gần đúng
            p, t, src, dst = move[1]
            n_shift += 1
            limit = 0.11 * max(d0["Q"][p, src, t], d0["Q"][p, dst, t]) + 1e-6
            ok_shift &= abs(d2["Q"][p, src, t] - d0["Q"][p, src, t]) <= limit
        else:
            ok_exact &= d2 is not None and same_dict(d0, d2)
    check("bước ngược của bước ngược là bước ban đầu, và nằm trong MOVES", ok_rev)
    check("K/X: áp một bước rồi áp bước ngược thì quay về đúng nghiệm cũ", ok_exact)
    check(f"shift: áp một bước rồi áp bước ngược thì về gần nghiệm cũ ({n_shift} bước)", n_shift > 0 and ok_shift)

    # ------------------------------------------------------- tabu
    tabu = ga.TabuList(tenure=2)
    move = ("X_m", (0, 0), +1)
    tabu.tick()
    tabu.forbid(move)
    rev = ga.reverse_move(move)
    tabu.tick()
    first_it = tabu.is_tabu(rev)
    tabu.tick()
    second_it = tabu.is_tabu(rev)
    tabu.tick()
    check("tabu: bước ngược bị cấm đúng tenure vòng lặp rồi hết cấm", first_it and second_it and not tabu.is_tabu(rev))
    check("tabu: bản thân bước vừa đi không bị cấm", not tabu.is_tabu(move))

    # ép tình huống (cấm mọi bước, chỉnh Z kỷ lục) để nhánh cấm và aspiration chắc chắn chạy
    x0 = xs[1]
    d = mdl.decode(x0)
    valid = [m for m in ga.MOVES if ga.apply_move(d, m) is not None]
    forbid_all = ga.TabuList(tenure=100)
    forbid_all.tick()
    for m in valid:
        forbid_all.forbid(ga.reverse_move(m))
    counter = ga._Counter(ev, 6000)
    fx = counter(x0)
    counter.best_f = np.inf                          # không bước nào đạt được aspiration
    trace = []
    _, fb, used = ga.tabu_search(x0, fx, counter, np.random.default_rng(1), 20, ga.MOVES, forbid_all, trace)
    check("tabu: mọi ứng viên bị cấm thì không đi bước nào, bốc lại tối đa 3 lần",
          fb == fx and used == 20 and all(s["kind"] == "blocked" for s in trace))
    counter = ga._Counter(ev, 6000)
    fx = counter(x0)
    counter.best_f = -np.inf                         # bước đầu tiên nào cũng đạt aspiration
    trace = []
    ga.tabu_search(x0, fx, counter, np.random.default_rng(1), 5, ga.MOVES, forbid_all, trace)
    check("tabu: bước bị cấm vẫn được đi khi đạt aspiration", [s["kind"] for s in trace] == ["aspiration"])

    # ------------------------------------------------------- SA
    counter = ga._Counter(ev, 6000)
    fx = counter(x0)
    trace = []
    x_best, f_best, used = ga.simulated_annealing(x0, fx, counter, np.random.default_rng(1), 20, ga.MOVES, 1e6, trace)
    check("SA: trả về nghiệm tốt nhất gặp được, không phải nghiệm cuối",
          used == 20 and f_best >= max([fx] + [s["f_to"] for s in trace]) - 1e-9)
    Ts = [s["T"] for s in trace]
    check("SA: nhiệt độ giảm dần và không dưới 0.01·T0",
          all(a >= b for a, b in zip(Ts, Ts[1:])) and min(Ts) >= 1e6 * 0.01 - 1e-6)

    # ------------------------------------------------------- SPGA gốc phải giữ nguyên
    bx, hist, _ = ga.run_ga(ev, "none", seed=1)
    check(f"SPGA gốc seed 1 = {SPGA_SEED1} (hiện {ev(bx)[0]:.4f})", abs(ev(bx)[0] - SPGA_SEED1) < 5e-5)

    # ------------------------------------------------------- demo phải khớp với ga.py
    import demo_server
    check("demo_server import được và dùng đúng tham số của ga.py",
          demo_server.DEFAULTS["ts_n_cand"] == ga.TS_N_CAND and demo_server.DEFAULTS["sa_p0"] == ga.SA_P0)
    print("Tất cả kiểm thử đều đạt.")


if __name__ == "__main__":
    main()
