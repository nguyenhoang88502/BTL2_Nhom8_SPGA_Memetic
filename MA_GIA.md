# Mã giả: SPGA gốc, GA-TS, GA-SA và GA-LP (Nhóm 8)

Ký hiệu dùng chung:

| Ký hiệu | Ý nghĩa | Giá trị |
|---|---|---|
| x | 1 nghiệm = vector 135 gen trong [0,1] (số máy mua K, số máy thuê X, sản lượng Q) | |
| Z(x) | (1−k)·lợi nhuận TB − k·MAD trên 30 kịch bản nhu cầu | k = 0,5 |
| B | ngân sách số lần tính Z (gồm cả lần tính trong tìm kiếm cục bộ) | 6000 |
| N | kích thước quần thể | 30 |
| p_c, p_m | xác suất lai ghép / đột biến mỗi gen | 0,9 / 0,05 |
| F, L, E | tìm kiếm cục bộ: cứ F thế hệ, cho L cá thể tốt nhất, tối đa E lần tính Z mỗi cá thể | 10, 3, 20 |
| 𝓝(x) | 174 bước lân cận: K ±1 (14), X ±1 theo quý (112), chuyển 10% sản lượng (48) | |
| x_LP | 1 nghiệm của GA-LP = vector 63 gen (chỉ máy: mua K_m, K_a; thuê X_m, X_a theo quý) | |

Ba thuật toán đầu **giống hệt nhau ở Thuật toán 1**, chỉ khác dòng 14 (`LOCAL`).
GA-LP (Thuật toán 5) dùng lại khung GA này với `LOCAL = không`, nhưng nhiễm sắc thể chỉ còn **63 gen máy móc** và mỗi lần chấm điểm là **1 lần giải LP** (Thuật toán 6).

---

## Thuật toán 1 – Khung GA chung

```
Input : hàm Z, ngân sách B, LOCAL ∈ {không, TABU, SA}
Output: nghiệm tốt nhất x*

 1  P ← N vector ngẫu nhiên U(0,1)^135;  với mỗi x ∈ P: x ← REPAIR(x); tính Z(x)
 2  nếu LOCAL = SA: T0 ← ƯỚC_LƯỢNG_T0(P)                       ▹ Thuật toán 4, tốn 30 lần tính Z
 3  g ← 0
 4  while còn ngân sách do
 5      g ← g + 1
 6      P' ← {cá thể tốt nhất của P}                          ▹ elitism
 7      while |P'| < N và còn ngân sách do
 8          a ← TOURNAMENT(P, 2);  b ← TOURNAMENT(P, 2)
 9          (c1, c2) ← LAI_GHÉP_HYBRID(a, b) với xác suất p_c, ngược lại (a, b)
10          với mỗi c ∈ {c1, c2}: c ← REPAIR(ĐỘT_BIẾN(c, p_m)); tính Z(c); thêm c vào P'
11      P ← P'
12      if LOCAL ≠ không và g mod F = 0 then
13          tabu ← danh sách rỗng                                ▹ dùng CHUNG cho L cá thể (chỉ GA-TS)
14          với L cá thể tốt nhất x_i của P:
                (y, Z(y)) ← TABU(x_i, tabu) hoặc SA(x_i, T0)  ▹ Thuật toán 2 / 3
15              x_i ← y                                         ▹ Lamarckian: ghi nghiệm tốt nhất gặp được
16  return x* = nghiệm có Z lớn nhất từng được tính
```

- `LAI_GHÉP_HYBRID`: chọn ngẫu nhiên 1 trong 4 kiểu (1 điểm, 2 điểm, uniform, arithmetic).
- `ĐỘT_BIẾN`: mỗi gen có xác suất p_m bị thay bằng giá trị U(0,1) mới.
- `REPAIR`: sửa nghiệm để thỏa ràng buộc công suất (mục 3.3 của báo cáo).
- SPGA gốc = Thuật toán 1 với `LOCAL = không` (bỏ dòng 2, 12–15).

---

## Thuật toán 2 – Tabu Search (GA-TS)

```
Input : nghiệm x, danh sách tabu dùng chung, n_cand = 5, tenure = 7
Output: nghiệm tốt nhất gặp được x_best

 1  x_best ← x;  dùng ← 0
 2  while dùng < E và còn ngân sách do
 3      đồng_hồ ← đồng_hồ + 1
 4      lặp tối đa 1 + 3 lần:
 5          C ← bốc ngẫu nhiên n_cand bước hợp lệ trong 𝓝(x);  tính Z của từng bước;  dùng += |C|
 6          A ← { bước ∈ C | không bị cấm }
 7              ∪ { bước ∈ C | bị cấm nhưng Z > kỷ lục toàn cục }      ▹ aspiration
 8          nếu A ≠ ∅: m ← bước có Z lớn nhất trong A;  thoát vòng lặp
 9      nếu không chọn được m: dừng
10      x ← áp m lên x                                ▹ đi tới m KỂ CẢ khi Z giảm
11      cấm BƯỚC_NGƯỢC(m) đến đồng_hồ + tenure
12      nếu Z(x) > Z(x_best): x_best ← x
13  return x_best
```

- `BƯỚC_NGƯỢC`: K/X đổi dấu (+1 ↔ −1); chuyển sản lượng src→dst thành dst→src.
- **Điểm mới:** danh sách tabu được tạo mới ở dòng 13 của Thuật toán 1 và **dùng chung** cho 3 cá thể. Cá thể sau không được đảo ngược bước mà cá thể trước vừa đi, nên 3 lần tìm kiếm tỏa ra 3 hướng khác nhau.

---

## Thuật toán 3 – Simulated Annealing (GA-SA)

```
Input : nghiệm x, nhiệt độ ban đầu T0
Output: nghiệm tốt nhất gặp được x_best

 1  α ← 0,01^(1/E);  x_best ← x
 2  for i = 1 .. E (khi còn ngân sách) do
 3      y ← áp 1 bước ngẫu nhiên hợp lệ trong 𝓝(x) lên x;  tính Z(y)
 4      T ← T0 · α^i                                  ▹ làm nguội cấp số nhân, T_E = 0,01·T0
 5      if Z(y) ≥ Z(x) then nhận
 6      else nhận với xác suất exp((Z(y) − Z(x)) / T)  ▹ luật Metropolis
 7      nếu nhận: x ← y;  nếu Z(x) > Z(x_best): x_best ← x
 8  return x_best
```

## Thuật toán 4 – Ước lượng T0 (chạy 1 lần, chỉ cho GA-SA)

```
 1  Δ ← ∅
 2  lặp 30 lần: chọn ngẫu nhiên x ∈ P; y ← 1 bước ngẫu nhiên từ x; tính Z(y)
 3              nếu Z(y) < Z(x): thêm Z(x) − Z(y) vào Δ
 4  return T0 = −median(Δ) / ln(0,5)               ▹ bước tệ điển hình được nhận với xác suất 50%
```

---

## Thuật toán 5 – GA-LP (Matheuristics: GA máy móc + LP sản lượng)

GA-LP dùng lại khung GA ở Thuật toán 1 với `LOCAL = không`, nhưng khác ở 3 chỗ:
nhiễm sắc thể chỉ còn 63 gen máy móc (bỏ 72 gen sản lượng), `REPAIR_ML` chỉ cần clip [0,1],
và chấm điểm bằng giải LP thay vì đọc sản lượng từ gen.

```
Input : hàm Z_LP, ngân sách B
Output: nghiệm tốt nhất x*

 1  P ← N vector ngẫu nhiên U(0,1)^63;  với mỗi x ∈ P: x ← REPAIR_ML(x); tính Z_LP(x)
 2  g ← 0
 3  while còn ngân sách do
 4      g ← g + 1
 5      P' ← {cá thể tốt nhất của P}                          ▹ elitism
 6      while |P'| < N và còn ngân sách do
 7          a ← TOURNAMENT(P, 2);  b ← TOURNAMENT(P, 2)
 8          (c1, c2) ← LAI_GHÉP_HYBRID(a, b) với xác suất p_c, ngược lại (a, b)
 9          với mỗi c ∈ {c1, c2}: c ← REPAIR_ML(ĐỘT_BIẾN(c, p_m)); tính Z_LP(c); thêm c vào P'
10      P ← P'
11  return x* = nghiệm có Z lớn nhất từng được tính
```

- `DECODE_ML`: 63 gen → số máy nguyên `K_m` (3), `K_a` (4), `X_m` (8×3), `X_a` (8×4).
- `REPAIR_ML`: clip từng gen về [0,1] — mọi tổ hợp máy đều khả thi, nên không cần sửa công suất như `REPAIR` của SPGA.

## Thuật toán 6 – Z_LP(x): giải LP phân bổ sản lượng (đánh giá 1 phương án máy)

```
Input : phương án máy x (63 gen)
Output: (Z, mean, mad)

 1  (K_m, K_a, X_m, X_a) ← DECODE_ML(x)
 2  b ← vế phải ràng buộc năng lực = K + X (từng quý, từng máy)
 3  giải LP bằng HiGHS với biến Q, Sp, Sm, sold, u, mean:
        max   (1−k)·Σ_n h_n − k·Σ_n u_n
        với   h_n = F0 + Σ_p (1+r)^(−p−1)·(doanh thu − tồn kho/thiếu − thuê máy) − đầu tư ròng
        sao cho:
          năng lực tester:   Σ_t Q[p,m,t] / CAP_M[m,t] ≤ K_m[m] + X_m[p,m]
          năng lực handler:  Σ_{m,t} Q[p,m,t] / CAP_H[a] ≤ K_a[a] + X_a[p,a]
          cân bằng tồn kho SP1;  bán MTO ≤ nhu cầu
          u_n ≥ h_n − mean(h);  u_n ≥ mean(h) − h_n         ▹ tuyến tính hóa MAD
 4  nếu LP không có nghiệm tối ưu: return Z_LP với Q = 0      ▹ fallback (luôn khả thi)
 5  trích Q* (sản lượng tối ưu)
 6  làm sạch: MTO sản xuất đúng mức bán lớn nhất qua các kịch bản
 7  tính lại Z bằng SCENARIO_PROFITS trên phương án sạch       ▹ khớp SPGA/GA-TS
 8  return (Z, mean, mad)
```

- `Q[p,m,t]` là quyết định **here-and-now** (chung cho mọi kịch bản); tồn kho `Sp/Sm`, bán MTO `sold` và độ lệch `u` là **theo từng kịch bản n**.
- LP được **xây dựng MỘT LẦN** (ma trận ràng buộc thưa); mỗi lần đánh giá chỉ cập nhật vế phải `b` rồi giải lại (~5–15 ms bằng HiGHS, nhanh hơn nhiều so với PuLP/CBC phải ghi file tạm mỗi lần).
- Fitness **tính lại bằng `scenario_profits`** (mô hình dùng chung) chứ không lấy thẳng từ mục tiêu LP, vì LP cho phép "bán dưới nhu cầu" để hạ MAD — quyền linh hoạt mà 3 thuật toán kia không có; tính lại để so sánh công bằng.

---

## So sánh nhanh

| | SPGA gốc | GA-TS | GA-SA |
|---|---|---|---|
| Tìm kiếm cục bộ | không | Tabu Search | Simulated Annealing |
| Mỗi vòng xem | – | 5 bước ứng viên, đi bước tốt nhất | 1 bước ngẫu nhiên |
| Nhận bước tệ hơn | – | Có (tất định: bước tốt nhất không bị cấm) | Có (ngẫu nhiên: xác suất giảm theo T) |
| Bộ nhớ | – | Danh sách tabu dùng chung cho 3 cá thể | Không |
| Ngân sách | 6000 | 6000 (gồm cả tìm kiếm cục bộ) | 6000 (gồm 30 lần ước lượng T0) |

GA-LP khác SPGA ở **cách mã hóa và cách chấm điểm**, không ở tìm kiếm cục bộ:

| | SPGA gốc | GA-LP |
|---|---|---|
| Nhiễm sắc thể | 135 gen (máy + sản lượng Q) | 63 gen (chỉ máy móc) |
| Sửa khả thi | REPAIR: co sản lượng theo công suất | REPAIR_ML: clip [0,1] |
| Chấm điểm | đọc sản lượng từ gen → tính Z | giải LP tìm Q* tối ưu → tính Z |
| Không gian tìm kiếm | máy × sản lượng (lớn) | chỉ máy (gọn hơn) |
| 1 lần đánh giá | rẻ | ~5–15 ms (1 lần giải LP) |
| Ngân sách | 6000 | 6000 |

Code tương ứng: `ga.py` (các hàm `run_ga_iter`, `tabu_search`, `simulated_annealing`, `estimate_t0`) và `matheuristics.py` (các hàm `decode_ml`, `repair_ml`, `LpEvaluator.evaluate`, `run_matheur`); số dòng trong mã giả được chú thích lại trong code bằng các mục `(1)`–`(4)`.
