# Mã giả: SPGA gốc, GA-TS và GA-SA (Nhóm 8)

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

Ba thuật toán **giống hệt nhau ở Thuật toán 1**, chỉ khác dòng 14 (`LOCAL`).

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

## So sánh nhanh

| | SPGA gốc | GA-TS | GA-SA |
|---|---|---|---|
| Tìm kiếm cục bộ | không | Tabu Search | Simulated Annealing |
| Mỗi vòng xem | – | 5 bước ứng viên, đi bước tốt nhất | 1 bước ngẫu nhiên |
| Nhận bước tệ hơn | – | Có (tất định: bước tốt nhất không bị cấm) | Có (ngẫu nhiên: xác suất giảm theo T) |
| Bộ nhớ | – | Danh sách tabu dùng chung cho 3 cá thể | Không |
| Ngân sách | 6000 | 6000 (gồm cả tìm kiếm cục bộ) | 6000 (gồm 30 lần ước lượng T0) |

Code tương ứng: `ga.py` (các hàm `run_ga_iter`, `tabu_search`, `simulated_annealing`, `estimate_t0`); số dòng trong mã giả được chú thích lại trong code bằng các mục `(1)`–`(4)`.
