# Mã giả: SPGA gốc, GA-TS và GA-SA (Nhóm 8)

Tài liệu này đi cùng `solver.py` và `flowcharts.html`. **Số "BƯỚC n" ở đây trùng với số trong chú thích của code và trong các hộp của flowchart**, nên đọc được theo đường: hộp flowchart, dòng mã giả, dòng code.

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

Ba thuật toán **giống hệt nhau ở Thuật toán 1**, chỉ khác ở BƯỚC 2 (chỉ GA-SA) và BƯỚC 7 (`LOCAL`: không / TABU / SA).
Ngoài ra có một bài toán MILP dùng làm đáp án chuẩn (xem cuối tài liệu), không thuộc 3 thuật toán so sánh.

Tên hàm trong `solver.py`: Thuật toán 1 = `run_ga_iter`, Thuật toán 2 = `tabu_search`, Thuật toán 3 = `simulated_annealing`, Thuật toán 4 = `estimate_t0`.

---

## Thuật toán 1 – Khung GA chung (`run_ga_iter`)

```
Input : hàm Z, ngân sách B, LOCAL ∈ {không, TABU, SA}
Output: nghiệm tốt nhất x*

BƯỚC 1  Khởi tạo quần thể: P ← N vector ngẫu nhiên U(0,1)^135;  với mỗi x ∈ P: x ← REPAIR(x); tính Z(x)
BƯỚC 2  (chỉ GA-SA) Ước lượng T0:  T0 ← ƯỚC_LƯỢNG_T0(P)              ▹ Thuật toán 4, tốn 30 lần tính Z
        g ← 0
        lặp BƯỚC 3 đến BƯỚC 9 cho đến khi hết ngân sách:
BƯỚC 3    Elitism:  g ← g + 1;  P' ← {cá thể tốt nhất của P}
BƯỚC 4    Chọn cha mẹ:  a ← TOURNAMENT(P, 2);  b ← TOURNAMENT(P, 2)
BƯỚC 5    Lai ghép:  (c1, c2) ← LAI_GHÉP_HYBRID(a, b) với xác suất p_c, ngược lại (c1, c2) ← (a, b)
BƯỚC 6    Đột biến và sửa:  với mỗi c ∈ {c1, c2}: c ← REPAIR(ĐỘT_BIẾN(c, p_m)); tính Z(c); thêm c vào P'
          (lặp BƯỚC 4 đến BƯỚC 6 cho đến khi |P'| = N hoặc hết ngân sách);  P ← P'
BƯỚC 7    Tìm kiếm cục bộ (chỉ khi LOCAL ≠ không và g mod F = 0):
              tabu ← danh sách rỗng                                  ▹ dùng CHUNG cho L cá thể (chỉ GA-TS)
              với L cá thể tốt nhất x_i của P:
                  (y, Z(y)) ← TABU(x_i, tabu)  hoặc  SA(x_i, T0)     ▹ Thuật toán 2 / 3
BƯỚC 8    Ghi ngược (Lamarckian):  x_i ← y                            ▹ ghi nghiệm tốt nhất gặp được đè lên cá thể cũ
BƯỚC 9    Kiểm tra dừng:  còn ngân sách thì quay lại BƯỚC 3;  hết thì return x* = nghiệm có Z lớn nhất từng được tính
```

- `LAI_GHÉP_HYBRID`: chọn ngẫu nhiên 1 trong 4 kiểu (1 điểm, 2 điểm, uniform, arithmetic).
- `ĐỘT_BIẾN`: mỗi gen có xác suất p_m bị thay bằng giá trị U(0,1) mới.
- `REPAIR`: sửa nghiệm để thỏa ràng buộc công suất (mục 3.3 của báo cáo).
- SPGA gốc = Thuật toán 1 với `LOCAL = không` (bỏ BƯỚC 2, 7, 8).

---

## Thuật toán 2 – Tabu Search (GA-TS) (`tabu_search`)

```
Input : nghiệm x, danh sách tabu dùng chung, n_cand = 5, tenure = 7, E = 20
Output: nghiệm tốt nhất gặp được x_best

BƯỚC 1  Khởi tạo:  x_best ← x;  dùng ← 0
        lặp BƯỚC 2 đến BƯỚC 9 trong khi dùng < E và còn ngân sách:
BƯỚC 2    Vòng mới:  đồng_hồ ← đồng_hồ + 1
BƯỚC 3    Bốc ứng viên:  C ← bốc ngẫu nhiên n_cand bước hợp lệ trong 𝓝(x);  tính Z của từng bước;  dùng += |C|
BƯỚC 4    Lọc bước được phép:
              A ← { bước ∈ C | không bị cấm }
                ∪ { bước ∈ C | bị cấm nhưng Z > kỷ lục toàn cục }      ▹ aspiration
BƯỚC 5    Nếu A = ∅:  bốc lại (BƯỚC 3, tối đa 3 lần nữa);  vẫn A = ∅ thì dừng, return x_best
BƯỚC 6    Đi bước:  m ← bước có Z lớn nhất trong A;  x ← áp m lên x      ▹ đi tới m KỂ CẢ khi Z giảm
BƯỚC 7    Cập nhật tabu:  cấm BƯỚC_NGƯỢC(m) đến đồng_hồ + tenure
BƯỚC 8    Cập nhật kỷ lục:  nếu Z(x) > Z(x_best): x_best ← x
BƯỚC 9    Kiểm tra:  còn ngân sách (dùng < E) thì quay lại BƯỚC 2;  hết thì return x_best
```

- `BƯỚC_NGƯỢC`: K/X đổi dấu (+1 ↔ −1); chuyển sản lượng src→dst thành dst→src.
- **Điểm mới:** danh sách tabu được tạo mới ở BƯỚC 7 của Thuật toán 1 và **dùng chung** cho 3 cá thể. Cá thể sau không được đảo ngược bước mà cá thể trước vừa đi, nên 3 lần tìm kiếm tỏa ra 3 hướng khác nhau.

---

## Thuật toán 3 – Simulated Annealing (GA-SA) (`simulated_annealing`)

```
Input : nghiệm x, nhiệt độ ban đầu T0, E = 20
Output: nghiệm tốt nhất gặp được x_best

BƯỚC 1  Khởi tạo:  α ← 0,01^(1/E);  x_best ← x
        lặp BƯỚC 2 đến BƯỚC 6 cho i = 1 .. E (khi còn ngân sách):
BƯỚC 2    Thử bước:  y ← áp 1 bước ngẫu nhiên hợp lệ trong 𝓝(x) lên x;  tính Z(y)
BƯỚC 3    Nhiệt độ:  T ← T0 · α^i                                      ▹ làm nguội cấp số nhân, T_E = 0,01·T0
BƯỚC 4    Quyết định nhận:  nếu Z(y) ≥ Z(x) thì nhận
                            ngược lại nhận với xác suất exp((Z(y) − Z(x)) / T)   ▹ luật Metropolis
BƯỚC 5    Nếu nhận:  x ← y;  nếu Z(x) > Z(x_best): x_best ← x
BƯỚC 6    Kiểm tra:  i < E thì quay lại BƯỚC 2;  hết thì return x_best
```

## Thuật toán 4 – Ước lượng T0 (chạy 1 lần, chỉ cho GA-SA) (`estimate_t0`)

```
BƯỚC 1  Δ ← ∅
BƯỚC 2  Lặp 30 lần:  chọn ngẫu nhiên x ∈ P;  y ← 1 bước ngẫu nhiên từ x;  tính Z(y)
BƯỚC 3  Nếu Z(y) < Z(x):  thêm Z(x) − Z(y) vào Δ
BƯỚC 4  return T0 = −median(Δ) / ln(0,5)                              ▹ bước tệ điển hình được nhận với xác suất 50%
```

---

## So sánh nhanh

| | SPGA gốc | GA-TS | GA-SA |
|---|---|---|---|
| Tìm kiếm cục bộ (BƯỚC 7) | không | Tabu Search | Simulated Annealing |
| Mỗi vòng xem | – | 5 bước ứng viên, đi bước tốt nhất | 1 bước ngẫu nhiên |
| Nhận bước tệ hơn | – | Có (tất định: bước tốt nhất không bị cấm) | Có (ngẫu nhiên: xác suất giảm theo T) |
| Bộ nhớ | – | Danh sách tabu dùng chung cho 3 cá thể | Không |
| Ngân sách | 6000 | 6000 (gồm cả tìm kiếm cục bộ) | 6000 (gồm 30 lần ước lượng T0) |

---

## MILP: đáp án chuẩn để đối chiếu (không phải thuật toán so sánh)

`solver.solve_milp` giải **chính bài toán này** dưới dạng quy hoạch tuyến tính nguyên hỗn hợp bằng PuLP + CBC, trên cùng 30 kịch bản (seed 2026) và cùng mục tiêu Z. Vì cho nghiệm tối ưu toàn cục (hoặc nghiệm tốt nhất tìm được khi hết giờ), nó được dùng làm **cận trên**: Z của MILP là mức cao nhất có thể đạt, nên khoảng cách từ Z của SPGA, GA-TS, GA-SA tới Z_MILP cho biết 3 thuật toán còn cách tối ưu bao xa. Trong giao diện demo, đó là đường nét đứt vàng trên biểu đồ hội tụ.

Ý chính của cách giải: số máy mua và số máy thuê là biến nguyên, sản lượng là biến liên tục; độ lệch MAD được tuyến tính hóa bằng biến phụ `u_n ≥ |h_n − mean|`; ràng buộc công suất, tồn kho và cân bằng vốn viết theo từng kịch bản. MILP chạy chậm hơn nhiều (vài giây đến vài phút) so với một lần chạy GA, nên chỉ chạy một lần để lấy mốc so sánh.
