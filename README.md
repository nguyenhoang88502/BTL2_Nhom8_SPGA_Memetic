# Phần 4 – Lập trình SPGA gốc, GA-TS, GA-SA + thực nghiệm

Bài toán: hoạch định mua/thuê máy kiểm tra chip (tester, handler) qua 8 quý để tối đa hóa điểm Z = (1−k)·lợi nhuận trung bình − k·MAD, trên 30 kịch bản nhu cầu. Ba thuật toán được so sánh: **SPGA gốc** (GA thuần), **GA-TS** (GA + Tabu Search), **GA-SA** (GA + Simulated Annealing). Một bài toán **MILP** (PuLP) chỉ dùng làm đáp án chuẩn (cận tối ưu toàn cục) để đối chiếu, không phải thuật toán thứ tư.

## Yêu cầu
- Python ≥ 3.10
- `pip install -r requirements.txt` (numpy, scipy, pandas, matplotlib, openpyxl, pulp)

## Cấu trúc
Thư mục gốc chỉ có **ba file Python**:

| File | Nội dung |
|---|---|
| `config.py` | **Mọi số liệu đầu vào**, không có logic: (A) số liệu bài toán (máy, giá, nhu cầu...), (B) tham số thuật toán (quần thể, Tabu, SA...), (C) thiết lập thí nghiệm (số kịch bản, seed, giới hạn thời gian MILP). Mỗi biến có một dòng chú thích bằng tiếng Việt nói rõ ý nghĩa và đơn vị. |
| `model.py` | **Mô hình bài toán**: đổi 135 gen thành phương án (`decode`, `encode`), sửa cho khả thi (`repair`), sinh kịch bản nhu cầu (`sample_scenarios`), tính lợi nhuận và điểm Z (`scenario_profits`, `evaluate`). Không có thuật toán tìm kiếm. |
| `solver.py` | **Mọi thuật toán**: toán tử GA, bước lân cận, Tabu Search, Simulated Annealing, vòng lặp GA chung, và ở cuối file các hàm `solve_spga`, `solve_ga_ts`, `solve_ga_sa`, `solve_milp`. Chạy `python solver.py` để so sánh nhanh 3 thuật toán. |

Thư mục `cong_cu/` (công cụ chạy thí nghiệm và demo, dùng ba file trên):

| File | Nội dung |
|---|---|
| `cong_cu/main.py` | Chạy 3 thuật toán × 10 seed, xuất CSV, hình, thống kê; chạy ablation |
| `cong_cu/tests.py` | Kiểm thử nhanh (ràng buộc, encode/decode, ngân sách evaluate, Lamarckian, bước ngược, tái lập, kết quả SPGA gốc) |
| `cong_cu/demo_server.py`, `cong_cu/demo.html` | Giao diện demo cho buổi báo cáo |
| `cong_cu/exporter.py` | Thống kê kiểm định, phân tích độ nhạy, xuất báo cáo Excel cho demo |

Tài liệu: `MA_GIA.md` (mã giả), `flowcharts.html` (sơ đồ khối), `PLAN.md` (kế hoạch), `tai_lieu/` (bản thảo báo cáo).
Ở thư mục gốc còn `CHAY_DEMO.bat` và `demo.ps1` để bật demo, và `output/` chứa kết quả.

## Đọc code theo thứ tự nào (dành cho người không lập trình)
1. **`config.py`**: xem bài toán có những con số gì (máy, giá, nhu cầu) và thuật toán có những tham số gì. Biến nào ghi "GIẢ ĐỊNH" là nhóm tự đặt, biến ghi "bài báo" là lấy từ bài báo Wang, Wang & Chen (2008).
2. **`model.py`**: đọc theo thứ tự `decode` → `repair` → `sample_scenarios` → `scenario_profits` → `evaluate`. Hàm `evaluate` cho ra điểm Z của một phương án.
3. **`solver.py`**: đọc phần hướng dẫn và bảng thuật ngữ ở đầu file, rồi theo thứ tự từ trên xuống: bộ đếm đánh giá → toán tử GA → bước lân cận → Tabu Search → Simulated Annealing → khung GA chung (`run_ga_iter`) → các hàm `solve_*` ở cuối.
4. Khi đọc thuật toán, đặt **`flowcharts.html`** và **`MA_GIA.md`** cạnh code: số "BƯỚC n" trong hộp sơ đồ, dòng mã giả và chú thích code là **cùng một số**, nên đi được từ hộp sơ đồ sang dòng mã giả rồi sang dòng code.

Mỗi hàm có docstring 4 dòng: "Làm gì", "Nhận vào", "Trả ra", và "Ví dụ" (khi cần) bằng ngôn ngữ đời thường.

## Ba thuật toán
Cả ba dùng chung quần thể 30, pc 0.9, pm 0.05, tournament 2, elitism 1, lai ghép hybrid, ngân sách 6000 lần `evaluate` (tính cả lần gọi trong tìm kiếm cục bộ), 30 kịch bản và seed 1–10.

- **SPGA gốc**: GA thuần.
- **GA-TS** (GA + Tabu Search): mỗi vòng bốc 5 bước hợp lệ trong 174 bước lân cận, đánh giá cả 5, rồi đi tới bước tốt nhất không bị cấm, kể cả khi bước đó tệ hơn nghiệm hiện tại. Bước ngược bị cấm trong 7 vòng. Aspiration: bước bị cấm vẫn được đi nếu vượt Z tốt nhất từ đầu lần chạy tới giờ. Nếu cả 5 đều bị cấm thì bốc lại, tối đa 3 lần. **Điểm mới:** một danh sách tabu dùng chung cho 3 cá thể trong cùng thế hệ, reset ở mỗi lần kích hoạt.
- **GA-SA** (GA + Simulated Annealing): mỗi vòng bốc 1 bước hợp lệ, nhận nếu Z không giảm, hoặc với xác suất `exp(ΔZ/T)`. T0 ước lượng một lần ở đầu mỗi lần chạy từ 30 bước ngẫu nhiên, `T0 = −median(|Δ| bước làm tệ) / ln(0.5)`; 30 lần này tính vào ngân sách. Làm nguội `T_i = T0·α^i`, đến lần evaluate thứ 20 thì `T = 0.01·T0`.

GA-TS và GA-SA có cùng điểm kích hoạt (mỗi 10 thế hệ, 3 cá thể tốt nhất, tối đa 20 lần evaluate mỗi cá thể), cùng tập bước lân cận và cùng kiểu Lamarckian: ghi vào quần thể nghiệm **tốt nhất** gặp được, nên không bao giờ làm tệ cá thể. Hai bản chỉ khác nhau ở chiến lược tìm kiếm cục bộ.

`ls_method="hc"` là bản hill climbing (first improvement) cũ, chỉ giữ để làm ablation (so sánh phụ).

**MILP (đáp án chuẩn):** `solver.solve_milp` giải đúng bài toán này bằng PuLP + CBC trên cùng 30 kịch bản, cho cận tối ưu toàn cục; khoảng cách từ Z của ba thuật toán tới Z của MILP cho biết còn cách tối ưu bao xa. Cần cài `pulp`.

## Cách chạy
```bash
python solver.py                # so sánh nhanh 3 thuật toán (seed 1), in bảng kết quả
python cong_cu/tests.py         # khoảng 20 giây, phải in "Tất cả kiểm thử đều đạt."
python cong_cu/main.py --quick  # chỉ thí nghiệm chính (khoảng 1.5 phút)
python cong_cu/main.py          # thí nghiệm chính + ablation
CHAY_DEMO.bat                   # (bấm đúp) bật demo trên trình duyệt
```
Các lệnh chạy được từ bất kỳ thư mục nào; kết quả luôn ghi vào `output/` ở thư mục gốc.

## Đầu ra (thư mục `output/`)
| File | Nội dung |
|---|---|
| `results.csv` | 30 dòng (3 thuật toán × 10 seed): fitness, lợi nhuận TB, MAD, thời gian, số lần evaluate để hội tụ, fitness trên 1000 kịch bản kiểm tra, thống kê tìm kiếm cục bộ |
| `summary.csv` | Trung bình, độ lệch chuẩn, min, max theo thuật toán |
| `stats.txt` | Friedman, Wilcoxon hai phía theo cặp hiệu chỉnh Holm, số seed thắng |
| `history.csv` | Fitness tốt nhất theo số lần gọi evaluate (vẽ biểu đồ hội tụ) |
| `ablation.csv`, `fig_ablation.png` | TS với n_cand ∈ {3, 5, 10} và tenure ∈ {3, 7, 15}; tabu chung và riêng; SA với xác suất nhận ban đầu ∈ {0.2, 0.5, 0.8}; hill climbing cũ |
| `best_portfolio.csv`, `best_allocation.csv` | Danh mục máy và phân bổ sản lượng của nghiệm tốt nhất (giống Bảng 1 bài báo) |
| `fig_convergence.png`, `fig_boxplot.png` | Biểu đồ hội tụ (dải ±1 độ lệch chuẩn) và boxplot |
| `bao_cao_kiem_dinh.xlsx` | Báo cáo Excel xuất từ demo (tab Kiểm định lặp) |

Kết quả của bản Memetic cũ được giữ trong `output_memetic_cu/`.

## Thông số (theo tab "Thông số chung")
- Quần thể 30, lai ghép 0.9, tournament 2, elitism 1, đột biến uniform pm = 0.05
- Ngân sách 6000 lần gọi `evaluate` cho cả 3 bản (tính cả lần gọi trong tìm kiếm cục bộ và 30 lần ước lượng T0 của SA)
- 30 kịch bản nhu cầu cố định (seed 2026), dùng chung cho mọi lần chạy
- Hàm mục tiêu: max 0.5·mean(h) − 0.5·MAD(h)
- Mọi giá trị trên nằm trong `config.py`

## Demo cho buổi báo cáo (giao diện HTML)
1. Bấm đúp **`CHAY_DEMO.bat`**. Script `demo.ps1` sẽ tự tìm Python 3, cài thư viện còn thiếu (`requirements.txt`), bật `cong_cu/demo_server.py` và mở trình duyệt.
2. **Cửa sổ terminal** hiện log trực tiếp: tham số, mỗi lần Z tốt nhất tăng, từng bước tìm kiếm cục bộ, kết quả cuối. TS ghi "đi bước tệ hơn (tabu)", "aspiration", "bước X bị cấm"; SA ghi "nhận bước tệ hơn, T = …, p = …". Đặt cạnh trình duyệt để thầy xem.
3. Trên trình duyệt, bấm ▶ Chạy là ba thuật toán chạy song song. Ô **Thuật toán** chọn thuật toán cần chạy; có thể đổi seed, số lần đánh giá, k và σ. Nút **Tham số LS** mở các tham số tìm kiếm cục bộ (điểm kích hoạt, n_cand, tenure, xác suất nhận ban đầu); nút **Hướng dẫn** giải thích 3 thuật toán. Màn hình gồm: điểm Z của từng thuật toán, biểu đồ hội tụ kèm đường nét đứt là cận MILP, phương án tester theo quý, bảng so sánh, các bước tìm kiếm cục bộ mới nhất. Tab 2 (Kiểm định lặp) và tab 3 (Phân tích độ nhạy) chạy nhiều seed / quét k và σ.
4. Nhấn Enter trong cửa sổ terminal (hoặc đóng nó) để tắt demo.

Không cần mạng khi demo (chỉ cần mạng lần đầu nếu máy chưa có thư viện). Riêng `flowcharts.html` cần mạng để tải thư viện vẽ sơ đồ (flowchart.js, Raphael) từ cdnjs.
