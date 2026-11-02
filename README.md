# Phần 4 – Lập trình SPGA gốc, GA-TS, GA-SA + thực nghiệm

## Yêu cầu
- Python ≥ 3.10
- `pip install numpy scipy pandas matplotlib`

## Cấu trúc
| File | Nội dung |
|---|---|
| `model.py` | (Phần 3) Bài toán: dữ liệu, sinh kịch bản nhu cầu, `repair`, `evaluate`, `decode`, `encode` |
| `ga.py` | GA: chọn lọc tournament, lai ghép hybrid (1 điểm / 2 điểm / uniform / arithmetic), đột biến uniform, `tabu_search`, `simulated_annealing`, `local_search` (hill climbing cũ), hàm `run_ga(evaluate, ls_method, seed)` với `ls_method` ∈ `none`, `tabu`, `sa`, `hc` |
| `main.py` | Chạy 3 thuật toán, mỗi bản 10 lần (seed 1–10), xuất CSV và hình, chạy ablation |
| `tests.py` | Kiểm thử nhanh (ràng buộc, encode/decode, ngân sách evaluate, Lamarckian, bước ngược, tái lập, kết quả SPGA gốc) |
| `demo_server.py`, `demo.html` | Giao diện demo cho buổi báo cáo |

## Ba thuật toán
Cả ba dùng chung quần thể 30, pc 0.9, pm 0.05, tournament 2, elitism 1, lai ghép hybrid, ngân sách 6000 lần `evaluate` (tính cả lần gọi trong tìm kiếm cục bộ), 30 kịch bản và seed 1–10.

- **SPGA gốc**: GA thuần.
- **GA-TS** (GA + Tabu Search): mỗi vòng bốc 5 bước hợp lệ trong 174 bước lân cận, đánh giá cả 5, rồi đi tới bước tốt nhất không bị cấm, kể cả khi bước đó tệ hơn nghiệm hiện tại. Bước ngược bị cấm trong 7 vòng. Aspiration: bước bị cấm vẫn được đi nếu vượt Z tốt nhất từ đầu lần chạy tới giờ. Nếu cả 5 đều bị cấm thì bốc lại, tối đa 3 lần. **Điểm mới:** một danh sách tabu dùng chung cho 3 cá thể trong cùng thế hệ, reset ở mỗi lần kích hoạt.
- **GA-SA** (GA + Simulated Annealing): mỗi vòng bốc 1 bước hợp lệ, nhận nếu `fy ≥ fx`, hoặc với xác suất `exp((fy − fx)/T)`. T0 ước lượng một lần ở đầu mỗi lần chạy từ 30 bước ngẫu nhiên, `T0 = −median(|Δ| bước làm tệ) / ln(0.5)`; 30 lần này tính vào ngân sách. Làm nguội `T_i = T0·α^i`, đến lần evaluate thứ 20 thì `T = 0.01·T0`.

GA-TS và GA-SA có cùng điểm kích hoạt (mỗi 10 thế hệ, 3 cá thể tốt nhất, tối đa 20 lần evaluate mỗi cá thể), cùng tập bước lân cận và cùng kiểu Lamarckian: ghi vào quần thể nghiệm **tốt nhất** gặp được, nên không bao giờ làm tệ cá thể. Hai bản chỉ khác nhau ở chiến lược tìm kiếm cục bộ.

`ls_method="hc"` là bản hill climbing (first improvement) cũ, chỉ giữ để làm ablation.

## Cách chạy
```bash
python tests.py          # khoảng 20 giây, phải in "Tất cả kiểm thử đều đạt."
python main.py --quick   # chỉ thí nghiệm chính
python main.py           # thí nghiệm chính + ablation
```

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

Kết quả của bản Memetic cũ được giữ trong `output_memetic_cu/`.

## Thông số (theo tab "Thông số chung")
- Quần thể 30, lai ghép 0.9, tournament 2, elitism 1, đột biến uniform pm = 0.05
- Ngân sách 6000 lần gọi `evaluate` cho cả 3 bản (tính cả lần gọi trong tìm kiếm cục bộ và 30 lần ước lượng T0 của SA)
- 30 kịch bản nhu cầu cố định (seed 2026), dùng chung cho mọi lần chạy
- Hàm mục tiêu: max 0.5·mean(h) − 0.5·MAD(h)

## Demo cho buổi báo cáo (giao diện HTML)
1. Bấm đúp **`CHAY_DEMO.bat`**. Script `demo.ps1` sẽ tự tìm Python 3, cài thư viện còn thiếu (`requirements.txt`), bật `demo_server.py` và mở trình duyệt.
2. **Cửa sổ terminal** hiện log trực tiếp: tham số, mỗi lần Z tốt nhất tăng, từng bước tìm kiếm cục bộ, kết quả cuối. TS ghi "đi bước tệ hơn (tabu)", "aspiration", "bước X bị cấm"; SA ghi "nhận bước tệ hơn, T = …, p = …". Đặt cạnh trình duyệt để thầy xem.
3. Trên trình duyệt chỉ có một màn hình, bấm ▶ Chạy là ba thuật toán chạy song song. Ô **Chạy** chọn Cả ba, SPGA gốc, GA-TS hoặc GA-SA; có thể đổi seed, số lần đánh giá, k và σ. Nút **Tìm kiếm cục bộ** mở các tham số (điểm kích hoạt, n_cand, tenure, xác suất nhận ban đầu); nút **Cách hoạt động** giải thích 3 thuật toán. Màn hình gồm: điểm Z của từng thuật toán, biểu đồ hội tụ, phương án tester theo quý, bảng so sánh (lần chạy này và kết quả đã lưu trong `output/`), các bước tìm kiếm cục bộ mới nhất.
4. Nhấn Enter trong cửa sổ terminal (hoặc đóng nó) để tắt demo.

Không cần mạng khi demo (chỉ cần mạng lần đầu nếu máy chưa có numpy).
