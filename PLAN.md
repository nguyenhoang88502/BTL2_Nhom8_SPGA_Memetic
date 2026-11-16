# Kế hoạch hoàn thiện — Nhóm 8 (SPGA Memetic)

> Mục tiêu: đóng gói và trình bày kết quả Phần 4 — so sánh 3 thuật toán (SPGA gốc, GA-TS, GA-SA) và đối chiếu với cận tối ưu toàn cục MILP — cho buổi báo cáo.

---

## 1. Tình hình hiện tại (tóm tắt cho người không chuyên)

Hệ thống đã hoàn chỉnh cả 3 "công thức" tìm phương án sản xuất, được so sánh công bằng trên cùng 30 kịch bản nhu cầu và cùng ngân sách đánh giá:

| Công thức | Ý tưởng | Màu trên biểu đồ |
|---|---|---|
| SPGA gốc | GA thuần (quần thể 30, lai ghép hybrid, đột biến uniform) | Xanh |
| GA-TS | GA + Tabu Search (danh sách tabu dùng chung — điểm mới) | Cam |
| GA-SA | GA + Simulated Annealing (ước lượng T0, làm nguội cấp số nhân) | Xanh lá |

Công cụ hỗ trợ trong demo:

- **Cận tối ưu toàn cục MILP** (PuLP + CBC) tự chạy nền mỗi khi bấm Chạy — vẽ đường nét đứt vàng trên biểu đồ hội tụ để thấy khoảng cách (optimality gap) của các GA so với phương án tối ưu lý thuyết.
- **Dashboard 3 tab**: (1) Chạy đơn — xem từng thế hệ và từng bước Tabu/SA; (2) Kiểm định lặp — chạy N seed, Friedman + Wilcoxon + t-test, kết luận H0/H1 theo mức ý nghĩa α, bảng xếp hạng Top 3, lịch sử từng run; (3) Phân tích độ nhạy — quét k ∈ [0,1] và σ ∈ [1000, 10000], tooltip chi tiết từng thuật toán. Xuất báo cáo Excel 1 sheet "Bao_Cao_Kiem_Dinh".
- **Giao diện** chuẩn phần mềm cũ (đen-trắng-xám, Arial, phẳng tuyệt đối, viền 1px) với color coding nhất quán để phân biệt 3 thuật toán; chạy hoàn toàn offline qua `CHAY_DEMO.bat`.

### Con số thực nghiệm (3 thuật toán, đã chạy trước)

| Công thức | Điểm trung bình | So với bản gốc |
|---|---|---|
| Bản gốc (SPGA) | 47,54 triệu | — |
| GA-TS | 47,75 triệu | +0,45% |
| GA-SA | 47,71 triệu | +0,37% |

GA-TS đứng nhất ở **6/10 lần chạy**. Khoảng cách tới cận MILP sẽ lấy từ một lần kiểm định lặp đầy đủ trên dashboard trước khi đưa vào báo cáo.

---

## 2. Mã giả

### 2.1 Khung GA chung (SPGA gốc, GA-TS, GA-SA dùng chung)

```
Khởi tạo quần thể ngẫu nhiên (30 cá thể), sửa cho khả thi (repair), chấm điểm Z
nếu là GA-SA: ước lượng T0 (30 lần đánh giá, CÓ tính vào ngân sách)
lặp cho đến khi hết ngân sách đánh giá (6000):
    giữ 1 cá thể tốt nhất sang thế hệ mới (elitism)
    lặp cho đến khi đủ 30 cá thể:
        chọn 2 cha mẹ bằng tournament k = 2
        lai ghép (pc = 0.9, ngẫu nhiên 1 trong 4 kiểu: 1 điểm, 2 điểm, uniform, arithmetic)
        đột biến uniform (pm = 0.05) → repair → chấm điểm
    nếu có tìm kiếm cục bộ và thế hệ chia hết cho 10:
        với 3 cá thể tốt nhất (mỗi cá thể tối đa 20 lần đánh giá):
            chạy Tabu Search / Simulated Annealing
            ghi nghiệm tốt nhất tìm được đè lên cá thể đó (Lamarckian)
trả về nghiệm tốt nhất từng gặp
```

### 2.2 GA-TS (Tabu Search có lấy mẫu lân cận)

```
best = x
lặp khi còn ngân sách (tối đa 20 lần đánh giá):
    bốc ngẫu nhiên 5 bước lân cận hợp lệ (trong 174 bước), chấm điểm cả 5
    được phép = các bước KHÔNG bị cấm
              + các bước bị cấm nhưng phá kỷ lục toàn cục (aspiration)
    nếu bốc 4 lần đều không có bước được phép: dừng
    đi tới bước tốt nhất trong số được phép (KỂ CẢ khi tệ hơn nghiệm hiện tại)
    cấm bước ngược của bước vừa đi trong 7 vòng tiếp theo
    nếu Z mới > Z(best): best = x
trả về best
ĐIỂM MỚI: MỘT danh sách tabu dùng chung cho cả 3 cá thể của cùng lần kích hoạt
(reset ở đầu mỗi lần kích hoạt) → 3 cá thể mài theo 3 hướng khác nhau,
không đảo ngược bước của nhau.
```

### 2.3 GA-SA (Simulated Annealing, luật Metropolis)

```
Ước lượng T0 (1 lần đầu chạy GA):
    lặp 30 lần: áp 1 bước ngẫu nhiên, nếu Z giảm thì ghi |ΔZ|
    T0 = -median(|ΔZ|) / ln(0.5)      // bước tệ điển hình được nhận 50% lúc đầu
α = 0.01^(1/20)                        // sau 20 lần đánh giá T còn 1% của T0
best = x
lặp tối đa 20 lần:
    y = áp 1 bước lân cận ngẫu nhiên, tính Z(y), T = T0·α^i
    nếu Z(y) ≥ Z(x): nhận
    ngược lại: nhận với xác suất exp((Z(y) − Z(x)) / T)
    nếu nhận và Z(y) > Z(best): best = y
trả về best
```

### 2.4 Cận tối ưu toàn cục MILP (PuLP + CBC) — đáp án chuẩn, không phải thuật toán so sánh

```
Cùng mô hình với 3 thuật toán: K (mua), X (thuê) là biến NGUYÊN, Q liên tục,
MAD được tuyến tính hóa, và toàn bộ được giải trong MỘT bài toán → nghiệm tối ưu toàn cục
(hoặc nghiệm tốt nhất tìm được khi hết 120 giây, `solver.solve_milp`).
Vai trò: đường baseline màu vàng trên biểu đồ hội tụ —
khoảng cách của GA tới Z_opt chính là optimality gap.
```

---

## 3. Việc đã xong ✅

- [x] Mã nguồn gọn lại thành 3 file: `config.py` (số liệu), `model.py` (bài toán), `solver.py` (3 thuật toán + MILP); công cụ chạy thí nghiệm và demo nằm trong `cong_cu/`.
- [x] Kiểm thử `cong_cu/tests.py` đạt.
- [x] Dashboard 3 tab: kiểm định lặp (Friedman/Wilcoxon/t-test, H0/H1/α, leaderboard, lịch sử run), phân tích độ nhạy k × σ kèm tooltip, xuất Excel 1 sheet.
- [x] MILP tự chạy nền khi bấm Chạy/Kiểm định + đường baseline vàng trên biểu đồ hội tụ.
- [x] Giao diện chuẩn phần mềm cũ (B&W, Arial, phẳng) + color coding nhất quán cho 3 thuật toán.
- [x] Bản thảo báo cáo (Chương 3–4, mô hình toán) trong `tai_lieu/`.

---

## 4. Việc còn lại 📋

| STT | Công việc | Người phụ trách | Hạn chót |
|---|---|---|---|
| 1 | Chạy kiểm định lặp đầy đủ với 3 thuật toán trên dashboard để lấy khoảng cách MILP vào báo cáo | ... | ... |
| 2 | Hoàn thiện bản báo cáo: gắn ảnh dashboard mới, bảng 3 thuật toán và mục MILP vào Chương 3–4 | ... | ... |
| 3 | Rà soát lại phần "điểm mới" (danh sách tabu dùng chung) đã được mô tả rõ trong báo cáo chưa | ... | ... |
| 4 | Tập chạy demo 1–2 lần cho trơn (mở `CHAY_DEMO.bat`, để cửa sổ terminal cạnh trình duyệt) | ... | ... |
| 5 | Chuẩn bị slide / phần trình bày ngắn gọn từ bảng tóm tắt mục 1 và mã giả mục 2 | ... | ... |
| 6 | Duyệt lại toàn bộ một lượt trước ngày báo cáo | ... | ... |

---

## 5. Gợi ý thứ tự thực hiện

1. **Trước hết:** chạy kiểm định lặp (việc 1) rồi hoàn thiện báo cáo (việc 2–3) — phần mất nhiều thời gian nhất.
2. **Sau đó:** tập demo (việc 4) để buổi báo cáo chạy mượt, không phụ thuộc mạng.
3. **Cuối cùng:** làm slide + tổng duyệt (việc 5–6).

---

## 6. Cách chạy nhanh (để kiểm tra lại kết quả)

```bash
python solver.py               # so sánh nhanh 3 thuật toán (seed 1)
python cong_cu/tests.py        # kiểm thử (~20 giây), phải in "Tất cả kiểm thử đều đạt."
python cong_cu/main.py --quick # chạy thí nghiệm chính
python cong_cu/main.py         # chạy đầy đủ (thí nghiệm chính + phân tích bổ sung)
```

Demo cho buổi báo cáo: bấm đúp **`CHAY_DEMO.bat`** rồi mở http://localhost:8765.

Lưu ý thời gian chạy: một lần chạy GA với ngân sách 6000 mất vài giây; cận MILP chạy nền vài giây đến vài phút (tối đa 120 giây).
