# Kế hoạch hoàn thiện — Nhóm 8 (SPGA Memetic)

> Mục tiêu: đóng gói và trình bày kết quả của Phần 4 (so sánh 3 thuật toán) cho buổi báo cáo.

---

## 1. Tình hình hiện tại (tóm tắt cho người không chuyên)

Chương trình đã chạy xong và có kết quả đầy đủ. Tóm lại:

- Ba "công thức" tìm phương án sản xuất đã được lập trình và so sánh công bằng (mỗi cái chạy 10 lần, cùng ngân sách).
- Hai bản cải tiến (GA-TS và GA-SA) cho kết quả **nhỉnh hơn** bản gốc một chút, GA-TS dẫn đầu.
- Điểm đáng lưu ý: mức cải thiện còn nhỏ, nên về mặt thống kê chưa thể khẳng định chắc chắn là "tốt hơn hẳn" — đây là một nhận định trung thực cần nêu trong báo cáo.

### Con số chính

| Công thức | Điểm trung bình | So với bản gốc |
|---|---|---|
| Bản gốc (SPGA) | 47,54 triệu | — |
| GA-TS | 47,75 triệu | +0,45% |
| GA-SA | 47,71 triệu | +0,37% |

GA-TS đứng nhất ở **6/10 lần chạy**.

---

## 2. Việc đã xong ✅

- [x] Viết xong mã nguồn (mô hình + 3 thuật toán).
- [x] Chạy kiểm thử (`tests.py`) — tất cả đều đạt.
- [x] Chạy thực nghiệm chính + phân tích bổ sung, xuất ra thư mục `output/` (bảng số liệu, biểu đồ hội tụ, biểu đồ hộp).
- [x] Chuẩn bị demo chạy trực tiếp trên trình duyệt (`CHAY_DEMO.bat`).
- [x] Viết bản thảo báo cáo (Chương 3–4, mô hình toán) trong thư mục `tai_lieu/`.

---

## 3. Việc còn lại 📋

| STT | Công việc | Người phụ trách | Hạn chót |
|---|---|---|---|
| 1 | Hoàn thiện bản báo cáo (chỉnh câu chữ, gắn hình ảnh và bảng kết quả thực tế vào Chương 3–4) | ... | ... |
| 2 | Rà soát lại phần "điểm mới" (danh sách tabu dùng chung) đã được mô tả rõ trong báo cáo chưa | ... | ... |
| 3 | Tập chạy demo 1–2 lần cho trơn (mở `CHAY_DEMO.bat`, để cửa sổ terminal cạnh trình duyệt) | ... | ... |
| 4 | Chuẩn bị slide / phần trình bày ngắn gọn từ bảng tóm tắt mục 1 | ... | ... |
| 5 | Duyệt lại toàn bộ một lượt trước ngày báo cáo | ... | ... |

---

## 4. Gợi ý thứ tự thực hiện

1. **Trước hết:** hoàn thiện báo cáo (việc 1–2) vì đây là phần mất nhiều thời gian nhất.
2. **Sau đó:** tập demo (việc 3) để buổi báo cáo chạy mượt, không phụ thuộc mạng.
3. **Cuối cùng:** làm slide + tổng duyệt (việc 4–5).

---

## 5. Cách chạy nhanh (để kiểm tra lại kết quả)

```bash
python tests.py          # kiểm thử (~20 giây), phải in "Tất cả kiểm thử đều đạt."
python main.py --quick   # chạy thí nghiệm chính
python main.py           # chạy đầy đủ (thí nghiệm chính + phân tích bổ sung)
```

Demo cho buổi báo cáo: bấm đúp **`CHAY_DEMO.bat`**.
