"""
config.py - TẤT CẢ số liệu đầu vào của dự án nằm ở đây (không có logic nào khác).

Ba file Python ở thư mục gốc liên hệ với nhau như sau:
    config.py  ->  model.py  ->  solver.py
    (số liệu)      (bài toán)     (thuật toán tìm lời giải)
Muốn đổi một con số (giá máy, nhu cầu, cỡ quần thể, ngân sách...) thì chỉ sửa ở file này.

Thứ tự đọc cả dự án: config.py (file này) -> model.py -> solver.py.

File chia làm 3 phần:
    PHẦN A. Số liệu của bài toán (máy, sản phẩm, tiền, nhu cầu).
    PHẦN B. Tham số của thuật toán (quần thể, Tabu Search, Simulated Annealing).
    PHẦN C. Thiết lập thí nghiệm (số kịch bản, hạt giống ngẫu nhiên, giới hạn thời gian MILP).

Mỗi biến có một dòng chú thích: ý nghĩa nghiệp vụ và đơn vị.
Lưu ý: giao diện demo (cong_cu/demo_server.py) có thể đổi K_RISK, SIGMA và MU ngay lúc chạy,
vì vậy các file khác luôn đọc chúng dưới dạng cfg.K_RISK, cfg.SIGMA, cfg.MU, không chép lại.
"""
import numpy as np

# =====================================================================================
# PHẦN A. SỐ LIỆU CỦA BÀI TOÁN
# Quy mô giữ như bài báo Wang, Wang & Chen (2008): 3 tester, 4 handler, 3 sản phẩm, 8 quý.
# Số liệu ghi "bài báo" lấy từ bài báo. Số liệu ghi "GIẢ ĐỊNH" là giả định của nhóm
# vì bài báo không công bố.
# =====================================================================================

# ------------------------------------------------------------------ kích thước
P = 8                                    # số quý lập kế hoạch (quý)
M = 3                                    # số loại tester (máy kiểm tra chip) - M1, M2, M3
A = 4                                    # số loại handler (máy nạp chip vào tester) - H1..H4
T = 3                                    # số loại sản phẩm - SP1, SP2, SP3
MTS = np.array([True, False, False])     # SP1 sản xuất để dự trữ kho (make-to-stock); SP2, SP3 sản xuất theo đơn (make-to-order)

# ------------------------------------------------------------ số liệu từ bài báo
F0 = 20_000_000.0        # vốn ban đầu (đồng)
I_RATE = 0.02            # lãi suất mỗi quý (2% = 0.02)
UTIL = 0.80              # hiệu suất mục tiêu của tester và handler (80%)
HOURS = 1800.0           # số giờ làm việc mỗi máy mỗi quý (giờ/quý)
SIGMA = 7000.0           # độ lệch chuẩn của nhu cầu mỗi sản phẩm mỗi quý (đơn vị sản phẩm)
K_RISK = 0.5             # mức ngại rủi ro k: 0 = chỉ quan tâm lợi nhuận, 1 = chỉ quan tâm ổn định (không có đơn vị)
K0_M = np.array([2, 1, 1])           # số tester sẵn có ở kỳ 0 của M1, M2, M3 (máy; Bảng 1a bài báo)
K0_A = np.array([2, 3, 3, 2])        # số handler sẵn có ở kỳ 0 của H1..H4 (máy; Bảng 1b bài báo)

# --------------------------------------------------------------- số liệu GIẢ ĐỊNH
MU = np.array([25_000.0, 22_000.0, 15_000.0])   # GIẢ ĐỊNH: nhu cầu trung bình mỗi quý của SP1, SP2, SP3 (đơn vị sản phẩm/quý)
B = np.array([200.0, 230.0, 260.0])             # GIẢ ĐỊNH: lợi nhuận trên mỗi đơn vị bán được của SP1, SP2, SP3 (đồng/đơn vị)
J_HOLD = np.array([10.0, 0, 0])                 # GIẢ ĐỊNH: phí lưu kho mỗi đơn vị tồn mỗi quý (đồng/đơn vị/quý; chỉ SP1)
L_BACK = np.array([40.0, 0, 0])                 # GIẢ ĐỊNH: phí thiếu hàng mỗi đơn vị thiếu mỗi quý (đồng/đơn vị/quý; chỉ SP1)

# C_MT[m, t] = 1 nếu tester m làm được sản phẩm t (theo cấu trúc Bảng 1c bài báo)
C_MT = np.array([[1, 1, 0],    # M1 làm SP1, SP2
                 [1, 0, 1],    # M2 làm SP1, SP3
                 [0, 1, 1]])   # M3 làm SP2, SP3
# R_MT[m, t] = tốc độ lý thuyết của tester m khi làm sản phẩm t (đơn vị/giờ; GIẢ ĐỊNH)
R_MT = np.array([[3.0, 2.6, 0.0],
                 [3.4, 0.0, 2.4],
                 [0.0, 3.0, 2.8]])
# H_OF[m, t] = handler đi cặp với (tester m, sản phẩm t); -1 nghĩa là cặp đó không tồn tại (cấu hình đơn giản hóa, GIẢ ĐỊNH)
H_OF = np.array([[0, 1, -1],   # M1-SP1 dùng H1, M1-SP2 dùng H2
                 [1, -1, 2],   # M2-SP1 dùng H2, M2-SP3 dùng H3
                 [-1, 3, 2]])  # M3-SP2 dùng H4, M3-SP3 dùng H3
R_H = np.array([4.0, 3.6, 3.2, 3.8])            # GIẢ ĐỊNH: tốc độ của handler H1..H4 (đơn vị/giờ)

E_M = np.array([1_500_000.0, 1_700_000.0, 1_900_000.0])        # GIẢ ĐỊNH: giá mua 1 tester M1, M2, M3 (đồng/máy)
D_M = 0.4 * E_M                                                # GIẢ ĐỊNH: giá thanh lý 1 tester = 40% giá mua (đồng/máy)
U_M = np.array([140_000.0, 160_000.0, 180_000.0])              # GIẢ ĐỊNH: phí chuyển/thuê 1 tester trong 1 quý (đồng/máy/quý)
E_A = np.array([150_000.0, 160_000.0, 170_000.0, 155_000.0])   # GIẢ ĐỊNH: giá mua 1 handler H1..H4 (đồng/máy)
D_A = 0.4 * E_A                                                # GIẢ ĐỊNH: giá thanh lý 1 handler = 40% giá mua (đồng/máy)
U_A = np.array([14_000.0, 15_000.0, 16_000.0, 14_500.0])       # GIẢ ĐỊNH: phí chuyển/thuê 1 handler trong 1 quý (đồng/máy/quý)

# ------------------------------------------------- giới hạn khi đổi gen thành số máy
K_EXTRA_M = 6            # GIẢ ĐỊNH: được mua thêm tối đa bao nhiêu tester mỗi loại so với ban đầu (máy)
K_EXTRA_A = 8            # GIẢ ĐỊNH: được mua thêm tối đa bao nhiêu handler mỗi loại so với ban đầu (máy)
X_MAX_M = 6              # GIẢ ĐỊNH: được chuyển/thuê tối đa bao nhiêu tester mỗi loại trong 1 quý (máy/quý)
X_MAX_A = 8              # GIẢ ĐỊNH: được chuyển/thuê tối đa bao nhiêu handler mỗi loại trong 1 quý (máy/quý)
Q_MAX_FACTOR = 1.5       # GIẢ ĐỊNH: 1 tester được làm tối đa 1.5 lần nhu cầu trung bình của sản phẩm đó trong 1 quý (hệ số, không có đơn vị)

# =====================================================================================
# PHẦN B. THAM SỐ CỦA THUẬT TOÁN
# =====================================================================================

# --------------------------------------- khung GA (giống nhau ở cả 3 thuật toán)
POP_SIZE = 30            # số cá thể (phương án) trong quần thể (cá thể)
PC = 0.9                 # xác suất lai ghép hai cha mẹ (0.9 = 90%)
PM = 0.05                # xác suất mỗi gen bị đột biến (0.05 = 5%)
TOURNAMENT = 2           # mỗi lần chọn cha mẹ, bốc ngẫu nhiên bấy nhiêu cá thể rồi lấy cá thể tốt nhất (cá thể)
ELITE = 1                # số cá thể tốt nhất được giữ nguyên sang thế hệ sau (cá thể)
BUDGET = 6000            # tổng số lần tính Z cho một lần chạy (lần; chế độ dừng theo số lần đánh giá)
TIME_LIMIT_SEC = 180.0   # giới hạn thời gian thực cho một lần chạy (giây; chế độ dừng theo thời gian)

# ----------------------- điểm kích hoạt tìm kiếm cục bộ (dùng chung GA-TS và GA-SA)
LS_FREQ = 10             # cứ bao nhiêu thế hệ thì chạy tìm kiếm cục bộ một lần (thế hệ)
LS_TOP = 3               # mỗi lần chỉ mài bấy nhiêu cá thể tốt nhất (cá thể)
LS_MAX_EVALS = 20        # mỗi cá thể được dùng tối đa bấy nhiêu lần tính Z (lần)
LS_SHIFT = 0.10          # bước lân cận "chuyển sản lượng": chuyển bấy nhiêu phần sản lượng giữa 2 tester (0.10 = 10%)
LS_METHODS = ("none", "tabu", "sa", "hc")   # các kiểu tìm kiếm cục bộ: không (SPGA gốc), Tabu Search, Simulated Annealing, hill climbing (chỉ để so sánh phụ)

# --------------------------------------------------------------- Tabu Search
TS_N_CAND = 5            # mỗi vòng xem thử bấy nhiêu bước lân cận bốc ngẫu nhiên (bước)
TS_TENURE = 7            # bước ngược của bước vừa đi bị cấm trong bấy nhiêu vòng (vòng)
TS_REDRAWS = 3           # nếu cả nhóm ứng viên đều bị cấm thì bốc lại tối đa bấy nhiêu lần (lần)

# ------------------------------------------------------- Simulated Annealing
SA_P0 = 0.5              # lúc bắt đầu, một bước tệ "điển hình" được nhận với xác suất này (0.5 = 50%)
SA_T0_SAMPLES = 30       # số lần tính Z dùng để ước lượng nhiệt độ ban đầu T0 (lần; có tính vào ngân sách)
SA_FINAL_RATIO = 0.01    # đến lần tính Z cuối của mỗi đợt SA thì nhiệt độ còn bấy nhiêu phần của T0 (0.01 = 1%)

# =====================================================================================
# PHẦN C. THIẾT LẬP THÍ NGHIỆM
# =====================================================================================
N_SCEN = 30              # số kịch bản nhu cầu dùng để chấm điểm Z (kịch bản)
SCEN_SEED = 2026         # hạt giống sinh bộ kịch bản; mọi lần chạy dùng chung một bộ (số nguyên)
TEST_SCEN = 1000         # số kịch bản của bộ kiểm tra độc lập, dùng để thử lại nghiệm tốt nhất (kịch bản)
TEST_SEED = 9999         # hạt giống sinh bộ kịch bản kiểm tra (số nguyên)
SEEDS = range(1, 11)     # các hạt giống của GA cho 10 lần chạy lặp: seed 1 đến 10 (số nguyên)
MILP_TIME_LIMIT_SEC = 120   # thời gian tối đa cho bài toán MILP chuẩn so sánh (giây)
