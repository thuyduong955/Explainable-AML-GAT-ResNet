## 🧹 Tiền xử lý dữ liệu Elliptic (Data Pipeline & Mapping)

**Người thực hiện:** Huỳnh Trần Ái Nữ
**Notebook:** `notebooks/preprocess_elliptic.ipynb` (chạy trên Google Colab)

### 1. Mục đích

Dữ liệu gốc của Elliptic dùng `txId` (số rất lớn, không liên tục) để định danh giao dịch, và file đặc trưng chưa được chuẩn hóa. Notebook này biến dữ liệu thô thành dạng **sẵn sàng đưa vào PyTorch Geometric** để nhóm Core AI (GAT-ResNet) và nhóm Baseline (Tree-based) dùng chung, đảm bảo mọi người làm trên cùng một bộ dữ liệu.

### 2. Dữ liệu đầu vào (3 file CSV thô)

| File | Nội dung | Kích thước |
|---|---|---|
| `elliptic_txs_features.csv` | Không có header. Cột 0 = `txId`, cột 1 = time step, cột 2..166 = 165 đặc trưng | 203,769 × 167 |
| `elliptic_txs_edgelist.csv` | Cạnh giữa các giao dịch (`txId1`, `txId2`) | 234,355 cạnh |
| `elliptic_txs_classes.csv` | Nhãn của từng giao dịch (`txId`, `class`) | 203,769 dòng |

Notebook tự tải 3 file từ thư mục Drive chung của nhóm (cần đăng nhập đúng tài khoản Google có quyền truy cập). **Không push file CSV thô lên GitHub.**

### 3. Các bước xử lý

1. **Tải và đọc dữ liệu**: tải 3 file về Colab, gán tên cột cho file features.
2. **Tạo từ điển `txId → node index`** (0 … 203,768): theo đúng thứ tự dòng trong file features, nên `x[i]` luôn là đặc trưng của node `i`.
3. **Cập nhật edgelist**: thay `txId` bằng node index. Kiểm tra không có cạnh nào bị thiếu node (kết quả: 0 cạnh lỗi).
4. **Mã hóa nhãn** và căn theo thứ tự node:
   - `1` = illicit (bất hợp pháp)
   - `0` = licit (hợp pháp)
   - `-1` = unknown (chưa gán nhãn)
5. **Kiểm tra NaN/inf**: kết quả là 0 giá trị thiếu. Code vẫn có sẵn bước điền bằng median nếu sau này dữ liệu đổi.
6. **Chuẩn hóa (StandardScaler)** cho 166 cột (time step + 165 đặc trưng). Scaler chỉ được **fit trên time step ≤ 34** (136,265 node, tập train) để tránh rò rỉ dữ liệu (data leakage), sau đó áp dụng cho toàn bộ dữ liệu. Có thể đổi bằng cờ `FIT_ON_TRAIN_ONLY`.
7. **Lưu kết quả và kiểm tra lại** file `.pt` sau khi lưu.

### 4. Kết quả thống kê

- Số node: **203,769** | Số cạnh: **234,355** | Số đặc trưng: **166**
- Nhãn: **4,545 illicit**, **42,019 licit**, **157,205 unknown**
- Mean/std của X xấp xỉ 0.07 / 1.29 (không đúng 0/1 vì scaler chỉ fit trên time step ≤ 34, các time step sau có phân phối hơi khác. Đây là điều bình thường).

### 5. Dữ liệu đầu ra (lưu trong Drive: `MyDrive/elliptic_processed/`)

| File | Mô tả |
|---|---|
| `elliptic_processed.pt` | Dict gồm `x` [N,166], `edge_index` [2,E], `y` [N], `time_step` [N], `tx_ids` [N] |
| `txid_to_idx.json` | Từ điển `txId → node index` để truy ngược về giao dịch gốc (dùng cho SHAP/explainability, dashboard) |
| `elliptic_edgelist_indexed.csv` | Edgelist đã đổi sang node index |
| `elliptic_features_scaled.csv` | Đặc trưng đã chuẩn hóa + `node_idx`, `txId`, `time_step_raw`, `y` (dùng cho các mô hình tree-based) |

> ⚠️ Các file output khá nặng, nên chia sẻ qua Drive, **không push lên GitHub**. Chỉ commit notebook và README này.

### 6. Cách sử dụng

```python
import torch
from torch_geometric.data import Data

d = torch.load('elliptic_processed.pt')
data = Data(x=d['x'], edge_index=d['edge_index'], y=d['y'])

# Lấy mask theo thời gian và nhãn đã biết
known = d['y'] != -1
train_mask = known & (d['time_step'] <= 34)
test_mask  = known & (d['time_step'] > 34)

# Truy ngược node index -> txId gốc
original_txid = d['tx_ids'][node_idx].item()
```

### 7. Lưu ý quan trọng cho cả nhóm

- **Chia train/test theo thời gian** (time step ≤ 34 train, > 34 test), không chia ngẫu nhiên.
- **Chỉ tính loss/metric trên node có nhãn** (`y != -1`). Node `unknown` vẫn nên giữ trong đồ thị để GAT truyền thông tin.
- Cột `time_step` nằm trong 166 đặc trưng và đã được chuẩn hóa. Nếu không muốn mô hình dùng time step làm đặc trưng, hãy bỏ cột đầu tiên của `x` (dùng `d['time_step']` để chia tập).
- Trong `edge_index`, cạnh là có hướng (theo dòng tiền). Nếu cần đồ thị vô hướng, dùng `torch_geometric.utils.to_undirected`.
