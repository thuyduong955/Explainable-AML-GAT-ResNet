# 🧠 GAT-ResNet: Core Model Architecture for AML Detection

Thư mục này chứa mã nguồn định nghĩa bộ khung kiến trúc lõi (skeleton) của mạng **GAT-ResNet**. Mô hình này được thiết kế chuyên biệt để phân loại các giao dịch hợp pháp (Licit) và bất hợp pháp (Illicit) trên đồ thị giao dịch Bitcoin, phục vụ cho hệ thống Anti-Money Laundering (AML).

Kiến trúc mô hình được triển khai bám sát theo nguyên bản đề xuất từ nghiên cứu của *Adloori et al. (2024)*.

## 🏗️ Đặc tả kiến trúc (Architecture Details)

Mô hình là sự kết hợp giữa **Graph Attention Network (GAT)** và **Residual Network (ResNet)**, được cấu hình mặc định để khớp nối hoàn hảo với tập dữ liệu Elliptic:

*   **Input Layer:** Nhận đầu vào là ma trận đồ thị với `166` đặc trưng (features) cho mỗi node (giao dịch).
*   **GAT Layer 1 & 2 (Trích xuất đặc trưng):** 
    *   Sử dụng lớp `GATConv` với `100` kênh ẩn (hidden channels) và `4` attention heads.
    *   Đầu ra của các heads được nối lại (concatenate), tạo thành tensor `400` chiều.
    *   Hàm kích hoạt `ELU` và `Dropout` với xác suất `0.5` được áp dụng ngay sau mỗi lớp để ngăn chặn overfitting.
*   **Residual Connection (Khối thặng dư):** Đầu ra của Layer 1 được cộng trực tiếp vào đầu ra của Layer 2 (`x_res = x1 + x2`) nhằm giải quyết vấn đề triệt tiêu đạo hàm (vanishing gradient) và bảo toàn thông tin đồ thị gốc.
*   **Output Layer (Phân loại):** Lớp `GATConv` thứ 3 ánh xạ tensor `400` chiều về `2` lớp đầu ra (Licit/Illicit), đi qua hàm `Softmax` để tính xác suất phân loại dự đoán.

## ⚙️ Yêu cầu môi trường (Requirements)

Để chạy được mô hình này, môi trường ảo (`.venv`) cần cài đặt PyTorch và PyTorch Geometric:
```bash
pip install torch
pip install torch_geometric