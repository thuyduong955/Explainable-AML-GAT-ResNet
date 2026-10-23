# `models/`

Thư mục này lưu các artifact của mô hình học máy và học sâu được huấn luyện
cho bài toán phát hiện giao dịch bất hợp pháp.

## Artifact dự kiến

- Checkpoint và trọng số của GAT-ResNet 3 lớp.
- Embedding node 100 chiều được trích xuất từ GAT-ResNet.
- Mô hình baseline và ensemble như Random Forest, XGBoost.
- Preprocessor, scaler, feature schema và mapping cần để suy luận.
- Metadata về dataset version, temporal split, seed, hyperparameters và metrics.

Mô hình ensemble sử dụng 271 đặc trưng gồm 166 đặc trưng Elliptic gốc, 5 đặc
trưng topo NetworkX và 100 chiều embedding GAT-ResNet. Mô hình cũng được dùng
để tạo risk score AML trong khoảng `0–100` và nhãn `Licit/Illicit`.

## Quy ước đặt tên

Tên file nên thể hiện mô hình và phiên bản, ví dụ:

```text
gat-resnet-baseline-v1.pt
xgboost-hybrid-v1.json
feature-schema-v1.json
```

Mỗi artifact cần có metadata hoặc tài liệu đi kèm để xác định cách tái tạo.
Không commit checkpoint lớn hoặc thông tin bí mật nếu chưa thống nhất cơ chế
lưu trữ của nhóm.

