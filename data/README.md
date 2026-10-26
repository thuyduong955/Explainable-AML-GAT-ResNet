# `data/`

Thư mục này chứa dữ liệu phục vụ pipeline phát hiện giao dịch Bitcoin bất hợp pháp
trên **Elliptic Dataset**.

## Phạm vi dữ liệu

- Ba tệp CSV nguồn của Elliptic: nút/giao dịch, cạnh và nhãn.
- Dữ liệu đã được làm sạch, ánh xạ transaction ID sang node index và chuẩn hóa.
- Các snapshot đồ thị có hướng theo 49 time step.
- Các ma trận đặc trưng topo được tính bằng NetworkX:
  `in-degree`, `out-degree`, `in/out-degree ratio`, `PageRank` và
  `clustering coefficient`.
- Các tệp trung gian hoặc kết quả tiền xử lý cần thiết cho notebook và mô hình.

Dataset có quy mô dự kiến 203.769 nodes, 234.355 edges và 166 đặc trưng
(94 local features, 72 aggregated features).

## Quy ước lưu trữ

Nên phân biệt dữ liệu theo các nhóm:

```text
data/
├── raw/          # Dữ liệu tải về, không chỉnh sửa
├── interim/      # Dữ liệu sau ánh xạ/làm sạch
└── processed/    # Tensor, snapshot và feature matrix dùng để huấn luyện
```

Các thư mục con trên chỉ được tạo khi thực sự cần. Không commit các CSV thô hoặc
artifact dung lượng lớn lên repository; dùng cơ chế lưu trữ phù hợp và ghi lại
nguồn, phiên bản, checksum hoặc hướng dẫn tải trong commit/README liên quan.

## Quy tắc chia dữ liệu

Pipeline sử dụng temporal split nghiêm ngặt:

- `t = 1–34`: train
- `t = 35–49`: test

Không trộn dữ liệu tương lai vào tập huấn luyện. Các node chưa có nhãn vẫn được
giữ lại trong message passing của đồ thị.

## Nguồn

- Elliptic Dataset: <https://www.kaggle.com/datasets/ellipticco/elliptic-data-set>
- Mô tả pipeline và các yêu cầu đầy đủ: [README tổng](../README.md)

