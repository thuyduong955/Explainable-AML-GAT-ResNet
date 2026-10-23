# `notebooks/`

Thư mục này chứa các Jupyter Notebook dùng để khám phá dữ liệu, chạy thực
nghiệm và tạo các artifact giải thích cho dự án **Explainable Hybrid
GAT-ResNet Ensemble**.

## Notebook dự kiến

Nên tổ chức notebook theo thứ tự pipeline:

1. Khám phá và kiểm tra ba tệp Elliptic CSV.
2. Tiền xử lý, ánh xạ transaction ID và xây dựng temporal graph snapshots.
3. Tính năm đặc trưng topo bằng NetworkX.
4. Huấn luyện và đánh giá các baseline: Logistic Regression, Random Forest,
   XGBoost, GCN, GAT và GAT-ResNet.
5. Trích xuất embedding 100 chiều và attention weights từ GAT-ResNet.
6. Hợp nhất `166 + 5 + 100 = 271` đặc trưng, huấn luyện ensemble cây và tạo
   AML risk score.
7. Tạo giải thích bằng SHAP, attention subgraphs và phân tích top-K features.

## Quy tắc thực nghiệm

- Dùng temporal split `t = 1–34` cho train và `t = 35–49` cho test.
- Ghi rõ seed, phiên bản dữ liệu, cấu hình mô hình và đường dẫn artifact.
- Không nhúng credential, private key hoặc dữ liệu nhạy cảm vào notebook.
- Không lưu output dung lượng lớn nếu không cần thiết; artifact quan trọng nên
  được đặt ở `models/` hoặc `data/processed/`.
- Notebook phải có thể chạy lại từ đầu hoặc nêu rõ cell/phụ thuộc không thể
  tái lập.

## Đánh giá

Báo cáo tối thiểu Precision, Recall, F1-score cho lớp `Illicit`, micro-average
F1 và Matthews Correlation Coefficient (MCC). Kết quả cần được so sánh giữa
baseline và mô hình ensemble.

