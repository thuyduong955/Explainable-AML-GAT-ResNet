# `notebooks/`

Thư mục này chứa các Jupyter Notebook dùng để thiết lập môi trường, khám phá dữ liệu, chạy thực nghiệm và tạo các artifact giải thích cho dự án **Explainable Hybrid GAT-ResNet Ensemble**.

## Danh mục Notebook

| STT | Notebook | Task Jira | Mô tả |
| :--- | :--- | :---: | :--- |
| **00** | [`00_setup_pyg_environment.ipynb`](./00_setup_pyg_environment.ipynb) | `BF-2` | Thiết lập môi trường PyTorch Geometric trên Colab GPU, mount Google Drive và kiểm tra dummy PyG Data object. |
| **01** | `01_eda_elliptic.ipynb` *(Dự kiến)* | `BF-4` | Khám phá, trực quan hóa và kiểm tra 3 tệp dữ liệu Elliptic CSV. |
| **02** | `02_preprocessing_snapshots.ipynb` *(Dự kiến)* | `BF-5` | Ánh xạ node ID, chuẩn hóa đặc trưng và dựng temporal graph snapshots. |
| **03** | `03_topological_features.ipynb` *(Dự kiến)* | `BF-6` | Trích xuất 5 đặc trưng tô-pô đồ thị cục bộ bằng NetworkX. |
| **04** | `04_baseline_models.ipynb` *(Dự kiến)* | `BF-7` | Huấn luyện baseline: Logistic Regression, Random Forest, XGBoost, GCN, GAT. |
| **05** | `05_gat_resnet_embeddings.ipynb` *(Dự kiến)* | `BF-10` | Huấn luyện GAT-ResNet, trích xuất embedding 100 chiều và attention weights. |
| **06** | `06_ensemble_risk_scoring.ipynb` *(Dự kiến)* | `BF-11` | Hợp nhất 271 đặc trưng (166 + 5 + 100) và huấn luyện ensemble cây phân loại AML. |
| **07** | `07_explainability_shap.ipynb` *(Dự kiến)* | `BF-12` | Phân tích giải thích mô hình bằng SHAP values và attention subgraphs. |

---

## Hướng dẫn chạy `00_setup_pyg_environment.ipynb` (Colab Setup)

1. **Thiết lập Runtime:**
   - Mở notebook trên Google Colab.
   - Vào `Runtime` -> `Change runtime type` -> Chọn **T4 GPU** (hoặc GPU tương đương).
2. **Mount Google Drive:**
   - Đảm bảo bạn đã có quyền truy cập vào thư mục shared Drive của dự án và đã thêm shortcut vào **My Drive** với tên `Explainable-AML-GAT-ResNet` (xem chi tiết tại [`data/README.md`](../data/README.md)).
3. **Thực thi:**
   - Chạy tuần tự các cell:
     - **Cell 1:** Xác minh CUDA và cài đặt `torch-geometric`, `torch-scatter`, `torch-sparse` khớp phiên bản.
     - **Cell 2:** Mount Google Drive và kiểm tra kết nối tới `data/`.
     - **Cell 3:** Tạo đồ thị mẫu `Data(x=[4, 166], edge_index=[2, 4])` trên GPU để kiểm tra tính toàn vẹn.

---

## Quy tắc thực nghiệm

- Dùng temporal split `t = 1–34` cho train và `t = 35–49` cho test.
- Ghi rõ seed, phiên bản dữ liệu, cấu hình mô hình và đường dẫn artifact.
- Không nhúng credential, private key hoặc dữ liệu nhạy cảm vào notebook.
- Không lưu output dung lượng lớn nếu không cần thiết; artifact quan trọng nên được đặt ở `models/` hoặc `data/processed/`.
- Notebook phải có thể chạy lại từ đầu hoặc nêu rõ cell/phụ thuộc không thể tái lập.

## Tiêu chí đánh giá

Báo cáo tối thiểu Precision, Recall, F1-score cho lớp `Illicit` (Class 1), micro-average F1 và Matthews Correlation Coefficient (MCC). Kết quả cần được so sánh giữa các baseline và mô hình ensemble.
