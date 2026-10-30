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

## Notebook `train_baselines.ipynb`

Notebook [`train_baselines.ipynb`](./train_baselines.ipynb) hiện thực hóa bước huấn luyện baseline trên bộ dữ liệu Elliptic đã được chuẩn hóa.

### Dữ liệu đầu vào và cấu hình

- Đọc dữ liệu từ `data/elliptic_features_scaled.csv`.
- Chỉ sử dụng các mẫu đã có nhãn, tức là các dòng có `y != -1`.
- Loại các cột định danh và nhãn khỏi tập đặc trưng: `node_idx`, `txId`, `time_step_raw` và `y`.
- Dùng temporal split để tránh trộn dữ liệu giữa các giai đoạn:
  - **Train:** `time_step_raw <= 34` — 29.894 mẫu.
  - **Test:** `time_step_raw >= 35` — 16.670 mẫu.
- Kết quả chạy hiện tại ghi nhận tổng cộng 46.564 giao dịch có nhãn.
- Seed của các mô hình được đặt là `42`; Random Forest sử dụng `n_jobs=-1`.

### Các baseline được huấn luyện

1. **Logistic Regression**
   - Khởi tạo với `random_state=42`.
   - Đánh giá bằng Precision, Recall, F1-score cho lớp `Illicit (1)` và `classification_report`.
2. **Random Forest**
   - `n_estimators=50`.
   - `max_features=50`.
   - `random_state=42`.
   - `n_jobs=-1`.
   - Đánh giá bằng cùng bộ chỉ số như Logistic Regression.

### Kết quả mẫu đã ghi trong notebook

| Mô hình | Precision (Illicit) | Recall (Illicit) | F1 (Illicit) | Accuracy |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.318 | 0.712 | 0.440 | 0.88 |
| Random Forest | 0.911 | 0.725 | 0.807 | 0.98 |

Các con số trên là output của lần chạy được lưu trong notebook; cần chạy lại notebook khi thay đổi dữ liệu, môi trường hoặc cấu hình mô hình.

### Artifact đầu ra

Notebook lưu các model bằng `joblib.dump` vào thư mục `models/`:

- `logistic_regression_model.joblib`
- `random_forest_model.joblib`

Khi chạy trên Google Colab, output mặc định là:

```text
/content/drive/MyDrive/Explainable-AML-GAT-ResNet/models/
```

Cell đầu tiên có nhánh cấu hình cho môi trường local với `../data/` và `../models/`. Tuy nhiên, các cell đọc dữ liệu và lưu model phía sau notebook hiện đặt lại `DATA_PATH` và `MODEL_DIR` theo đường dẫn Google Drive. Khi chạy ngoài Colab, cần kiểm tra và chỉnh hai biến này sang đường dẫn local trước khi thực thi các cell tương ứng.

### Cách chạy

1. Mở `train_baselines.ipynb` trong Jupyter hoặc Google Colab.
2. Bảo đảm file `data/elliptic_features_scaled.csv` tồn tại; trên Colab, mount Google Drive và kiểm tra đúng tên thư mục dự án.
3. Chạy các cell theo thứ tự: mount/cấu hình đường dẫn, import thư viện, đọc dữ liệu, temporal split, huấn luyện hai baseline và lưu artifact.
4. Kiểm tra output metric và sự xuất hiện của hai file `.joblib` trong thư mục `models/`.

Các dependency chính của notebook gồm `pandas`, `numpy`, `scikit-learn` và `joblib`.

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
