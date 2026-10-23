# Explainable Hybrid GAT-ResNet Ensemble for AML Compliance

**Môn học:** IS355 - Blockchain Technology  
**Nhóm:** Snakes  
**Giảng viên hướng dẫn:** Lê Võ Đình Kha
## 📌 Giới thiệu đồ án
Dự án xây dựng hệ thống phát hiện giao dịch rửa tiền (AML - Anti-Money Laundering) trên mạng lưới Bitcoin (Elliptic Dataset). 
Hệ thống kết hợp mô hình Graph Neural Network (GAT-ResNet) và các mô hình học máy dạng cây (Tree-based), đồng thời tích hợp Explainable AI (SHAP) để giải thích kết quả. 
Trạng thái rủi ro sẽ được lưu trữ on-chain thông qua Smart Contract viết bằng Vyper trên nền tảng Ape Framework.

## 👥 Thành viên nhóm
1. **Lâm Thị Thùy Dương** (Nhóm trưởng) - Core AI
2. **Nguyễn Thái Thiện Ý** - Data Pipeline & Baseline
3. **Huỳnh Trần Ái Nữ** - Data Pipeline & Mapping
4. **Đỗ Hoàng Bảo** - ML Environment & Baseline
5. **Trương Minh Hiếu** - Blockchain Infrastructure & Testing
6. **Võ Hoàng Quý** - Smart Contract Vyper

## 📁 Cấu trúc thư mục Monorepo
- `data/`: Lưu trữ file dữ liệu đã qua tiền xử lý. *(Lưu ý: Không push các file CSV thô dung lượng lớn lên đây)*.
- `notebooks/`: Các file Jupyter Notebook (.ipynb) để phân tích dữ liệu và chạy thực nghiệm đồ thị.
- `models/`: Chứa file trọng số mô hình đã huấn luyện (.pt, .pkl).
- `contracts/`: Mã nguồn Smart Contract (.vy) của nền tảng Ape Framework.
- `tests/`: Kịch bản kiểm thử (pytest) cho Smart Contract và On-chain logic.
- `dashboard/`: Mã nguồn giao diện ứng dụng (Streamlit).

## 🛠 Hướng dẫn Cài đặt ban đầu (Dành cho thành viên)
Để bắt đầu làm việc, mỗi thành viên cần clone repository này về máy cục bộ:
```bash
git clone https://github.com/thuyduong955/Explainable-AML-GAT-ResNet
cd Explainable-AML-GAT-ResNet
```

## 🚀 QUY TRÌNH LÀM VIỆC VỚI GIT (TEAM BẮT BUỘC ĐỌC)

**⚠️ QUY TẮC TỐI THƯỢNG: TUYỆT ĐỐI KHÔNG PUSH TRỰC TIẾP LÊN NHÁNH `main` ⚠️**  
Tất cả các tính năng, đoạn code mới phải được phát triển trên một nhánh (branch) riêng biệt và gộp vào `main` thông qua **Pull Request (PR)**.

### Bước 1: Kéo code mới nhất về máy (Pull)
Trước khi bắt đầu làm một task Jira mới, hãy đảm bảo bạn đang ở nhánh `main` và cập nhật code mới nhất từ GitHub về máy:
```bash
git checkout main
git pull origin main
```

### Bước 2: Tạo nhánh làm việc riêng (Branching)
Mỗi task trên Jira sẽ tương ứng với một nhánh riêng.
**Quy tắc đặt tên nhánh:** loại-công-việc/Mã-Jira-mô-tả-ngắn.
**Ví dụ:** Ý đang làm task BF-4 tải dữ liệu thì: `git checkout -b feature/BF-4-load-data`

### Bước 3: Code và Lưu thay đổi (Commit)
Sau khi hoàn thành công việc trong thư mục của mình, tiến hành lưu lại code.
**Quy tắc bắt buộc:** Thông điệp commit phải luôn bắt đầu bằng mã Jira để hệ thống tự động cập nhật thẻ trên bảng Kanban.
**Ví dụ:**
```bash
git add .
git commit -m "BF-4: Hoàn thành load 3 file CSV bằng pandas"
```

### Bước 4: Đẩy nhánh lên GitHub (Push)
Đẩy nhánh cá nhân chứa code mới của bạn lên GitHub.
**Ví dụ:**
```bash
git push origin feature/BF-4-load-data
```
Nếu là lần push đầu tiên của nhánh này, Git có thể yêu cầu dùng lệnh:
```bash
git push --set-upstream origin feature/BF-4-load-data
```
### Bước 5: Tạo Pull Request (PR) để nghiệm thu và Gộp code
1. Truy cập vào trang GitHub của dự án.
2. GitHub sẽ tự động hiển thị thông báo màu xanh lá cây với nút "Compare & pull request" ở ngay đầu trang, hãy bấm vào đó.
3. Viết mô tả ngắn gọn về những gì bạn đã làm trong PR này.
4. Ở mục Reviewers (bên phải), hãy tag/assign Dương (hoặc người kiểm tra chéo) vào để review code.
5. Nhấn "Create pull request".
6. Sau khi code được kiểm tra và xác nhận không có lỗi (Approve), sẽ được Merge pull request để chính thức trộn code của bạn vào nhánh `main`.
