# `dashboard/`

Thư mục này chứa giao diện Streamlit để trình bày kết quả phát hiện AML và
tương tác với AML Compliance Gatekeeper trên testnet.

## Luồng chính

1. Nhận transaction hoặc node cần kiểm tra.
2. Nạp preprocessor và mô hình từ `models/`.
3. Tạo dự đoán `Licit/Illicit`, risk score `0–100` và các chỉ báo giải thích.
4. Hiển thị SHAP feature attributions, attention subgraph và thông tin topo
   khi có sẵn.
5. Dùng Web3.py gửi risk score tới Smart Contract trong `contracts/`.
6. Hiển thị transaction hash, trạng thái xác nhận và một trong ba mức cảnh báo.

## Yêu cầu vận hành

- Chỉ sử dụng model artifact và feature schema tương thích với nhau.
- Hiển thị rõ thời điểm, model version, network/chain ID và trạng thái giao
  dịch on-chain.
- Xử lý minh bạch lỗi thiếu model, RPC lỗi, transaction thất bại hoặc score
  không hợp lệ; không hiển thị thành công giả.
- Không đặt private key, seed phrase hoặc RPC secret trực tiếp trong source
  code. Dùng biến môi trường hoặc cơ chế secrets của môi trường triển khai.
- Dashboard trong phạm vi đồ án là demo batch/offline chạy trên testnet, không
  thay thế quy trình điều tra AML thực tế.

## Artifact hiển thị

Các kết quả nên liên kết được với notebook và model tương ứng, gồm risk score,
nhãn dự đoán, các feature quan trọng, SHAP plot/attention subgraph và thông tin
giao dịch smart contract. Xem [README tổng](../README.md) để biết quy trình
branch và Pull Request của nhóm.

