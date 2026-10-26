# `contracts/`

Thư mục này chứa Smart Contract Vyper **AML Compliance Gatekeeper**. Contract
là lớp compliance on-chain của hệ thống, tiếp nhận AML risk score từ dashboard,
ghi nhận quyết định và áp dụng các mức cảnh báo giao dịch.

## Trách nhiệm chính

- Nhận và lưu risk score trong khoảng `0–100`.
- Phân loại quyết định theo ba mức cảnh báo:
  cho phép, cảnh báo hoặc chặn.
- Cung cấp các hàm đọc trạng thái để dashboard và công cụ kiểm tra sử dụng.
- Phát event cho các cập nhật risk score/quyết định quan trọng.
- Kiểm tra quyền gọi các hàm ghi dữ liệu và bảo vệ invariant của contract.

## Phát triển và kiểm thử

Contract được viết bằng Vyper, quản lý và triển khai thông qua Ape Framework.
Các test pytest cho logic smart contract và on-chain nên được đặt trong
`tests/` khi thư mục kiểm thử được thêm vào repository.

Trước khi triển khai testnet, cần kiểm tra tối thiểu:

- score ngoài miền `0–100` bị từ chối;
- ngưỡng ba mức cảnh báo hoạt động đúng;
- caller không có quyền không thể ghi hoặc thay đổi cấu hình;
- event và dữ liệu lưu on-chain khớp với transaction;
- test coverage, transaction confirmation và gas cost của các hàm chính.

Không commit private key, seed phrase hoặc RPC credential. Địa chỉ contract,
chain ID và thông tin triển khai nên được quản lý qua biến môi trường hoặc file
cấu hình cục bộ không commit.

