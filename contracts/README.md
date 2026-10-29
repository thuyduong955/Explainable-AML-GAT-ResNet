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

## Hướng dẫn cài đặt môi trường và biên dịch

Để biên dịch và làm việc với Smart Contract Vyper cục bộ, các thành viên thực hiện tạo môi trường ảo Python (`.venv`) tại thư mục gốc của dự án:

### 1. Tạo và kích hoạt môi trường ảo (`.venv`)

- **Windows (PowerShell):**
  ```powershell
  python -m venv .venv
  .\.venv\Scripts\Activate.ps1
  ```
- **Linux / macOS:**
  ```bash
  python3 -m venv .venv
  source .venv/bin/activate
  ```

### 2. Cài đặt Ape Framework và plugin Vyper

```bash
pip install --upgrade pip
pip install eth-ape ape-vyper
```
*(Hoặc cài đặt plugin dựa trên file `ape-config.yaml` ở thư mục gốc bằng lệnh `ape plugins install .`)*

### 3. Biên dịch Smart Contract

Chạy lệnh sau tại thư mục gốc của dự án để kiểm tra cú pháp và biên dịch các file `.vy` trong `contracts/`:

```bash
ape compile
```
Kết quả biên dịch sẽ được lưu tự động trong thư mục `.build/` (đã được cấu hình trong `.gitignore`).
*(Lưu ý trên Windows: giữ comment/docstring trong file `.vy` bằng tiếng Anh không dấu để tránh lỗi `charmap` của trình biên dịch Vyper standalone).*

## Đặc tả giao diện `AMLGatekeeper.vy`

### 1. Mức quyết định (Decision Tiers) & Ngưỡng mặc định
- `DECISION_ALLOW = 0` (`"ALLOW"`): Cho phép giao dịch (`score < warn_threshold`, mặc định `< 40`).
- `DECISION_WARN = 1` (`"WARN"`): Cảnh báo / cần rà soát (`warn_threshold <= score < block_threshold`, mặc định `40–74`).
- `DECISION_BLOCK = 2` (`"BLOCK"`): Chặn giao dịch (`score >= block_threshold`, mặc định `>= 75`).

### 2. Hàm ghi trạng thái (Write Functions)
- `set_risk_score(new_score: uint256)`: Cập nhật `risk_score` (`0–100`) và `current_decision` mới nhất, phát event `RiskScoreUpdated`.
- `record_tx_risk(tx_id: uint256, new_score: uint256) -> uint8`: Ghi nhận điểm rủi ro và mức cảnh báo cho một mã giao dịch `tx_id` cụ thể vào `tx_assessments[tx_id]`, phát event `RiskScoreUpdated` và trả về mức quyết định (`0`, `1`, `2`).
- `set_thresholds(new_warn: uint256, new_block: uint256)`: *(Owner only)* Cập nhật ngưỡng cảnh báo và chặn, đảm bảo invariant `0 < new_warn < new_block <= 100`.
- `set_operator(operator: address, is_authorized: bool)`: *(Owner only)* Cấp hoặc thu hồi quyền ghi điểm rủi ro cho địa chỉ operator.
- `transfer_ownership(new_owner: address)`: *(Owner only)* Chuyển quyền sở hữu contract.

### 3. Hàm đọc trạng thái (View / Pure Functions)
- `risk_score() -> uint256`, `current_decision() -> uint8`, `total_assessed() -> uint256`
- `warn_threshold() -> uint256`, `block_threshold() -> uint256`, `owner() -> address`
- `classify_score(score: uint256) -> uint8`: Phân loại thử một điểm rủi ro (`0–100`) theo ngưỡng hiện tại.
- `get_decision_label(decision: uint8) -> String[10]`: Trả về chuỗi `"ALLOW"`, `"WARN"` hoặc `"BLOCK"`.
- `get_tx_assessment(tx_id: uint256) -> Assessment`: Trả về bản ghi `(risk_score, decision, timestamp, assessed)` của `tx_id`.
- `is_tx_blocked(tx_id: uint256) -> bool`: Kiểm tra nhanh giao dịch `tx_id` có bị chặn (`BLOCK`) hay không.

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

