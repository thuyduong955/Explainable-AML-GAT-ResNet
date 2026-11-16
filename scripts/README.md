# `scripts/`

Thư mục này chứa script tích hợp (Integration Bridge) sử dụng thống nhất **Ape Framework** để kết nối đầu ra điểm rủi ro từ mô hình AI với Smart Contract Vyper [`AMLGatekeeper.vy`](../contracts/AMLGatekeeper.vy) trên local testnet.

## Luồng hoạt động của [`ape_bridge.py`](./ape_bridge.py)

1. **Nhận và chuẩn hóa đầu ra từ mô hình AI (`AIRiskPrediction`):**
   - Hỗ trợ đọc kết quả dự đoán dạng batch từ file `.json` / `.csv` (`AI_OUTPUT_PATH`), chạy suy luận từ model artifact trong `models/` (`MODEL_PATH` + `FEATURE_INPUT_PATH`), hoặc nhận trực tiếp ma trận xác suất 2 lớp `[Licit, Illicit]` (`Softmax` / `predict_proba`) qua hàm `extract_predictions_from_model_probs`.
   - Tự động quy đổi xác suất giao dịch bất hợp pháp (`prob_illicit ∈ [0.0, 1.0]`) sang điểm rủi ro nguyên `risk_score ∈ [0, 100]` và kiểm tra hợp lệ trước khi gửi giao dịch.
2. **Kết nối Local Testnet & Smart Contract (`get_or_deploy_gatekeeper`):**
   - Nếu biến môi trường `CONTRACT_ADDRESS` được cung cấp: kết nối trực tiếp tới contract `AMLGatekeeper` đã deploy sẵn (`project.AMLGatekeeper.at(...)`).
   - Nếu chưa có `CONTRACT_ADDRESS`: tự động deploy `AMLGatekeeper` lên mạng local testnet (`ethereum:local:test`) bằng tài khoản test mặc định (`accounts.test_accounts[0]`) hoặc tài khoản cấu hình qua `APE_ACCOUNT_ALIAS`.
3. **Gửi giao dịch On-chain & Kiểm chứng (`submit_prediction_onchain`):**
   - Gửi transaction gọi `record_tx_risk(tx_id, risk_score, sender=...)`.
   - Giải mã và kiểm chứng event `RiskScoreUpdated` từ transaction receipt.
   - Đọc ngược trạng thái on-chain qua `get_tx_assessment(tx_id)`, `get_decision_label(decision)` và `is_tx_blocked(tx_id)` để đảm bảo dữ liệu lưu trên Smart Contract khớp 100% với giao dịch vừa gửi.

---

## Hướng dẫn chạy

Kích hoạt môi trường ảo `.venv` tại thư mục gốc của dự án rồi chạy bằng Ape CLI (hoặc Python):

```bash
ape run ape_bridge
```
*(Hoặc chạy trực tiếp: `python scripts/ape_bridge.py`)*

### Cấu hình qua biến môi trường (Tùy chọn)

Không đặt private key hoặc RPC secret trực tiếp trong mã nguồn. Có thể truyền cấu hình qua biến môi trường:

| Biến môi trường | Ý nghĩa |
| :--- | :--- |
| `AI_OUTPUT_PATH` | Đường dẫn tới file `.json` hoặc `.csv` chứa đầu ra dự đoán của mô hình AI (`tx_id`/`txId` và `prob_illicit` hoặc `risk_score`). |
| `MODEL_PATH` | Đường dẫn tới model artifact (`.joblib`) trong `models/` để chạy suy luận trực tiếp. |
| `FEATURE_INPUT_PATH` | Đường dẫn tới file CSV chứa 166 đặc trưng chuẩn hóa (dùng kèm `MODEL_PATH`). |
| `CONTRACT_ADDRESS` | Địa chỉ contract `AMLGatekeeper` đã deploy sẵn (nếu bỏ trống, script tự deploy mới trên local testnet). |
| `APE_ACCOUNT_ALIAS` | Alias tài khoản Ape dùng để ký giao dịch (nếu bỏ trống, dùng `accounts.test_accounts[0]`). |
| `BRIDGE_RECEIPT_OUT` | Đường dẫn file `.json` để xuất toàn bộ thông tin receipt (`tx_hash`, `block_number`, `gas_used`, `decision_label`,...). |

---

## Tái sử dụng trong Dashboard / Notebook

Các thành phần khác (như `dashboard/`) có thể import trực tiếp các hàm từ `scripts.ape_bridge` để dùng thống nhất trên nền Ape Framework:

```python
from ape import networks
from scripts.ape_bridge import (
    extract_predictions_from_model_probs,
    get_or_deploy_gatekeeper,
    submit_batch_predictions,
)

with networks.parse_network_choice("ethereum:local:test"):
    contract, signer, _ = get_or_deploy_gatekeeper()
    predictions = extract_predictions_from_model_probs(
        tx_ids=[230425980, 232022460],
        probabilities=[[0.88, 0.12], [0.09, 0.91]],
    )
    results = submit_batch_predictions(predictions, contract=contract, sender=signer)
```
