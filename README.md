# Driver Drowsiness Detection

Hệ thống phát hiện ngủ gật và ngáp cho tài xế ô tô bằng Deep Learning.

## Công nghệ sử dụng

- Python
- TensorFlow / Keras
- CNN V3
- MediaPipe FaceMesh
- OpenCV
- Flask
- SQLite
- HTML, CSS, JavaScript

## Chức năng chính

- Nhận diện khuôn mặt qua webcam.
- Phát hiện mắt nhắm bằng CNN kết hợp EAR.
- Sử dụng EAR để cảnh báo dự phòng.
- Phát hiện ngáp thông qua MAR.
- Phát âm thanh cảnh báo khi phát hiện ngủ gật.
- Lưu lịch sử cảnh báo vào SQLite.
- Hiển thị thống kê và lịch sử trên Dashboard.
- Xuất dữ liệu cảnh báo ra CSV.
- Bật và tắt camera trực tiếp trên Dashboard.

## Cách chạy trên Windows

### Cách 1: Khởi động nhanh

Nhấp đúp file `start.bat`.

Sau đó mở:

http://127.0.0.1:5000

### Cách 2: Chạy bằng Terminal

Mở PowerShell tại thư mục dự án:

```powershell
.\.venv\Scripts\Activate.ps1
python app.py
```

## Cài đặt môi trường trên máy mới

Cần cài Python tương thích với các thư viện trong requirements.txt.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Cần có file `models/eye_cnn_v3.keras` trước khi chạy.

## Thông số phát hiện

- Ngưỡng CNN: 0.5
- Ngưỡng EAR: 0.21
- Thời gian phát hiện ngủ gật CNN + EAR: 2 giây
- Thời gian cảnh báo dự phòng EAR: 3 giây
- Ngưỡng MAR: 0.50
- Thời gian xác nhận ngáp: 1.5 giây

## Lưu ý

- Sử dụng trên Windows và cần webcam hoạt động.
- Không đóng cửa sổ CMD trong khi đang sử dụng Dashboard.
- Đây là phần mềm nghiên cứu và demo đồ án, chưa được kiểm chứng để thay thế các thiết bị an toàn khi lái xe thực tế.
