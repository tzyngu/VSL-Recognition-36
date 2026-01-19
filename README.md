# VSL Recognition AI (Vietnamese Sign Language)

## 📖 Giới thiệu (Description)
Hệ thống nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL) thời gian thực, hỗ trợ **36 ký hiệu** cơ bản. Dự án sử dụng **MediaPipe** để trích xuất đặc trưng bàn tay (Landmarks) và mô hình **Deep Learning (LSTM)** để phân loại chuỗi cử chỉ.

**Tính năng chính:**
- 🖐️ Nhận diện 36 ký hiệu VSL (Chữ cái & Từ ngữ thông dụng).
- ⚡ Xử lý thời gian thực với độ trễ thấp.
- 🎨 Giao diện Streamlit hiện đại (Dark Luxury Theme).

## 🛠️ Công nghệ (Tech Stack)
- **Core AI**: TensorFlow/Keras (LSTM Model).
- **Computer Vision**: MediaPipe Hands, OpenCV.
- **Interface**: Streamlit.

## 🚀 Cài đặt & Chạy
```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy ứng dụng
streamlit run streamlit_app.py
```
