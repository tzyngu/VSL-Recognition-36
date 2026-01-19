# VSL Recognition AI (Vietnamese Sign Language)

##  Giới thiệu (Description)
Hệ thống nhận diện Ngôn ngữ Ký hiệu Việt Nam (VSL) thời gian thực, hỗ trợ 36 kí hiệu cơ bản. Dự án sử dụng **MediaPipe** để trích xuất đặc trưng bàn tay (Landmarks) và mô hình **Deep Learning (LSTM)** để phân loại chuỗi cử chỉ. (Real-time Vietnamese Sign Language (VSL) recognition system for 36 gestures. Powered by MediaPipe hand tracking and a Deep Learning (LSTM) model.)

**Tính năng chính:**
-  Nhận diện 36 ký hiệu VSL (Chữ cái & Từ ngữ thông dụng).
-  Xử lý thời gian thực với độ trễ thấp.
-  Giao diện Streamlit hiện đại (Dark Luxury Theme).

##  Công nghệ (Tech Stack)
- **Core AI**: TensorFlow/Keras (LSTM Model).
- **Computer Vision**: MediaPipe Hands, OpenCV.
- **Interface**: Streamlit.

##  Cài đặt & Chạy
```bash
# 1. Cài đặt thư viện
pip install -r requirements.txt

# 2. Chạy ứng dụng
streamlit run streamlit_app.py
```
<img width="474" height="203" alt="image" src="https://github.com/user-attachments/assets/7d0c5f6d-5d50-418a-a601-5f258d30520d" />





<img width="1213" height="521" alt="image" src="https://github.com/user-attachments/assets/54f91f8a-cbe3-4f7a-b8e3-d449d86db7d5" />


