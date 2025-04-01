import cv2
from ultralytics import YOLO

# Load mô hình YOLO segmentation
model = YOLO("D:\1. DU AN\33. Detect sample\Detect-sample\models\seg.pt")

# Mở kết nối đến camera laptop (ID 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể mở camera")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình từ camera")
        break
    # Chạy dự đoán segmentation trên khung hình hiện tại
    results = model(frame)
    # Lấy hình ảnh đã được đánh dấu kết quả (bounding box, mask,...)
    annotated_frame = results[0].plot()
    # Hiển thị kết quảq
    cv2.imshow("Demo", annotated_frame)
    # Nhấn 'q' để thoát vòng lặp
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
