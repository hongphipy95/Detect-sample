import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
import cv2
import numpy as np
import math
from ultralytics import YOLO
from PIL import Image, ImageTk

# -------------------------------
# Cấu hình và tham số
# -------------------------------
MODEL_PATH = r"D:\1. DU AN\33. Detect sample\Detect-sample\models\seg.pt"  
# Mô hình YOLOv8 segmentation với 2 lớp: "tube" và "blood"
TUBE_DIAMETER = 13          # mm
TUBE_HEIGHT_REAL = 100      # mm
TUBE_RADIUS = TUBE_DIAMETER / 2
VOLUME_DEFAULT = 4.5  # Mức máu mặc định (ml)
BLOOD_DEFAULT = (VOLUME_DEFAULT*1000.0)/(math.pi * (TUBE_RADIUS ** 2))  # Mức máu mặc định (mm)
CAMERA_INDEX = 0            # Thay đổi nếu cần mở camera khác

# -------------------------------
# Các hàm xử lý ảnh
# -------------------------------
def load_model(model_path):
    model = YOLO(model_path)
    return model

def compute_blood_volume(fill_height_real, tube_radius):
    volume_mm3 = math.pi * (tube_radius ** 2) * fill_height_real
    volume_ml = volume_mm3 / 1000.0  # 1 ml = 1000 mm^3
    return volume_ml

def process_frame(frame, model):
    """
    Sử dụng mô hình segmentation để phát hiện:
      - "tube": bounding box của ống mẫu xét nghiệm.
      - "blood": mask của máu bên trong ống.
    Dựa vào ROI của tube và mask của blood, tính tỷ lệ máu (fill ratio),
    quy đổi sang chiều cao thực (mm) và tính thể tích mẫu máu.
    Vẽ các thông tin lên ảnh:
      - Bounding box tube (màu xanh lá).
      - Đường mức máu phát hiện (màu xanh dương) dựa vào điểm có y nhỏ nhất của mask blood.
      - Đường mức mặc định (màu vàng) dựa trên BLOOD_DEFAULT.
      - Nếu mức máu phát hiện không đạt (khác VOLUME_DEFAULT), hiển thị "FAILS".
      Nếu không phát hiện được tube, hiển thị thông báo "RỖNG".
    """
    processed_frame = frame.copy()
    results = model(frame)
    if not results:
        print("Không có kết quả segmentation.")
        return processed_frame

    result = results[0]  # Giả sử xử lý một frame

    # Lấy thông tin lớp từ result.boxes.cls (với thứ tự trùng khớp với result.masks)
    tube_index = None
    blood_index = None
    for i, cls in enumerate(result.boxes.cls.tolist()):
        class_name = model.names[int(cls)]
        if class_name.lower() == "tube":
            if tube_index is None:
                tube_index = i
            else:
                cur_box = result.boxes.xyxy[tube_index].cpu().numpy()
                new_box = result.boxes.xyxy[i].cpu().numpy()
                area_cur = (cur_box[2]-cur_box[0]) * (cur_box[3]-cur_box[1])
                area_new = (new_box[2]-new_box[0]) * (new_box[3]-new_box[1])
                if area_new > area_cur:
                    tube_index = i
        elif class_name.lower() == "blood":
            if blood_index is None:
                blood_index = i
            else:
                cur_box = result.boxes.xyxy[blood_index].cpu().numpy()
                new_box = result.boxes.xyxy[i].cpu().numpy()
                area_cur = (cur_box[2]-cur_box[0]) * (cur_box[3]-cur_box[1])
                area_new = (new_box[2]-new_box[0]) * (new_box[3]-new_box[1])
                if area_new > area_cur:
                    blood_index = i

    # Nếu không phát hiện được tube, thông báo "RỖNG" và trả về frame
    if tube_index is None:
        cv2.putText(processed_frame, "Empty", (20,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        return processed_frame

    # Lấy bounding box của tube
    tube_box = result.boxes.xyxy[tube_index].cpu().numpy().astype(int)
    x1, y1, x2, y2 = tube_box
    cv2.rectangle(processed_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    tube_height = y2 - y1

    if blood_index is None:
        print("Blood không được phát hiện!")
        blood_fill_pixel = 0
    else:
        # Lấy mask của blood từ result.masks.data và chuyển sang numpy array
        blood_mask_full = result.masks.data[blood_index].cpu().numpy()
        blood_mask_full = (blood_mask_full > 0.5).astype(np.uint8) * 255

        # Resize mask nếu kích thước không khớp với frame
        if blood_mask_full.shape[0] != frame.shape[0] or blood_mask_full.shape[1] != frame.shape[1]:
            blood_mask_full = cv2.resize(blood_mask_full, (frame.shape[1], frame.shape[0]))

        # Cắt lấy ROI của tube từ blood mask
        tube_roi_blood_mask = blood_mask_full[y1:y2, x1:x2]
        coords = cv2.findNonZero(tube_roi_blood_mask)
        if coords is None:
            blood_fill_pixel = 0
        else:
            min_y = coords[:, 0, 1].min()
            blood_fill_pixel = tube_height - min_y
            blood_level_y = y1 + min_y
            cv2.line(processed_frame, (x1, blood_level_y), (x2, blood_level_y), (255, 0, 0), 2)
            # Hiển thị thông tin thể tích dựa trên mask blood
            blood_height_real = blood_fill_pixel / float(tube_height) * TUBE_HEIGHT_REAL
            volume_ml = compute_blood_volume(blood_height_real, TUBE_RADIUS)
            label_level = f"Volume: {volume_ml:.1f} ml"
            cv2.putText(processed_frame, label_level, (x1+80, blood_level_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    fill_ratio = blood_fill_pixel / float(tube_height) if tube_height > 0 else 0
    blood_height_real = fill_ratio * TUBE_HEIGHT_REAL
    volume_ml = compute_blood_volume(blood_height_real, TUBE_RADIUS)
    cv2.putText(processed_frame, f"Volume: {volume_ml:.1f} ml", (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Vẽ đường mức máu mặc định
    default_fill_pixel = tube_height - int((BLOOD_DEFAULT / TUBE_HEIGHT_REAL) * tube_height)
    default_line_y = y1 + default_fill_pixel
    cv2.line(processed_frame, (x1, default_line_y), (x2, default_line_y), (66, 174, 255), 2)
    cv2.putText(processed_frame, f"Volume Default: {VOLUME_DEFAULT:.1f} ml", (x1+80, default_line_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (66, 174, 255), 2)

    # So sánh kết quả
    if volume_ml > VOLUME_DEFAULT + 1 or volume_ml < VOLUME_DEFAULT - 1:
        cv2.putText(processed_frame, "FAILS", (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    else:
        cv2.putText(processed_frame, "PASS", (x1, y2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return processed_frame

# -------------------------------
# Ứng dụng PyQt5 tích hợp GUI với video webcam
# -------------------------------
class App(QMainWindow):
    def __init__(self, model):
        super().__init__()
        self.setWindowTitle("Blood Volume Detection")
        self.video_height = 640
        self.model = model
        
        # Setup camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        ret, frame = self.cap.read()
        if not ret:
            print("Cannot receive frame from webcam")
            self.close()
            return
            
        # Calculate dimensions
        orig_height, orig_width = frame.shape[:2]
        self.video_width = int(orig_width * (self.video_height / orig_height))
        self.current_frame = None

        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # Create video display label
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.video_label)

        # Create save button
        self.save_btn = QPushButton("SAVE")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: green;
                color: white;
                font-size: 14pt;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #006400;
            }
        """)
        self.save_btn.clicked.connect(self.process_current_frame)
        layout.addWidget(self.save_btn)

        # Setup window size
        self.resize(self.video_width, self.video_height + 80)

        # Setup timer for frame updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(15)  # Update every 15ms

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            # Resize frame
            frame = cv2.resize(frame, (self.video_width, self.video_height))
            
            try:
                processed = process_frame(frame, self.model)
            except Exception as e:
                print("Error processing frame:", e)
                processed = frame.copy()
                
            self.current_frame = processed.copy()
            
            # Convert to QImage and display
            rgb_image = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            self.video_label.setPixmap(QPixmap.fromImage(qt_image))

    def process_current_frame(self):
        if self.current_frame is None:
            return
            
        processed = self.current_frame.copy()
        processed = cv2.resize(processed, (self.video_width, self.video_height))

        # Create new window for results
        self.result_window = ResultWindow(processed)
        self.result_window.show()

    def closeEvent(self, event):
        self.cap.release()
        self.timer.stop()
        event.accept()

class ResultWindow(QWidget):
    def __init__(self, image):
        super().__init__()
        self.setWindowTitle("Detection and Volume")
        
        layout = QVBoxLayout()
        
        # Convert image to QPixmap
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qt_image)
        
        # Display image
        label = QLabel()
        label.setPixmap(pixmap)
        layout.addWidget(label)
        
        self.setLayout(layout)
        self.resize(w, h)

# -------------------------------
# Main
# -------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    model = load_model(MODEL_PATH)
    window = App(model)
    window.show()
    sys.exit(app.exec_())
