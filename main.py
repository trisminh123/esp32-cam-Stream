import cv2
import urllib.request
import numpy as np
import time
from ultralytics import YOLO

# Load mô hình YOLOv8 với file train best.pt
model = YOLO("best.pt")

# Địa chỉ IP của ESP32-CAM (Cập nhật đúng địa chỉ của bạn)
url = "http://192.168.43.211/capture"


def capture_images_continuously(url):
    """Liên tục chụp ảnh từ ESP32-CAM và xử lý bằng YOLO"""
    while True:
        try:
            # Lấy ảnh từ ESP32-CAM
            with urllib.request.urlopen(url) as stream:
                img_array = np.asarray(bytearray(stream.read()), dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

                if img is not None:
                    # Phát hiện lửa bằng YOLO
                    results = model(img)

                    # Vẽ bounding boxes nếu phát hiện lửa
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            confidence = box.conf[0].item() * 100  # Convert confidence
                            if confidence >= 60:  # Chỉ hiển thị khi confidence >= 60%
                                x1, y1, x2, y2 = map(int, box.xyxy[0])
                                class_id = int(box.cls[0].item())
                                label = model.names[class_id] if class_id < len(model.names) else "Unknown"

                                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                                cv2.putText(
                                    img,
                                    f"{label} {confidence:.1f}%",
                                    (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.9,
                                    (0, 255, 0),
                                    2,
                                )

                    # Hiển thị ảnh
                    cv2.imshow("ESP32-CAM Capture", img)

            # Dừng 0.5 giây trước khi chụp ảnh tiếp theo (giảm tải)
            time.sleep(0.5)

            # Nhấn 'q' để thoát
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        except urllib.error.HTTPError as e:
            print(f"HTTP Error: {e.code} - {e.reason}")
            time.sleep(1)
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(1)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    capture_images_continuously(url)
