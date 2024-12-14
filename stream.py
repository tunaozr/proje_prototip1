import cv2
import json
import time
from ultralytics import YOLO

model = YOLO('/Users/tunaozr/Desktop/prototip1/best.pt')  # best.pt nin pathini kopyala buraya bende burası

cap = cv2.VideoCapture(0)  #burda webcam kullanılıyor
detections_list = []
last_save_time = time.time() #son zamanı tutuyor

save_interval = 2  #2 saniyede bir jsona kaydediyor (dosyayı şişirmemek için)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Kameradan görüntü alınamadı!")
        break

    results = model(frame)

    for result in results:
        detection_info = []
        for box in result.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            class_name = model.names[class_id]

            if class_name:  # burası sadece bir class belirlendiyse jsona atıyor
                detection_info.append({
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": confidence,
                    "bbox": [x1, y1, x2, y2]
                })

        if detection_info and (time.time() - last_save_time > save_interval):
            detections_list.append({
                "frame_index": len(detections_list) + 1, 
                "detections": detection_info
            })

            with open("detections.json", "w") as f:
                json.dump(detections_list, f, indent=4)
            print("JSON dosyasına tespitler kaydedildi.")
            last_save_time = time.time()

    annotated_frame = results[0].plot()
    cv2.imshow("YOLOv8 Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
