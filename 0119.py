import cv2
import time
from ultralytics import YOLO
import pygame  # 用於播放音效

# 設置影片輸入和結果輸出
input_video_path = "C:\yolov8\999.mp4"  # 替換為你的影片路徑
output_video_path = "C:/8882.mp4"  # 儲存結果的影片路徑

# 設置畫面分辨率
wCam, hCam = 640, 480

# 初始化影片捕獲和結果儲存
cap = cv2.VideoCapture(input_video_path)
cap.set(3, wCam)
cap.set(4, hCam)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 設定編碼格式
out = cv2.VideoWriter(output_video_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (wCam, hCam))
pTime = 0

# 初始化音頻播放器
pygame.mixer.init()
pygame.mixer.music.load("C:\yolov8\ccc.mp3")  # 替換為你的音頻文件路徑

# 加載 YOLO 模型
try:
    model = YOLO("best.pt")
    print("模型加載成功")
except Exception as e:
    print(f"模型加載失敗: {e}")
    exit()

# 定義目標類別
classNames = ["CAT", "PASS"]

# 設置參數
CONFIDENCE_THRESHOLD = 90  # 置信度閾值
AREA_THRESHOLD = 0.5  # 面積占比閾值（30%）
KNOWN_REAL_WIDTH = 0.5  # 已知目標實際寬度（米）
FOCAL_LENGTH = 700  # 攝像頭焦距（單位：像素，需根據攝像頭標定）

# 用於追踪音效播放狀態
is_playing = False

while True:
    success, img = cap.read()
    if not success:
        print("影片播放結束或無法讀取")
        break

    # 調整圖片大小
    img = cv2.resize(img, (wCam, hCam))

    # 執行 YOLO 模型預測
    results = model.predict(img, stream=True)

    # 計算總像素面積
    total_area = wCam * hCam
    category_areas = {name: 0 for name in classNames}  # 初始化各類別面積累計
    display_texts = []  # 用於存儲顯示信息

    for r in results:
        boxes = r.boxes
        for box in boxes:
            # 獲取邊界框坐標
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

            # 獲取置信度
            confidence = round(box.conf[0].item() * 100, 2)
            if confidence < CONFIDENCE_THRESHOLD:
                continue

            # 獲取類別索引和名稱
            cls = int(box.cls[0])
            if cls >= len(classNames):
                print(f"未知類別索引: {cls}")
                continue
            class_name = classNames[cls]

            # 計算邊界框面積並累計到對應類別
            box_area = (x2 - x1) * (y2 - y1)
            category_areas[class_name] += box_area

            # 繪製邊界框和標籤
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            label = f'{class_name} {confidence}%'
            cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # 計算目標框的像素寬度
            object_pixel_width = x2 - x1

            # 使用三角測量法計算距離
            if object_pixel_width > 0:  # 避免除以零
                distance = (KNOWN_REAL_WIDTH * FOCAL_LENGTH) / object_pixel_width
                display_texts.append(f"{class_name}: {distance:.2f} m")
                print(f"目標: {class_name}, 距離: {distance:.2f} 米")

    # 判斷是否有類別達到面積占比閾值
    area_reached_threshold = False
    for category, area in category_areas.items():
        area_ratio = area / total_area
        display_texts.append(f"{category}: {area_ratio * 100:.2f}%")
        if area_ratio > AREA_THRESHOLD:
            area_reached_threshold = True

    # 音效控制
    if area_reached_threshold:
        if not is_playing:  # 確保只播放一次
            pygame.mixer.music.play()
            is_playing = True
    else:
        if is_playing:  # 停止播放
            pygame.mixer.music.stop()
            is_playing = False

    # 在右下角顯示計算距離和類別占比大小
    for i, text in enumerate(display_texts):
        cv2.putText(img, text, (20, hCam - 20 - i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

    # 計算並顯示 FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    # 將結果寫入輸出影片
    out.write(img)

    # 顯示圖像
    cv2.imshow("YOLO Detection", img)

    # 按 'q' 鍵退出
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv2.destroyAllWindows()
pygame.mixer.quit()
