from ultralytics import YOLO
import cv2
import numpy as np
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'yolov8n.pt')

class people_remover:

    """
    一个用于移除图像中人物的类，保留最大的人（作为主体），并自动修复背景。
    使用YOLOv8模型进行人物检测，然后使用OpenCV的inpaint功能修复背景。
    """
    def __init__(self):

        """
        初始化函数，自动下载YOLOv8n模型
        """
        self.model = YOLO(model_path)
        print("路人移除模块已加载")

    def remove_people(self, image):

        result = self.model(image, verbose=False)[0]

        mask = np.zeros(image.shape[:2], dtype=np.uint8)

        persons = []

        for box in result.boxes:

            cls = int(box.cls[0])

            # COCO类别0 = person
            if cls == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2))

        if len(persons) == 0:
            return image

        #  保留最大的人（拍照主体）
        areas = [(x2-x1)*(y2-y1) for x1,y1,x2,y2 in persons]
        main_id = np.argmax(areas)

        # 删除其他人
        for i, (x1, y1, x2, y2) in enumerate(persons):

            if i == main_id:
                continue

            mask[y1:y2, x1:x2] = 255

        # 扩张避免残影
        kernel = np.ones((5,5), np.uint8)
        mask = cv2.dilate(mask, kernel)

        # 修补背景
        output = cv2.inpaint(image, mask, 7, cv2.INPAINT_TELEA)

        return output
    
# 使用示例
if __name__ == "__main__":
    cap = cv2.VideoCapture(1)  # 打开摄像头
    remover = people_remover()
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 去除人像
        frame = remover.remove_people(frame)

        # 显示结果
        cv2.imshow('People Removed', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break