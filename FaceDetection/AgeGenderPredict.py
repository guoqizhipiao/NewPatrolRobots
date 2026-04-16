from ultralytics import YOLO
import cv2
import os

curr_dir = os.path.dirname(os.path.abspath(__file__))
models = os.path.join(curr_dir, "models")

face_detector_path = os.path.join(models, "yolov8n-face.pt")
gender_classifier = os.path.join(models, "AgeGender_cls", "weights", "best.pt")

# 该类用于执行年龄与性别的预测，预测100次(预测出现超过60次人脸,低于60次则忽略)
# 该类已经完全封装OK了只需要在类外实例化一个对象,然后调用Get_User_AgeGender即可,并且该函数返回的是一个列表
class AgeGenderPredict:
    def __init__(self):
        self.__AgeGender = []
        self.__face_detector = YOLO(face_detector_path)
        self.__gender_classifier = YOLO(gender_classifier)

    def _Predict(self):
        cap = cv2.VideoCapture(0)
        dict = {"female_elders":0,"female_kids":0,"female_young":0,"male_elders":0,"male_kids":0,"male_young":0}
        results = []

        max_num_of_face = 0     # 记录最多检测到的人脸数量
        count = 0               # 控制检测次数

        while count<50:
            ret, frame = cap.read()
            if not ret:
                print("错误，未检测到图像!")
                break

            # 人脸定位
            det_results = self.__face_detector(frame)
            # 记录本次检测到的人脸数量
            num_of_face = len(det_results[0].boxes)

            if max_num_of_face < num_of_face:
                for _ in range(num_of_face - max_num_of_face):
                    results.append(dict.copy())
                    max_num_of_face = num_of_face

            # 用来记录下方for循环处理到了第几张人脸
            num = 0

            # 处理每个检测结果
            for box in det_results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                face_roi = frame[y1:y2, x1:x2]

                # 性别分类
                cls_results = self.__gender_classifier(face_roi)
                AgeGender = cls_results[0].names[cls_results[0].probs.top1]

                results[num][AgeGender] += 1
                # 绘制结果
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, AgeGender, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                num += 1

            # 预测结果显示,测试结束后将该行注释掉
            cv2.imshow("Real-time Detection", frame)
            if cv2.waitKey(1) == 27:  # ESC退出
                break
            # 一直到这里
            count += 1

        print("最多检测到{}张人脸！".format(max_num_of_face))
        for result in results:
            # 当一张人脸出现超过60次就计入
            if sum(result.values()) >= 30:
                self.__AgeGender.append(max(result, key=result.get))

        cap.release()
        cv2.destroyAllWindows()


    def Get_User_AgeGender(self):
        self._Predict()
        return self.__AgeGender