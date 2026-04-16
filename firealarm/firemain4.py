from multiprocessing import Process, Queue, Event, shared_memory
import cv2
import time
import torch
from ultralytics import YOLO
import numpy as np
import os
import sys

curr_path = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_path)
sys.path.insert(0, parent_dir)

from camera.camera2 import camera_shared_memory


class fire_detection():
    def __init__(self, frame_queue, start_event, stop_event, fire_event, name, shape):
        self.frame_queue = frame_queue
        self.start_event = start_event
        self.stop_event = stop_event
        self.fire_event = fire_event

        self.name = name
        self.shape = shape
        self.shm = shared_memory.SharedMemory(name=self.name)
        self.frame = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)

        self.curr_dir = os.path.dirname(__file__)
        self.model_path = os.path.join(self.curr_dir, 'best26.pt')
        self.init_image_path = os.path.join(self.curr_dir, 'init_image.jpg')
        parent_dir = os.path.dirname(self.curr_dir)
        sys.path.insert(0, parent_dir)

        self.model = YOLO(self.model_path)
        self.model.to('cuda')
        self.model.fuse()
        self.init_image = cv2.imread(self.init_image_path)
        self.init_image = cv2.resize(self.init_image, (640, 480))
        results = self.model(self.init_image, verbose=False, workers=0, device='cuda', half=True)
        if results:
            print("模型加载成功", self.model_path)

        self.detection_interval = 2
        self.last_detection_time = 0
        self.testsnumber = 100

        self.inference_size = 320

        self.detect_every = 6
        self.frame_id = 0

        self.window_time = 5
        self.window_start = time.time()
        self.fire_count = 0
        self.detect_count = 0

    def frame_detection(self, frame):
        is_fire = False
        frame = cv2.resize(frame, (self.inference_size, int(self.inference_size * 0.75)))
        results = self.model(frame, verbose=False, stream=True)
        if results:
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    if len(boxes) == 0:
                        continue

                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    
                    if result.names[cls] == "fire" and conf > 0.2:
                        is_fire = True
                    
                    xyxy = box.xyxy[0].int().cpu().numpy()
                    x1, y1, x2, y2 = xyxy

                    label = f"{result.names[cls]} {conf:.2f}"
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2) 
                
        return (frame, is_fire)

    def run(self):
        print("""火焰检模块启动！""")

        while not self.stop_event.is_set():
            if self.start_event.is_set():
                self.frame_id += 1

                frame = self.frame.copy()


                if self.frame_id % self.detect_every == 0:
                    #print("检测中...")
                    is_fire = False
                    frame, is_fire = self.frame_detection(frame)
                    self.detect_count += 1
                    if is_fire:
                        self.fire_count += 1
                if time.time() - self.window_start > self.window_time:
                    ratio = self.fire_count / max(self.detect_count, 1)
                    if ratio > 0.4:
                        print(f"检测到疑似火焰的次数: {self.fire_count} / {self.detect_count}")
                        self.fire_event.set()
                    else:
                        self.fire_event.clear()

                    self.fire_count = 0
                    self.detect_count = 0
                    self.window_start = time.time()

                if self.frame_queue.full():
                    self.frame_queue.get()
                self.frame_queue.put(frame)
            else:
                #print("等待")
                time.sleep(0.1)
            time.sleep(0.05)

        print("""火焰检测已关闭！""")

def fire_main(frame_queue, start_event, stop_event, fire_event, name, shape):
    detector = fire_detection(frame_queue, start_event, stop_event, fire_event, name, shape)
    detector.run()


if __name__ == '__main__':
    q = Queue(maxsize=5)
    start_event = Event()
    stop_event = Event()
    fire_event = Event()

    start_event.set()

    camera = camera_shared_memory()
    camera.start_camera()
    camera.start_event.set()

    name = camera.shm.name
    shape = camera.shape

    work = Process(target=fire_main, args=(q, start_event, stop_event, fire_event, name, shape))
    work.start()

    while True:
        if not q.empty():
            frame = q.get()
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("frame", frame)
            if fire_event.is_set():
                print("检测到疑似火焰")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


