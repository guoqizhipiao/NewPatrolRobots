from multiprocessing import Process, Queue, Event, shared_memory, Array 
import numpy as np
import os
import sys
from PIL import Image
import time
import cv2

curr_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

from camera.camera2 import camera_shared_memory
from photomodule.framehandle import frame_handle

class potho():
    def __init__(self, frame_queue, start_event, stop_event, filere, name, shape):
        self.name = name
        self.shape = shape
        self.shm = shared_memory.SharedMemory(name=self.name)
        self.frame = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)

        self.frame_queue = frame_queue
        self.start_event = start_event
        self.stop_event = stop_event

        self.filere = filere
        curr_dir = os.path.dirname(__file__)
        self.filere_dir = os.path.join(curr_dir, "filereimage")
        self.filere_images = []
        self.load_filere()

        self.handle = frame_handle()
        self.position = (0,0)
        self.size = 1

        self.run()


    def load_filere(self):
        valid_extensions = (".png")
        for filename in os.listdir(self.filere_dir):
            if filename.lower().endswith(valid_extensions):
                img_path = os.path.join(self.filere_dir, filename)
                try:
                    img = Image.open(img_path)
                    self.filere_images.append(img)
                except Exception as e:
                    print(f"无法加载图像：{img_path} - {str(e)}")
        print(self.filere_images)
        print(f"加载了 {len(self.filere_images)} 个滤镜图像")

    def run(self):
        while not self.stop_event.is_set():
            if not self.start_event.is_set():
                time.sleep(1)
                continue

            frame = self.frame.copy()
            if self.filere[0] != 0:
                #print(self.filere[0])
                frame = self.handle.cv2_to_pillow(frame)
                frame = self.handle.add_content_to_image(frame, self.filere_images[self.filere[0]-1], self.position, self.size)
                frame = self.handle.pillow_to_cv2(frame)
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)
            time.sleep(0.01)

        print("potho进程已停止")

def potho_main(q, start_event, stop_event, filere, name, shape):
    photo = Process(target=potho, args=(q, start_event, stop_event, filere, name, shape))
    photo.start()

if __name__ == "__main__":
    q = Queue(maxsize=5)

    start_event = Event()
    stop_event = Event()
    filere = Array("i", [0])

    camera = camera_shared_memory()
    camera.start_camera()
    camera.start_event.set()
    print(1)
    potho_main(q, start_event, stop_event, filere, camera.shm.name, camera.shape)
    print(2)
    start_event.set()
    print(3)
    import msvcrt
    while True:
        if not q.empty():
            frame = q.get()
            frame = cv2.resize(frame, (640, 480))
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            time.sleep(0.01)
        else:
            time.sleep(1)

        if msvcrt.kbhit():
            line = msvcrt.getch().decode('utf-8')
            if line == "1":
                filere[0] = 1
            elif line == "2":
                filere[0] = 2
            elif line == "3":
                filere[0] = 3

    stop_event.set()
    camera.close()


