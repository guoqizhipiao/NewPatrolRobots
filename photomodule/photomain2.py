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
from photomodule.passerbyremove import people_remover

class photo():
    def __init__(self, frame_queue, start_event, stop_event, filere, passerby_remove, name, shape):
        # 共享内存
        self.name = name
        # 图像形状
        self.shape = shape
        # 创建共享内存对象
        self.shm = shared_memory.SharedMemory(name=self.name)
        # 创建numpy数组
        self.frame = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)

        self.frame_queue = frame_queue
        self.start_event = start_event
        self.stop_event = stop_event
        # 加载滤镜
        self.filere = filere
        # 滤镜目录
        self.filere_dir = os.path.join(parent_dir, "static", "filters")
        self.filere_images = []
        self.load_filere()
        # 图像转换类
        self.handle = frame_handle()
        # 滤镜位置大小
        self.position = (0,0)
        self.size = 1
        #加载路人消除模块
        self.people_remover_model = people_remover()
        self.passerby_remove = passerby_remove
        #启动循环
        self.run()

    def load_filere(self):
        # 加载滤镜文件
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
            # 获取图像
            frame = self.frame.copy()
            # 检查选择滤镜
            if self.filere[0] != 0:
                #print(self.filere[0])
                frame = self.handle.cv2_to_pillow(frame)
                frame = self.handle.add_content_to_image(frame, self.filere_images[self.filere[0]-1], self.position, self.size)
                frame = self.handle.pillow_to_cv2(frame)
            # 检查路人消除
            if self.passerby_remove.is_set():
                frame = self.people_remover_model.remove_people(frame)
            # 将图像放入队列
            if self.frame_queue.full():
                self.frame_queue.get()
            self.frame_queue.put(frame)
            time.sleep(0.01)
        print("photo进程已停止")

def photo_main(q, start_event, stop_event, filere, passerby_remove, name, shape):
    photo_process = Process(target=photo, args=(q, start_event, stop_event, filere, passerby_remove, name, shape))
    photo_process.start()

if __name__ == "__main__":
    # 用例
    q = Queue(maxsize=5)

    start_event = Event()
    stop_event = Event()
    filere = Array("i", [0])

    camera = camera_shared_memory()
    camera.start_camera()
    camera.start_event.set()
    print(1)
    photo_main(q, start_event, stop_event, filere, camera.shm.name, camera.shape)
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


