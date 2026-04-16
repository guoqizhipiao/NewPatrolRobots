from multiprocessing import shared_memory
import numpy as np
import cv2
from multiprocessing import Event, Process
import time

def process_camera(start_event, shm_name, frame_event, shape, fps):

    shm = shared_memory.SharedMemory(name=shm_name)
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, shape[1])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[0])
    cap.set(cv2.CAP_PROP_FPS, fps)
    while True:
        if not start_event.is_set():
            time.sleep(1)
            continue
        ret, frame = cap.read()
        if not ret:
            continue
        frame_buffer[:] = frame
        frame_event.set()  
        time.sleep(0.01)


class camera_shared_memory():
    """一个使用 共享内存 传递子进程图像的类"""
    def __init__(self):

        self.shape = (480, 640, 3)
        self.fps = 30

        size = int(np.prod(self.shape))
        self.shm = shared_memory.SharedMemory(create=True, size=size)
        self.frame_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)

        self.start_event = Event()
        self.frame_event = Event()

        self.P_camera = None
    
    def start_camera(self):
        self.P_camera = Process(
            target=process_camera,                               
            args=(self.start_event, self.shm.name, self.frame_event, self.shape, self.fps)
        )
        self.P_camera.start()
    
    def get_frame(self):
        self.frame_event.wait()
        self.frame_event.clear()
        return self.frame_buffer.copy()
    
    def close(self):
        self.start_event.clear()
        if self.P_camera:
            self.P_camera.terminate()
            self.P_camera.join()
        self.shm.close()
        self.shm.unlink()

def work_thread(name,shape):
    shm = shared_memory.SharedMemory(name=name)
    frame = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)    
    while True:
        img = frame.copy()
        cv2.imshow('camera', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        time.sleep(0.01)

if __name__ == '__main__':
    camera = camera_shared_memory()
    camera.start_camera()
    camera.start_event.set()

    name = camera.shm.name
    shape = camera.shape
    work = Process(target=work_thread, args=(name,shape))
    work.start()
    work.join()

    camera.close()






