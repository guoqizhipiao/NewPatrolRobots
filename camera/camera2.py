from multiprocessing import shared_memory
import numpy as np
import cv2
from multiprocessing import Event, Process
import time

cap_number = 1

def process_camera(start_event, shm_name, frame_event, shape, fps):
    """子进程，负责从摄像头读取图像，并写入共享内存
        start_event: 控制摄像头开始/停止捕获的信号
        shm_name: 共享内存的名称
        frame_event: 是否有读取帧的信号
        shape: 视频帧的形状
        fps: 视频帧的帧率
    """
    # 创建共享内存对象，使用指定的名称
    shm = shared_memory.SharedMemory(name=shm_name)
    # 创建一个NumPy数组，使用共享内存的缓冲区作为存储空间
    # 数组形状和类型与视频帧匹配
    frame_buffer = np.ndarray(shape, dtype=np.uint8, buffer=shm.buf)

    # 初始化视频捕获对象，使用默认摄像头（索引0）
    cap = cv2.VideoCapture(cap_number)
    # 设置视频捕获的宽度
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, shape[1])
    # 设置视频捕获的高度
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, shape[0])
    # 设置视频捕获的帧率
    cap.set(cv2.CAP_PROP_FPS, fps)
    # 进入无限循环，持续捕获视频帧
    while True:
        # 检查开始事件是否已设置，如果未设置则暂停1秒
        if not start_event.is_set():
            time.sleep(1)
            continue
        # 读取一帧视频
        ret, frame = cap.read()
        # 如果读取失败，跳过当前循环
        if not ret:
            continue
        # 将捕获的帧复制到共享内存缓冲区
        frame_buffer[:] = frame
        # 设置帧事件，表示新的一帧已准备好
        frame_event.set()  
        # 短暂休眠，控制循环频率
        time.sleep(0.01)


class camera_shared_memory():
    """一个使用 共享内存 传递子进程图像的类"""
    def __init__(self):

        # 设置图像的基本属性
        self.shape = (480, 640, 3)  # 定义图像的形状为480x640像素，3通道(RGB)
        self.fps = 30  # 设置帧率为30帧每秒
        # 计算图像总大小并创建共享内存
        size = int(np.prod(self.shape))  # 计算图像数组的总元素个数
        self.shm = shared_memory.SharedMemory(create=True, size=size)  # 创建指定大小的共享内存
        self.frame_buffer = np.ndarray(self.shape, dtype=np.uint8, buffer=self.shm.buf)  # 创建使用共享内存缓冲区的NumPy数组
        # 创建事件对象用于线程间通信
        self.start_event = Event()  # 用于标记相机是否开始的事件
        self.frame_event = Event()  # 用于标记新帧是否可用的事件
        # 初始化相机进程变量
        self.P_camera = None  # 将用于存储相机进程的引用
    
    def start_camera(self):
        """
        启动摄像头处理进程
        此方法创建一个新的进程来处理摄像头任务，并启动该进程
        """
        self.P_camera = Process(                              # 创建一个新的进程对象
            target=process_camera,                                                           # 指定进程要执行的目标函数
            args=(self.start_event, self.shm.name, self.frame_event, self.shape, self.fps)  # 传递给目标函数的参数元组
        )
        self.P_camera.start()                                 # 启动进程
    
    def get_frame(self):
        """从共享内存中复制一帧图像"""
        # 等待帧事件触发，确保有新帧可用
        self.frame_event.wait()
        # 清除帧事件，为下一帧做准备
        self.frame_event.clear()
        # 返回帧缓冲区的副本，避免外部修改影响内部数据
        return self.frame_buffer.copy()
    
    def close(self):

        """
        关闭摄像头和共享内存资源的方法
        """
        self.start_event.clear()  # 清除开始事件，停止线程运行
        if self.P_camera:  # 检查摄像头进程是否存在
            self.P_camera.terminate()  # 终止摄像头进程
            self.P_camera.join()  # 等待摄像头进程结束
        self.shm.close()  # 关闭共享内存
        self.shm.unlink()  # 从系统中删除共享内存




# 使用样例        
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






