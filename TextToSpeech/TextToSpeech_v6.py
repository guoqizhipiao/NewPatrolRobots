# tts_threaded_fixed.py
import pyttsx3
import threading
import queue
import pythoncom
from typing import Optional

class TextToSpeech:
    def __init__(self):
        self._task_queue = queue.Queue()
        self._running = True
        # 启动一个专用的 TTS 工作线程（只此一个！）
        self._worker_thread = threading.Thread(target=self._tts_worker, daemon=True)
        self._worker_thread.start()

    def _tts_worker(self):
        """专用 TTS 线程：顺序处理所有语音任务"""
        pythoncom.CoInitialize()  # 仅在此线程初始化 COM
        engine = pyttsx3.init()
        try:
            while self._running:
                try:
                    task = self._task_queue.get(timeout=1)
                    if task is None:
                        break

                    text, volume, rate, save_path = task
                    engine.setProperty('volume', volume)
                    engine.setProperty('rate', rate)

                    if save_path:
                        engine.save_to_file(text, save_path)
                        engine.runAndWait()
                        print(f"💾 音频已保存至：{save_path}")
                    else:
                        engine.say(text)
                        engine.runAndWait()

                    self._task_queue.task_done()
                except queue.Empty:
                    continue
        except Exception as e:
            print(f"❌ TTS 工作线程错误：{e}")
        finally:
            engine.stop()
            pythoncom.CoUninitialize()

    def text_to_speech(
        self,
        text: str,
        volume: float = 1.0,
        rate: int = 150,
        save_path: Optional[str] = None
    ):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("文本不能为空")
        if not (0.0 <= volume <= 1.0):
            raise ValueError("音量必须在 0.0 到 1.0 之间")
        if not (1 <= rate <= 300):
            raise ValueError("语速必须在 1 到 300 之间")

        self._task_queue.put((text.strip(), volume, rate, save_path))

    def shutdown(self, wait=True):
        """可选：优雅关闭"""
        self._running = False
        self._task_queue.put(None)  # 停止信号
        if wait and self._worker_thread.is_alive():
            self._worker_thread.join()


# ==================== 测试 ====================
if __name__ == "__main__":
    import time
    print("🚀 启动多个语音任务（通过队列串行播放）...")
    tts = TextToSpeech()  # 只创建一个实例！
    tts.text_to_speech("这是第一条消息")
    tts.text_to_speech("这是第二条消息")
    tts.text_to_speech("这是第三条消息。")
    time.sleep(5)
    print("✅ 主线程结束")