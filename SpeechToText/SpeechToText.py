import os
import queue
import time
import sounddevice as sd
from vosk import Model, KaldiRecognizer
import json
import numpy as np

curr_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(curr_dir, "models")
vosk_dir = os.path.join(models_dir, "vosk")
vosk_model_path = os.path.join(vosk_dir, "vosk-model-small-cn-0.3")

class SpeechToText:
    def __init__(
            self,
            model_path=vosk_model_path,
            device=None,
            sample_rate=16000,
            silence_threshold=500
    ):
        self.sample_rate = sample_rate
        self.silence_threshold = silence_threshold
        self.device = self._get_input_device(device)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型未找到: {model_path}")
        self.model = Model(model_path)
        self.recognizer = KaldiRecognizer(self.model, sample_rate)
        self.audio_queue = queue.Queue()
        self.stream = None
        self.recognized_text = None

    def _get_input_device(self, device):
        if device is not None:
            return device
        default_device = sd.default.device
        if isinstance(default_device, tuple):
            return default_device[0]
        return default_device

    def _audio_callback(self, indata, frames, time, status):
        if not self._is_silent(indata):
            self.audio_queue.put(bytes(indata))

    def _is_silent(self, indata):
        return np.linalg.norm(indata) * 10 < self.silence_threshold

    def listening(self, timeout_seconds=None, partial_callback=None):
        try:
            self.stream = sd.RawInputStream(
                samplerate=self.sample_rate,
                blocksize=8000,
                device=self.device,
                dtype="int16",
                channels=1,
                callback=self._audio_callback
            )
            self.stream.start()
            start_time = time.time()

            while True:
                if timeout_seconds and (time.time() - start_time > timeout_seconds):
                    break
                try:
                    data = self.audio_queue.get(timeout=1)
                except queue.Empty:
                    continue
                if self.recognizer.AcceptWaveform(data):
                    result = json.loads(self.recognizer.Result())
                    text = result.get("text", "").strip()
                    if text:
                        self.recognized_text = text
                        break
                else:
                    partial = json.loads(self.recognizer.PartialResult())
                    partial_text = partial.get("partial", "").strip()
                    if partial_text and partial_callback:
                        partial_callback(partial_text)
        except KeyboardInterrupt:
            pass
        finally:
            if self.stream:
                self.stream.stop()
                self.stream.close()

    def get_recognized_text(self):
        text = self.recognized_text
        self.recognized_text = None
        return text


if __name__ == "__main__":

     def print_partial(partial_text):
         print(f"实时片段: {partial_text}")

     recognizer = SpeechToText()
     while True:
         recognizer.listening(partial_callback=print_partial)
         recognized_text = recognizer.get_recognized_text()
         if recognized_text:
             print(f"主程序获取到的文本: {recognized_text}")
             # 这里可以添加主程序对识别结果的其他操作
             print("主程序对识别结果进行其他操作...")

         user_input = input("输入 'q' 结束监听，其他任意键继续监听: ")
         if user_input.lower() == 'q':
             break