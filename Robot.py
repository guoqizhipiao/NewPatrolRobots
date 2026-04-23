import re
import time
import cv2
import json
from multiprocessing import Process, Queue, Event, Array

from FaceDetection.AgeGenderPredict import AgeGenderPredict
from SpeechToText.SpeechToText import SpeechToText
from LLM.DeepSeek import DeepSeek
from TextToSpeech.TextToSpeech_v6 import TextToSpeech
from LLM.CheckStartOllama import check_start_ollama
from firealarm.firemain4 import fire_main
from photomodule.photomain2 import potho_main
from camera.camera2 import camera_shared_memory



class Robot:
    """
       Robot 类即为智巡管理机器人
       主要功能模块包括：
       1. 人脸年龄性别识别：通过 AgeGenderPredict 类检测游客的年龄和性别。
       2. 语音识别：使用 SpeechToText 类将游客的语音转换为文本。
       3. 大语言模型问答：借助 DeepSeek 类与大语言模型交互，根据游客信息生成回答。
       4. 语音合成：利用 TextToSpeech 类将回答转换为语音并朗读。
       主要属性：
       - age_gender: 存储检测到的游客年龄和性别信息。
       - num: 记录检测到的游客数量。
       - user_inputs: 保存游客的输入信息（语音或文本）。
       - user_count: 统计服务的游客数量。
       - answer_part: 存储大语言模型生成的回答内容。
       - think_part: 存储大语言模型生成的思考内容。
       - rate: 语音朗读的语速，范围为 -100% 到 +100%。
       - volume: 语音朗读的音量，范围为 -99 到 0。
       - location: 机器人当前所处的位置。
       """
    def __init__(self):
        self.face_detection = AgeGenderPredict()                     # 加载人脸性别年龄预测类
        print("=" * 20, "年龄性别预测模块始化成功！", "=" * 20)
        self.speech_to_text = SpeechToText()                         # 加载语音转文本类
        print("=" * 20, "语音转文本模块始化成功！", "=" * 20)
        ret = check_start_ollama()
        if not ret:
            print("Ollama服务启动失败，请手动启动Ollama服务！")
        self.deepseek = DeepSeek()                                   # 加载大语言模型类
        print("=" * 20, "大语言模型模块始化成功！", "=" * 20)
        self.deepseek.build_langchain()                              # 构建langchain链路
        self.text_to_speech = TextToSpeech()                         # 加载文本转换语音类
        print("=" * 20, "文本转语音模块始化成功！", "=" * 20)

        self.camera = camera_shared_memory() # 加载摄像头类
        self.camera.start_camera() # 启动摄像头

        self.camera_shm_name = self.camera.shm.name # 获取共享内存名称
        self.camera_shape = self.camera.shape # 获取摄像头分辨率

        self.using_camera = 0 # 使用摄像头模块数量

        self.processed_fire_frame_queue = Queue(maxsize=5) #处理后的帧

        self.fire_stop_event = Event() #关闭进程
        self.fire_start_event = Event() #开启检测
        self.fire_event = Event() #检测到火焰
        self.fire_status = "no_fire"# 火焰状态
        self.fire_debounce_time = 5 #火焰检测防抖时间
        # 加载火焰检测进程
        self.fire_process = Process(
            target=fire_main,
              args=(
                self.processed_fire_frame_queue,
                self.fire_start_event,
                self.fire_stop_event,
                self.fire_event,
                self.camera_shm_name,
                self.camera_shape))
        
        self.fire_process.start()# 启动火焰检测进程
        print("=" * 20, "火焰检测模块始化成功！", "=" * 20)

        print("#" * 20, "机器人初始化成功！", "#" * 20)

        self.age_gender = ""
        self.num = 0
        self.user_inputs = None
        self.user_count = 1
        self.answer_part = ""
        self.think_part = ""
        self.rate = 200  # 语速，范围为-100%到+100%    +100%最快   #pyttsx3   1~300
        self.volume = 0.5  # 音量，范围为-99到0         0最大      #pyttsx3   0.0~1.0
        self.location = "红门"

        self.photo_start_event = Event() # 开启拍照进程
        self.photo_stop_event = Event() # 关闭拍照进程
        self.photo_filere = Array("i", [0]) # 滤镜编号
        self.photo_frame_queue = Queue(maxsize=5) # 拍照队列
        # 加载拍照进程
        self.photo_module = potho_main(
            self.photo_frame_queue,
            self.photo_start_event,
            self.photo_stop_event,
            self.photo_filere,
            self.camera_shm_name,
            self.camera_shape)
        self.photo_status = False

    # 分割答案与思考
    def _split_think_answer(self,input_str):
        # 删除"*"与"#"
        input_str = input_str.replace('*', '')
        input_str = input_str.replace('#', '')
        # 使用正则表达式查找 <think> 到 </think> 之间的内容
        pattern = r'<think>(.*?)</think>'
        match = re.search(pattern, input_str, re.DOTALL)

        if match:
            thought_part = match.group(1).strip()
            # 去除 <think> 标签及其内部内容
            answer_part = input_str.replace(f'<think>{match.group(1)}</think>', '').strip()
            return [thought_part, answer_part]
        else:
            return ["未找到思考部分的内容。", input_str.strip()]

    # 检测年龄与性别
    def detect_age_gender(self):
        age_gender = self.face_detection.Get_User_AgeGender()
        if "female_elder" in age_gender or "male_elder" in age_gender:
            self.volume = 1
            self.rate = 150
        else:
            self.rate = 0.5
            self.volume = 200
        self.num = len(age_gender)
        self.age_gender = " and ".join(str(item) for item in age_gender)

    # 语音转文本
    def detect_speech(self):
        self.speech_to_text.listening()
        self.user_inputs = self.speech_to_text.get_recognized_text()
        if self.user_inputs:
            return 1        # 代表检测到内容
        else:
            return 0        # 代表未检测到内容

    # 检测文本输入
    def detect_write(self):
        self.user_inputs = input("请输入问题！")
        if self.user_inputs:
            return 1        # 同理
        else:
            return 0

    # 朗读
    def _Read(self):
        print("🔊 朗读中...")
        print(f"🗣️ 朗读内容: {self.answer_part}\n")
        self.text_to_speech.text_to_speech(text=self.answer_part, volume=self.volume, rate=self.rate)
        

    # 与大语言模型进行对话
    def talk_to_robot(self):
        try:
            print("🤖 正在与大语言模型进行对话...")
            result = self.deepseek.Ask_LLM(num=self.num, agegender=self.age_gender, query=self.user_inputs, user_id=str(self.user_count))
            self.think_part,self.answer_part = self._split_think_answer(result)
        except Exception as e:
            print("发生异常：", e)
            print("请检查Ollama服务是否正常运行!")

    # 朗读答案
    def read_the_answer(self):
        print("准备朗读答案...")
        self._Read()
        # 用于将用户输入清空
        self.user_inputs = None

    # 更新机器人所处位置
    def change_location(self, location):
        self.location = location
        self.deepseek.location = location

    # 更新用户
    def change_user(self):
        self.user_count += 1

    # 开始服务
    def service_start(self):
        self.text_to_speech.text_to_speech(text="已知晓您的问题，正在思考请稍等！", volume=self.volume, rate=self.rate)
        

    # 结束服务
    def service_end(self):
        self.text_to_speech.text_to_speech(text="您还有其他的问题吗？欢迎您再次提问！祝您游玩愉快！", volume=self.volume, rate=self.rate)
        

    # 测试方法
    def test(self):
        self.num = int(input("请输入人数 int"))
        print(self.num)
        self.age_gender = input("请输入年龄性别 str")
        print(self.age_gender)
        self.user_inputs = input("请输入提问 str")
        print(self.user_inputs)

    def start_camera(self):
        self.using_camera += 1
        if self.using_camera == 1:
            self.camera.start_event.set()

    def stop_camera(self):
        self.using_camera -= 1
        if self.using_camera <= 0:
            self.camera.start_event.clear()

    def start_photo(self):
        if not self.photo_status:
            self.start_camera()
            self.photo_status = True
            self.photo_start_event.set()
    
    def stop_photo(self):
        if self.photo_status:
            self.stop_camera()
            self.photo_status = False
            self.photo_start_event.clear()
            

    def generate_photo_frames(self):
        while True:
            if self.photo_frame_queue:
                frame = self.photo_frame_queue.get()
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)

    def generate_fire_frames(self):
        while True:
            if not self.processed_fire_frame_queue.empty():
                frame = self.processed_fire_frame_queue.get()
                frame = cv2.resize(frame,(640, 480))
                ret, buffer = cv2.imencode('.jpg', frame)
                if ret:
                    frame = buffer.tobytes()
                    yield (b'--frame\r\n'
                            b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.1)

    def fire_status_stream(self):
        yield f"data: {json.dumps({'status': self.fire_status})}\n\n"

        last_sent_status = self.fire_status
        debounce_time_have_fire = False
        last_change_time = 0
        while True:
            now = time.time()
            if self.fire_event.is_set():
                #print("检测到火焰")
                debounce_time_have_fire = True
                last_change_time = now
                self.fire_status = "fire"
            else:
                if not debounce_time_have_fire:
                    self.fire_status = "no_fire"
                else:
                    if now-last_change_time > self.fire_debounce_time:
                        debounce_time_have_fire = False
                        self.fire_status = "no_fire"

            if self.fire_status != last_sent_status:
                yield f"data: {json.dumps({'status': self.fire_status})}\n\n"
                last_sent_status = self.fire_status
            time.sleep(0.1)

    def start_fire_alarm(self):
        self.start_camera()
        self.fire_start_event.set()

    def stop_fire_alarm(self):
        self.stop_camera()
        self.fire_start_event.clear()
    def update_filter(self, filter_id):
        self.photo_filere[0] = int(filter_id)


if __name__ == "__main__":
    print("🚀 开始测试 Robot 核心功能...\n")

    try:
        robot = Robot()
        
        robot.fire_start_event.set()
        robot.camera.start_event.set()

        while robot.processed_fire_frame_queue.empty():
            print("等待火焰检测结果...")
            time.sleep(1)
        print("火焰检测成功运行！")

        # while True:
        #     if not robot.processed_fire_frame_queue.empty():
        #         frame = robot.processed_fire_frame_queue.get()
        #         frame = cv2.resize(frame,(640, 480))
        #         cv2.imshow("frame", frame)                
        #         if cv2.waitKey(1) & 0xFF == ord('q'):
        #             break


        # ===== 模拟用户信息（跳过 detect_age_gender）=====
        robot.num = 1
        robot.age_gender = "female_adult"  # 可选: male_child, female_elder 等
        robot.user_inputs = "泰山有什么好玩的？"

        print(f"👤 模拟游客: {robot.num} 人, 特征: {robot.age_gender}")
        print(f"💬 用户提问: {robot.user_inputs}\n")

        # ===== 模拟服务开始 =====
        print("🔊 播放 '正在思考' 提示音...")
        try:
            pass
            robot.service_start()  # 可能因 TTS 网络问题失败
        except Exception as e:
            print(f"⚠️  service_start 失败（可忽略）: {e}")

        # ===== 调用大模型 =====
        print("🧠 调用大语言模型生成回答...")
        robot.talk_to_robot()
        print(f"💡 思考部分: {robot.think_part}")
        print(f"🗣️ 回答部分: {robot.answer_part}\n")

        # ===== 朗读答案 =====
        if robot.answer_part.strip():
            print("🔊 朗读回答...")
            try:
                pass
                robot.read_the_answer()
            except Exception as e:
                print(f"❌ TTS 播放失败: {e}")
                print("📄 (文本回退) 回答内容:", robot.answer_part)
        else:
            print("❌ 回答为空！")

        # ===== 服务结束 =====
        print("🔚 播放结束语...")
        try:
            pass
            robot.service_end()
        except Exception as e:
            print(f"⚠️  service_end 失败: {e}")

        print("\n✅ 测试完成！")

    except KeyboardInterrupt:
        print("\n🛑 用户中断")
    except Exception as e:
        print(f"\n💥 测试过程中发生严重错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()