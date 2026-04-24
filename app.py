from Robot import Robot
from flask import Flask, render_template, jsonify, Response, request, stream_with_context 
from flask_cors import CORS
import json
import time
import os
import threading
import sys
import signal

try:
    sys.excepthook = sys.__excepthook__
except AttributeError:
    pass

app = Flask(__name__)
CORS(app)


# 全局取消语音标记
cancel_voice = False

'''@app.after_request
def add_charset(response):
    response.headers["Content-Type"] = "text/html; charset=utf-8"
    return response'''

@app.route('/')
def index():
    return render_template('index.html')

# =====================
# 取消语音接口（新增）
# =====================
@app.route('/cancel_voice', methods=['POST'])
def cancel_voice_route():
    global cancel_voice
    cancel_voice = True
    print("🛑 语音已取消")
    return jsonify(status="canceled")

# =====================
# 语音识别（支持取消）
# =====================
@app.route('/detect_speech', methods=['POST'])
def detect_speech():
    global cancel_voice
    cancel_voice = False  # 重置
    print("语音转文本（可取消）")

    result_status = 0
    result_text = ""

    def run_voice():
        nonlocal result_status, result_text
        if cancel_voice:
            return
        try:
            status = robot.detect_speech()
            result_status = status
            result_text = robot.user_inputs if status == 1 else ""
        except:
            pass

    # 启动线程
    t = threading.Thread(target=run_voice)
    t.start()

    # 等待完成 或 取消
    while t.is_alive():
        if cancel_voice:
            print("❌ 用户取消录音")
            return jsonify(status=0, text=None)
        t.join(timeout=0.2)

    return jsonify(
        status=result_status,
        text=result_text if result_text else None
    )

@app.route('/detect_write', methods=['POST'])
def detect_write():
    print("文本输入")
    data = request.get_json()
    text = data.get('text')
    if text:
        robot.user_inputs = text
        return jsonify({
            'status': 1,
            'text': robot.user_inputs
        })
    else:
        return jsonify({
            'status': 0,
            'text': None
        })

@app.route('/detect_age_gender', methods=['POST'])
def detect_age_gender():
    print("检测年龄与性别")
    robot.detect_age_gender()
    return jsonify({'status': 'success'})

@app.route('/service_start', methods=['POST'])
def service_start():
    print("开始服务")
    #robot.service_start()
    return jsonify({'status': 'success'})

@app.route('/talk_to_robot')
def talk_to_robot():
    print("与大语言模型进行对话")
    try:
        robot.talk_to_robot()
        return jsonify({
            'think_part': robot.think_part,
            'answer_part': robot.answer_part
        })
    except Exception as e:
        print(f"Error in talk_to_robot: {e}")
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/read_the_answer', methods=['POST'])
def read_the_answer():
    print("朗读答案")
    #robot.read_the_answer()
    return jsonify({"status": 1})

@app.route('/service_end', methods=['POST'])
def service_end():
    print("结束服务")
    robot.service_end()
    return jsonify({'status': 'success'})

@app.route('/change_user', methods=['POST'])
def change_user():
    print("更新用户")
    robot.change_user()
    return jsonify({'status': 'success'})

@app.route('/video_fire')
def video_feed():
    return Response(stream_with_context(robot.generate_fire_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/fire_status_stream')
def fire_status_stream():
    return Response(stream_with_context(robot.fire_status_stream()), mimetype='text/event-stream')

@app.route('/start_fire_alarm', methods=['POST'])
def start_fire_alarm():
    print("启动火灾识别")
    robot.start_fire_alarm()
    return jsonify({'status': 'success'})

@app.route('/stop_fire_alarm', methods=['POST'])
def stop_fire_alarm():
    print("关闭火灾识别")
    robot.stop_fire_alarm()
    return jsonify({'status': 'success'})

@app.route('/video_photo')
def video_photo():
    return Response(stream_with_context(robot.generate_photo_frames()), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_photo', methods=['POST'])
def start_camera():
    try:
        print("启动拍摄")
        robot.start_photo()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route('/stop_photo', methods=['POST'])
def stop_camera():
    try:
        print("关闭拍摄")
        robot.stop_photo()
        return jsonify({'status': 'success'})
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})
    
@app.route('/update-filter', methods=['POST'])
def update_filter():
    print("更新滤镜")
    data = request.json
    filter_id = data.get('filterId')
    print(f"滤镜ID: {filter_id}")
    robot.update_filter(filter_id)
    return jsonify({'status': 'success'})

@app.route('/start_passerby_remove', methods=['POST'])
def start_passerby_remove():
    print("启动路人消除")
    robot.start_passerby_remove()
    return jsonify({'status': 'success'})

@app.route('/stop_passerby_remove', methods=['POST'])
def stop_passerby_remove():
    print("关闭路人消除")
    robot.stop_passerby_remove()
    return jsonify({'status': 'success'})

def cleanup_function():
    print("\n正在执行清理任务...")
    robot.stop_camera()
    robot.stop_fire_alarm()
    robot.stop_photo()
    robot.photo_stop_event.set()
    robot.fire_stop_event.set()
    robot.camera.close()
    time.sleep(1)
    print("清理完成，程序退出。")

# 定义信号处理函数
def signal_handler(sig, frame):
    print(f'\n检测到中断信号 (Ctrl+C)，信号编号: {sig}')
    cleanup_function()
    os._exit(0)

if __name__ == '__main__':
    robot = Robot()
    signal.signal(signal.SIGINT, signal_handler)
    print("服务已启动，按 Ctrl+C 停止...")
    try:
        app.run(host='0.0.0.0', port=5000, debug=False)
    except KeyboardInterrupt:
        cleanup_function()
        sys.exit(0)