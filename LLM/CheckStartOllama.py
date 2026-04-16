import requests
import os
import sys
import subprocess
import time

def start_ollama():
    try:
        env = os.environ.copy()
        creationflags = 0
        if sys.platform == 'win32':
            creationflags = 0x00000200 | 0x00000008
        process = subprocess.Popen(
            ['ollama', 'serve'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.PIPE,
            env=env,
            text=True,
            creationflags=creationflags,
            close_fds=True
        )

        for _ in range(10):
            time.sleep(1)
            if check_ollama():
                print("ollama服务已启动")
                print("ollama服务进程ID:", process.pid)
                return process
        print("ollama服务启动超时")
        try:
            process.terminate()
            process.wait()
        except:
            pass
    except:
        print("启动ollama服务失败")
        return False

def check_ollama():
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        if response.status_code == 200:
            return True
        else:
            print(f"⚠ Ollama 服务响应异常: HTTP {response.status_code}")
            return False
    except requests.ConnectionError:
        return False
    except requests.Timeout:
        print("⚠ Ollama 连接超时")
        return False
    except Exception as e:
        print(f"✗ 检查 Ollama 服务失败: {e}")
        return False
    
def check_start_ollama():
    if check_ollama():
        print("ollama服务已启动")
        return True
    else:
        print("尝试启动ollama服务")
        process = start_ollama()
        if process:
            if check_ollama():
                print("ollama服务启动成功")
                return True
            else:
                print("ollama服务启动失败")
                return False
        else:
            print("ollama服务启动失败")
            return False

if __name__ == '__main__':
    check_start_ollama()

