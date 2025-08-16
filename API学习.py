import os
import json
import time
import requests
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class DeepSeekAPI:
    """
    DeepSeek API交互类
    支持多轮对话、流式响应、文件读取、参数配置、对话历史保存与读取等功能
    """

    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = "https://api.deepseek.com/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        self.conversation_history = []
        self.default_params = {
            "model": "deepseek-chat",
            "temperature": 0.7,
            "max_tokens": 2048,
            "top_p": 0.9,
            "stream": False
        }

        # 创建对话历史保存目录
        self.history_dir = os.path.join(os.getcwd(), "对话历史")
        if not os.path.exists(self.history_dir):
            os.makedirs(self.history_dir)

        # 设置带重试机制的会话
        self.session = requests.Session()
        retry_strategy = Retry(
            total=3,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def _handle_response(self, response):
        """处理API响应"""
        if response.status_code != 200:
            error_msg = f"API请求失败: {response.status_code} - {response.text}"
            raise Exception(error_msg)
        return response.json()

    def _stream_response(self, response):
        """处理流式响应"""
        buffer = ""
        for chunk in response.iter_lines():
            if chunk:
                decoded = chunk.decode('utf-8')
                if decoded.startswith("data: "):
                    data = decoded[6:]
                    if data == "[DONE]":
                        break
                    try:
                        json_data = json.loads(data)
                        content = json_data["choices"][0]["delta"].get("content", "")
                        print(content, end='', flush=True)
                        buffer += content
                    except:
                        pass
        print()
        return buffer

    def chat(self, prompt, model=None, temperature=None, max_tokens=None, stream=False, system_message=None):
        """与DeepSeek模型对话"""
        # 更新参数
        params = {**self.default_params, "stream": stream}
        if model: params["model"] = model
        if temperature: params["temperature"] = temperature
        if max_tokens: params["max_tokens"] = max_tokens

        # 构建消息历史
        messages = self.conversation_history.copy()

        if system_message:
            messages.insert(0, {"role": "system", "content": system_message})

        messages.append({"role": "user", "content": prompt})

        # 构造请求体
        payload = {
            "messages": messages, **params
        }

        # 发送请求
        endpoint = f"{self.base_url}/chat/completions"
        try:
            if stream:
                print("模型回复: ", end='', flush=True)
                response = self.session.post(
                    endpoint,
                    headers=self.headers,
                    json=payload,
                    stream=True
                )
                full_response = self._stream_response(response)
            else:
                response = self.session.post(
                    endpoint,
                    headers=self.headers,
                    json=payload
                )
                result = self._handle_response(response)
                full_response = result["choices"][0]["message"]["content"]
                print(f"模型回复: {full_response}")

            # 更新对话历史
            self.conversation_history.append({"role": "user", "content": prompt})
            self.conversation_history.append({"role": "assistant", "content": full_response})

            # 自动保存对话历史
            self.save_history()

            return full_response

        except Exception as e:
            print(f"请求发生错误: {str(e)}")
            return None

    def read_file(self, file_path):
        """读取文件并添加到对话上下文"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()

            self.conversation_history.append({
                "role": "system",
                "content": f"用户上传了文件: {os.path.basename(file_path)}\n文件内容:\n{content}"
            })
            print(f"已读取文件: {file_path} ({len(content)}字符)")
            # 保存历史
            self.save_history()
            return content
        except Exception as e:
            print(f"读取文件失败: {str(e)}")
            return None

    def reset_history(self):
        """重置对话历史"""
        self.conversation_history = []
        print("对话历史已重置")

    def print_history(self):
        """打印当前对话历史"""
        print("\n当前对话历史:")
        for i, msg in enumerate(self.conversation_history):
            role = msg["role"]
            content_preview = msg["content"][:50] + "..." if len(msg["content"]) > 50 else msg["content"]
            print(f"{i + 1}. [{role}] {content_preview}")

    def list_models(self):
        """获取可用模型列表"""
        try:
            response = self.session.get(
                f"{self.base_url}/models",
                headers=self.headers
            )
            models = self._handle_response(response)["data"]
            print("\n可用模型:")
            for model in models:
                print(f"- {model['id']} (创建时间: {model['created']})")
            return models
        except Exception as e:
            print(f"获取模型列表失败: {str(e)}")
            return []

    # 新增功能：保存对话历史
    def save_history(self, filename=None):
        """保存当前对话历史到文件"""
        if not self.conversation_history:
            print("对话历史为空，无需保存")
            return

        # 生成默认文件名（时间戳+随机数）
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"history_{timestamp}.json"

        file_path = os.path.join(self.history_dir, filename)

        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.conversation_history, f, ensure_ascii=False, indent=2)
            print(f"对话历史已保存至: {file_path}")
        except Exception as e:
            print(f"保存对话历史失败: {str(e)}")

    # 新增功能：加载对话历史
    def load_history(self, filename):
        """从文件加载对话历史"""
        file_path = os.path.join(self.history_dir, filename)

        if not os.path.exists(file_path):
            print(f"文件不存在: {file_path}")
            return False

        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.conversation_history = json.load(f)
            print(f"已加载对话历史: {file_path} (共{len(self.conversation_history)}条消息)")
            return True
        except Exception as e:
            print(f"加载对话历史失败: {str(e)}")
            return False

    # 新增功能：列出保存的对话历史
    def list_saved_histories(self):
        """列出所有保存的对话历史文件"""
        if not os.path.exists(self.history_dir):
            print("没有保存的对话历史")
            return []

        files = [f for f in os.listdir(self.history_dir) if f.endswith('.json')]
        if not files:
            print("没有保存的对话历史")
            return []

        print("\n保存的对话历史:")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        return files


# 主程序
def main():
    # 初始化API
    api_key = input("请输入DeepSeek API密钥: ").strip()
    deepseek = DeepSeekAPI(api_key)

    # 显示可用模型
    deepseek.list_models()

    # 询问是否加载历史对话
    saved_histories = deepseek.list_saved_histories()
    if saved_histories:
        load_choice = input("\n是否加载历史对话? (输入序号，如1；不加载请按回车): ").strip()
        if load_choice.isdigit():
            idx = int(load_choice) - 1
            if 0 <= idx < len(saved_histories):
                deepseek.load_history(saved_histories[idx])

    print("\nDeepSeek API交互程序已启动")
    print("可用命令:")
    print("  /file <路径> - 读取文件到上下文")
    print("  /reset - 重置对话历史")
    print("  /history - 查看当前对话历史")
    print("  /params - 查看当前参数设置")
    print("  /save [文件名] - 手动保存对话历史（文件名可选）")
    print("  /load <文件名> - 加载指定对话历史")
    print("  /list - 列出所有保存的对话历史")
    print("  /exit - 退出程序")

    while True:
        try:
            # 获取用户输入
            user_input = input("\n用户输入: ").strip()

            if not user_input:
                continue

            # 处理命令
            if user_input.startswith('/'):
                parts = user_input.split(maxsplit=1)
                command = parts[0].lower()
                args = parts[1] if len(parts) > 1 else ""

                if command == '/exit':
                    # 退出前自动保存
                    if deepseek.conversation_history:
                        save_choice = input("是否保存当前对话历史? (y/n): ").lower() == 'y'
                        if save_choice:
                            deepseek.save_history()
                    print("程序已退出")
                    break

                elif command == '/reset':
                    deepseek.reset_history()

                elif command == '/history':
                    deepseek.print_history()

                elif command == '/params':
                    print("\n当前参数设置:")
                    for key, value in deepseek.default_params.items():
                        print(f"  {key}: {value}")

                elif command == '/file' and args:
                    deepseek.read_file(args)

                elif command == '/save':
                    deepseek.save_history(args)

                elif command == '/load' and args:
                    deepseek.load_history(args)

                elif command == '/list':
                    deepseek.list_saved_histories()

                else:
                    print("未知命令，可用命令: /file, /reset, /history, /params, /save, /load, /list, /exit")
                continue

            # 处理普通对话
            stream = input("使用流式输出? (y/n): ").lower() == 'y'
            deepseek.chat(user_input, stream=stream)

        except KeyboardInterrupt:
            print("\n程序已中断")
            break
        except Exception as e:
            print(f"发生错误: {str(e)}")


if __name__ == "__main__":
    main()