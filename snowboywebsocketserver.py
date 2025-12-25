import asyncio
import threading
import wave
import json
import os
import time
import urllib.parse
import queue
from collections import deque  # 双端队列，用于平滑音频缓冲
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Callable

import numpy as np
import sounddevice as sd
import opuslib
import websockets
import requests
import soxr  # 必须安装: pip install soxr


# ================= 配置与工具部分 =================
# TODO：配置用网页和配置读取，这里不能写死

class Config:
    def __init__(self):
        # WebSocket / HTTP 配置
        # self.ws_url = os.getenv('WS_URL', 'ws://192.168.0.102:8091/ws/xiaozhi/v1/')
        # self.api_url = os.getenv('API_URL', 'http://192.168.0.102:8091/api/chat/upload-audio')
        self.ws_url = os.getenv('WS_URL', 'ws://120.26.145.173:8091/ws/xiaozhi/v1/')
        self.api_url = os.getenv('API_URL', 'http://120.26.145.173:8091/api/chat/upload-audio')

        # 音频参数 (Opus 标准)
        self.input_sample_rate = 16000  # 录音采样率
        self.output_sample_rate = 24000  # 服务端下发采样率 (Opus常用)
        self.channels = 1
        self.frame_duration = 60  # ms

        # 计算帧大小 (Opus层面的帧大小)
        self.input_frame_size = int(self.input_sample_rate * (self.frame_duration / 1000))
        self.output_frame_size = int(self.output_sample_rate * (self.frame_duration / 1000))


config = Config()

# 全局变量
HTTP_PORT = 9897
WS_PORT = 9898


def downmix_to_mono(data, keepdims=False):
    """将多声道混音为单声道"""
    if data.ndim > 1 and data.shape[1] > 1:
        mono = np.mean(data, axis=1)
        if keepdims:
            return mono[:, np.newaxis]
        return mono
    return data


def upmix_mono_to_channels(data, channels):
    """单声道转多声道"""
    if channels == 1:
        return data[:, np.newaxis] if data.ndim == 1 else data
    return np.tile(data[:, np.newaxis], (1, channels))


# ================= 核心客户端类 =================

class AudioClient:
    """
    整合了网络通信与高质量音频处理的客户端
    采用 soxr 重采样 + deque 缓冲池 解决 Windows 平台的沙哑和卡顿问题
    """

    def __init__(self):
        # --- 网络相关 ---
        self.device_id = os.getenv('DEVICE_ID', 'dev-device')
        self.access_token = os.getenv('ACCESS_TOKEN', 'test-token')
        self.HEADERS = {
            "Authorization": f"Bearer {self.access_token}",
            "Protocol-Version": "1",
            "Device-Id": self.device_id,
            "Client-Id": self.device_id,
        }
        self.session_id = None
        self.websocket = None
        self.handing_message_task = None
        self.loop = None  # asyncio loop引用
        self._reconnecting_lock = asyncio.Lock()  # 防止并发重连

        # --- 音频设备参数 ---
        self.input_device_id = None
        self.output_device_id = None

        # 【新增】音量控制倍数，默认 1.5 倍
        self.output_volume = 1.5

        # 实际硬件采样率 (初始化时检测)
        self.device_input_sample_rate = config.input_sample_rate
        self.device_output_sample_rate = config.output_sample_rate
        self.input_channels = 1
        self.output_channels = 1

        # Opus 编解码器
        self.opus_encoder = None
        self.opus_decoder = None

        # 流
        self.input_stream = None
        self.output_stream = None

        # --- 关键：重采样与缓冲 ---
        # 1. 播放缓冲：两级结构
        #    Level 1: _output_buffer (Queue) - 存放解码后的 24k 音频块
        #    Level 2: _resample_output_buffer (deque) - 存放重采样后待播放的采样点
        self._output_buffer = queue.Queue(maxsize=500)
        self._resample_output_buffer = deque()

        # 2. 重采样器
        self.output_resampler = None  # soxr instance

        # 状态
        self._is_recording = False
        self._recorded_frames = []

    async def initialize_audio(self):
        """初始化音频设备、编解码器及重采样器"""
        try:
            # 1. 自动检测设备能力
            self._detect_device_capabilities()

            # 2. 创建 Opus 编解码器 (固定 16k 入, 24k 出)
            self.opus_encoder = opuslib.Encoder(
                config.input_sample_rate,
                config.channels,
                opuslib.APPLICATION_VOIP,
            )
            self.opus_decoder = opuslib.Decoder(
                config.output_sample_rate,
                config.channels
            )

            # 3. 创建输出重采样器 (解决沙哑问题)
            if self.device_output_sample_rate != config.output_sample_rate:
                print(f"🔧 初始化高质量重采样: {config.output_sample_rate}Hz -> {self.device_output_sample_rate}Hz")
                self.output_resampler = soxr.ResampleStream(
                    config.output_sample_rate,  # 输入: 24000
                    self.device_output_sample_rate,  # 输出: 如 48000
                    num_channels=1,
                    dtype="float32",
                    quality="QQ"
                )
            else:
                self.output_resampler = None

            # ==============服务端不处理输入流==============
            # 4. 创建音频流 (使用设备原生参数)
            # input_block_size = int(self.device_input_sample_rate * (config.frame_duration / 1000))
            output_block_size = int(self.device_output_sample_rate * (config.frame_duration / 1000))

            # print(f"🎙️ 输入流配置: Rate={self.device_input_sample_rate}Hz, Block={input_block_size}")
            # self.input_stream = sd.InputStream(
            #      device=self.input_device_id,
            #      samplerate=self.device_input_sample_rate,
            #      channels=self.input_channels,
            #      dtype=np.float32,
            #      blocksize=input_block_size,
            #      callback=self._input_callback,
            #      latency="low",
            # )
            self.input_stream = None  # 显式置空
            # ============================================

            print(f"🔊 输出流配置: Rate={self.device_output_sample_rate}Hz, Block={output_block_size}")
            self.output_stream = sd.OutputStream(
                device=self.output_device_id,
                samplerate=self.device_output_sample_rate,
                channels=self.output_channels,
                dtype=np.float32,
                blocksize=output_block_size,
                callback=self._output_callback,
                latency="low",
            )

            self.output_stream.start()
            print("✅ 音频系统初始化完成")

        except Exception as e:
            print(f"❌ 初始化音频设备失败: {e}")

    def _detect_device_capabilities(self):
        """检测输入输出设备的实际能力"""
        try:
            in_dev = sd.query_devices(kind='input')
            out_dev = sd.query_devices(kind='output')

            self.input_channels = min(in_dev['max_input_channels'], 2)
            self.output_channels = min(out_dev['max_output_channels'], 2)

            self.device_input_sample_rate = int(in_dev['default_samplerate'])
            self.device_output_sample_rate = int(out_dev['default_samplerate'])

            print(f"Hardware: Input({self.device_input_sample_rate}Hz), Output({self.device_output_sample_rate}Hz)")
        except Exception as e:
            print(f"Device detection failed, using defaults: {e}")

    #  打断播放队列
    def clear_audio_queue(self):
        """强制清空播放队列，实现立即静音/打断"""
        # 1. 清空一级 Queue
        try:
            while not self._output_buffer.empty():
                self._output_buffer.get_nowait()
        except queue.Empty:
            pass

        # 2. 清空二级 Deque (重采样缓冲)
        self._resample_output_buffer.clear()
        print("🛑 收到唤醒指令，播放流已强制中断 (Buffer Cleared)")

    # ------------------ 音频回调 (SoundDevice 线程) ------------------
    # =========== 服务端不处理输出逻辑 ===========
    def _input_callback(self, indata, frames, time_info, status):
        # """输入回调"""
        # if not self._is_recording:
        #      return

        # try:
        #      if self.input_channels > 1:
        #          audio_data = downmix_to_mono(indata, keepdims=False)
        #      else:
        #          audio_data = indata.flatten()

        #      # 转换为 int16
        #      audio_data_int16 = (audio_data * 32768.0).astype(np.int16)
        #      self._recorded_frames.append(audio_data_int16.copy())

        # except Exception as e:
        #      print(f"Input callback error: {e}")
        pass

    # ==========================================

    def _output_callback(self, outdata, frames, time_info, status):
        """输出回调"""
        if status and "underflow" not in str(status).lower():
            pass

        try:
            # 阶段 1: 填充缓冲区
            while len(self._resample_output_buffer) < frames:
                try:
                    audio_packet = self._output_buffer.get_nowait()
                    if self.output_resampler:
                        resampled_data = self.output_resampler.resample_chunk(audio_packet, last=False)
                        if len(resampled_data) > 0:
                            self._resample_output_buffer.extend(resampled_data)
                    else:
                        self._resample_output_buffer.extend(audio_packet)
                except queue.Empty:
                    break

            # 阶段 2: 消费缓冲区
            if len(self._resample_output_buffer) >= frames:
                frame_data = [self._resample_output_buffer.popleft() for _ in range(frames)]
                mono_samples = np.array(frame_data, dtype=np.float32)

                # 【新增】音量放大与削波
                # 1. 乘以倍数
                mono_samples = mono_samples * self.output_volume
                # 2. 防止爆音 (Clipping)
                mono_samples = np.clip(mono_samples, -1.0, 1.0)

                if self.output_channels > 1:
                    outdata[:] = upmix_mono_to_channels(mono_samples, self.output_channels)
                else:
                    outdata[:, 0] = mono_samples
            else:
                outdata.fill(0)

        except Exception as e:
            print(f"Output callback error: {e}")
            outdata.fill(0)

    # ------------------ 音频操作 (Asyncio 线程) ------------------

    async def write_audio(self, opus_data: bytes):
        """解码并放入播放队列"""
        try:
            pcm_data = self.opus_decoder.decode(opus_data, config.output_frame_size)
            audio_int16 = np.frombuffer(pcm_data, dtype=np.int16)
            audio_float = audio_int16.astype(np.float32) / 32768.0

            try:
                self._output_buffer.put_nowait(audio_float)
            except queue.Full:
                print("⚠️ 播放队列已满，丢弃音频帧")

        except opuslib.OpusError as e:
            print(f"Opus decode error: {e}")
        except Exception as e:
            print(f"Write audio error: {e}")

    # =============服务端不处理输入逻辑=============
    async def start_recording(self):
        # self._recorded_frames.clear()
        # if not self.input_stream.active:
        #      self.input_stream.start()
        # self._is_recording = True
        # print("🎙️ 开始录音...")
        pass

    async def stop_recording(self) -> Optional[str]:
        # if not self._is_recording:
        #      return None

        # self._is_recording = False

        # if not self._recorded_frames:
        #      return None

        # audio_data = np.concatenate(self._recorded_frames)
        # timestamp = int(time.time())
        # filename = f"./recording/recording_{timestamp}.wav"

        # # 写入WAV
        # with wave.open(filename, 'wb') as wav_file:
        #      wav_file.setnchannels(1)
        #      wav_file.setsampwidth(2)
        #      wav_file.setframerate(self.device_input_sample_rate)
        #      wav_file.writeframes(audio_data.tobytes())

        # print(f"✅ 录音已保存: {filename}")
        # return filename
        pass

    # ==========================================

    # ------------------ 网络通信逻辑 (增强版 - 修复 .open 属性错误) ------------------

    async def connect(self):
        """建立 WebSocket 连接"""
        if self.websocket:
            try:
                await self.websocket.close()
            except:
                pass

        print(f"Connecting to WS: {config.ws_url}")
        self.websocket = await websockets.connect(
            uri=config.ws_url,
            additional_headers=self.HEADERS,
            ping_interval=20,
            ping_timeout=20,
            close_timeout=10,
            max_size=10 * 1024 * 1024
        )
        print("Websocket connected!")

    async def sayhello(self):
        """发送 Hello 包并请求 Audio 特性"""
        print(f"Sending hello with OutputSR={config.output_sample_rate}")
        hello_message = {
            "type": "hello",
            "version": 1,
            "features": {"audio": True},
            "transport": "websocket",
            "audio_params": {
                "format": "opus",
                "sample_rate": config.output_sample_rate,
                "channels": config.channels,
                "frame_duration": config.frame_duration,
            },
        }
        # not self.websocket.closed => "1" == str(self.websocket.state)
        # 似乎是websocket的实现差异
        # open:1
        if (self.websocket is not None) and ("1" == str(self.websocket.state)):
            await self.websocket.send(json.dumps(hello_message))
        else:
            raise ConnectionError("Cannot say hello: Websocket not open")

    async def handle_server_hello(self, data: dict):
        self.session_id = data.get("session_id")
        if self.session_id:
            print(f"✅ Session ID acquired: {self.session_id}")
        else:
            print("⚠️ Server Hello received but no Session ID")

    async def message_handler(self):
        """持续处理消息"""
        try:
            async for message in self.websocket:
                try:
                    if isinstance(message, str):
                        data = json.loads(message)
                        if data.get("type") == "hello":
                            await self.handle_server_hello(data)
                        elif data.get("type") == "tts" and data.get("state") == "stop":
                            pass
                        else:
                            pass
                    elif isinstance(message, bytes):
                        await self.write_audio(message)
                except Exception as e:
                    print(f"Msg processing error: {e}")
                    continue
        except Exception as e:
            print(f"⚠️ WebSocket loop disconnected: {e}")
        finally:
            # 任何原因导致循环退出，都标记连接无效
            self.session_id = None
            if (self.websocket is not None) and ("1" == str(self.websocket.state)):
                await self.websocket.close()
            # 触发自动重连（如果需要保持常驻），在长任务时可以用这个
            # asyncio.create_task(self.reconnect())

    async def reconnect(self):
        """带锁的重连逻辑"""
        if self._reconnecting_lock.locked():
            return

        async with self._reconnecting_lock:
            print("🔄 Reconnecting logic started...")

            # 1. 停止旧的 handler 任务
            if self.handing_message_task and not self.handing_message_task.done():
                self.handing_message_task.cancel()
                try:
                    await self.handing_message_task
                except asyncio.CancelledError:
                    pass

            self.session_id = None

            # 2. 尝试连接循环
            backoff = 1
            while True:
                try:
                    await self.connect()
                    await self.sayhello()
                    self.handing_message_task = asyncio.create_task(self.message_handler())
                    # 3. 连接成功后，等待 session id 一小段时间
                    for _ in range(20):  # 等待 2秒
                        if self.session_id:
                            print("✅ Reconnected and Session ID ready.")
                            return
                        await asyncio.sleep(0.1)

                    # 如果连上了但没收到 Session ID，继续重试
                    print("Connected but no Session ID, retrying...")
                except Exception as e:
                    print(f"Reconnect failed: {e}. Retrying in {backoff}s...")
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, 10)  # 指数退避

    async def check_and_restore_connection(self) -> bool:
        """
        主动检查连接状态，如果不可用则尝试恢复。
        返回 True 表示当前连接可用（有 Socket 且有 SessionID）。
        """
        # 1. 检查物理连接 - 修改：使用 not self.websocket.closed
        is_socket_alive = self.websocket is not None and ("1" == str(self.websocket.state))
        # 2. 检查逻辑连接
        is_session_valid = self.session_id is not None

        if is_socket_alive and is_session_valid:
            return True

        print(f"⚠️ Connection check failed (Socket={is_socket_alive}, Session={is_session_valid}). Restoring...")

        # 强制触发重连
        await self.reconnect()

        # 再次检查
        if (self.websocket is not None) and ("1" == str(self.websocket.state)) and self.session_id:
            return True
        return False

    async def send_recording_to_api(self, filename):
        """上传录音文件，上传前强制检查连接"""
        if not os.path.exists(filename):
            print(f"File not found: {filename}")
            return False

        try:
            # Step 1: 确保连接和 Session ID 可用
            connection_ok = await self.check_and_restore_connection()
            if not connection_ok:
                print("❌ Failed to restore connection. Cannot upload.")
                return False

            print(f"📤 Uploading {filename} with Session ID: {self.session_id}")

            # Step 2: 准备数据
            data = {'sessionId': self.session_id}

            # 使用 requests (同步库)
            with open(filename, 'rb') as wav_file:
                files = {'file': (filename, wav_file, 'audio/wav')}
                response = requests.post(config.api_url, files=files, data=data, timeout=10)

            if response.status_code == 200:
                print(f"✅ Upload success: {response.text}")
                return True
            else:
                print(f"❌ Upload failed: {response.status_code} - {response.text}")
                # 如果是 401/403 等，可能 session 过期，置空 session 触发下次重连
                if response.status_code in [400, 401, 403, 500]:
                    print("Invalidating session due to upload error.")
                    self.session_id = None
                return False

        except Exception as e:
            print(f"❌ Upload exception: {e}")
            return False


        finally:
            # TODO: 重要！删除上传后的文件！！
            pass


# ================= HTTP Server =================

class MyHttpHandler(BaseHTTPRequestHandler):
    client: AudioClient = None

    def do_GET(self):
        parsed_path = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed_path.query)

        if parsed_path.path == '/do_send':
            if 'fname' not in params:
                self.send_error(400, "Missing fname")
                return
            fname = params['fname'][0]
            # 放到 Loop 中执行，防止阻塞 HTTP 线程
            asyncio.run_coroutine_threadsafe(
                self.client.send_recording_to_api(fname),
                self.client.loop
            )
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Upload task scheduled")

        elif parsed_path.path == '/record_start':
            asyncio.run_coroutine_threadsafe(self.client.start_recording(), self.client.loop)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Recording started")

        elif parsed_path.path == '/record_stop':
            future = asyncio.run_coroutine_threadsafe(self.client.stop_recording(), self.client.loop)
            try:
                filename = future.result(timeout=5)
                if filename:
                    # 录音停止后自动上传
                    asyncio.run_coroutine_threadsafe(
                        self.client.send_recording_to_api(filename),
                        self.client.loop
                    )
                    msg = f"Stopped and uploading {filename}"
                else:
                    msg = "Stopped (no data)"
            except Exception as e:
                msg = f"Error stopping: {e}"
            self.send_response(200)
            self.end_headers()
            self.wfile.write(msg.encode())


        # 在 MyHttpHandler 的 do_GET 方法中添加判断分支
        elif parsed_path.path == '/interrupt_play':
            # 调用上面新增的清空方法
            if self.client:
                self.client.clear_audio_queue()
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"Playback interrupted")

        # 【新增接口】实时调整音量
        elif parsed_path.path == '/set_volume':
            if 'v' in params:
                try:
                    vol = float(params['v'][0])
                    self.client.output_volume = vol
                    print(f"🔊 Volume set to {vol}x")
                    msg = f"Volume set to {vol}"
                except ValueError:
                    msg = "Invalid volume value"
            else:
                msg = "Missing 'v' param (usage: /set_volume?v=3.0)"

            self.send_response(200)
            self.end_headers()
            self.wfile.write(msg.encode())

        else:
            self.send_error(404)


# ================= Main =================

async def websocket_server(websocket, path):
    try:
        async for message in websocket:
            await websocket.send("Echo")
    except:
        pass


async def main():
    # 1. 创建客户端
    client = AudioClient()
    client.loop = asyncio.get_running_loop()

    # 2. 初始化音频
    await client.initialize_audio()

    # 3. 注入 HTTP Handler
    MyHttpHandler.client = client

    # 4. 启动本地 WS Server (可选)
    print(f"Starting Local WS Server on {WS_PORT}")
    ws_server = await websockets.serve(websocket_server, "0.0.0.0", WS_PORT)

    # 5. 启动 HTTP Server (线程)
    def run_http_server():
        http_server = HTTPServer(('0.0.0.0', HTTP_PORT), MyHttpHandler)
        print(f"HTTP server started on port {HTTP_PORT}")
        http_server.serve_forever()

    threading.Thread(target=run_http_server, daemon=True).start()

    # 6. 初始连接（注意sayhello会开启一个新的对话链，并在超时后自动发送退出音频）
    # await client.connect()
    # await client.sayhello()
    client.handing_message_task = asyncio.create_task(client.message_handler())

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
        if client.input_stream: client.input_stream.stop()
        if client.output_stream: client.output_stream.stop()


if __name__ == "__main__":
    asyncio.run(main())