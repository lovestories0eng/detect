#向服务端发送数据的线程
import cv2
import base64
import websocket
import threading

class SendThread (threading.Thread):   
    def __init__(self, video_url, ws_server_url, camera_id):
        threading.Thread.__init__(self)
        #"rtsp://admin:a1234567@192.168.8.11:554/h264/ch1/main/av_stream"
        self._video_url = video_url
        self._ws = None
        self._ws_server_url = ws_server_url
        self._capture = None
        self._send = True
        #视频播放质量4流畅3标清2高清1超清
        self._quality = 4
        self._camera_id = camera_id

    def run(self):
        websocket.enableTrace(False)
        self._ws = websocket.WebSocketApp(self._ws_server_url,
                            on_message=self.on_message,
                            on_error=self.on_error,
                            on_close=self.on_close,
                            on_open=self.on_open)
        self._ws.run_forever()

    def send_frame(self):

        self._capture = cv2.VideoCapture(self._video_url)
        frame_width = int(self._capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(self._capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        while self._send:
            ret, frame = self._capture.read()
            if ret:
                # 576,324
                r = int(2 + self._quality * 0.5)
                frame = cv2.resize(frame,(frame_width // r,frame_height // r))
                #图片质量1-100
                jpeg_quality = 100 - 10 * self._quality
                img_param = [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_quality]
                # 转化
                ret, frame = cv2.imencode('.jpg', frame, img_param)
                if ret:
                    self._ws.send(base64.b64encode(frame).decode("utf-8"))

    def on_message(self, message):
        print(message)

    def on_error(self, error):
        print(error)
        global send_thread_pool
        del send_thread_pool[self._camera_id]
        self._send = False
        self._capture.release()

    def on_close(self):
        print("closed video send!")
        global send_thread_pool
        del send_thread_pool[self._camera_id]
        self._send = False
        self._capture.release()

    def on_open(self):
        print('连接媒体服务器成功!')
        global send_thread_pool
        send_thread_pool[self._camera_id] = self
        self.send_frame()

    def set_quality(self, quality):
        self._quality = quality
        
test = SendThread(0, "ws://localhost:8168", 0)
test.run()

