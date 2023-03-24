#!/usr/bin/python
# -*-coding:utf-8 -*-
import socket
import cv2
import numpy
import base64
from time import sleep

# socket.AF_INET 用于服务器与服务器之间的网络通信
# socket.SOCK_STREAM 代表基于TCP的流式socket通信
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# 连接服务端
address_server = ('localhost', 8010)
sock.connect(address_server)

# 从摄像头采集图像
# 参数是0,表示打开笔记本的内置摄像头,参数是视频文件路径则打开视频
capture = cv2.VideoCapture(0)
# capture.read() 按帧读取视频
# ret,frame 是capture.read()方法的返回值
# 其中ret是布尔值，如果读取帧正确，返回True;如果文件读到末尾，返回False。
# frame 就是每一帧图像，是个三维矩阵
ret, frame = capture.read()
encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]
while ret:
    # 首先对图片进行编码，因为socket不支持直接发送图片
    # '.jpg'表示把当前图片frame按照jpg格式编码
    # result, img_encode = cv2.imencode('.jpg', frame)
    img_encode = cv2.imencode('.jpg', frame, encode_param)[1]
    data = numpy.array(img_encode)
    img = data.tobytes()
    img = ("data:image/jpeg;base64," + base64.b64encode(img).decode()).encode("utf-8")
    # 首先发送图片编码后的长度
    sock.send(img)
    # sleep(1)
    ret, frame = capture.read()
    cv2.resize(frame, (640, 480))

sock.close()
cv2.destroyAllWindows()
