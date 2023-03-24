import imageio
from flask import Flask, request, render_template
from gevent import pywsgi
import cv2

app = Flask(__name__)


@app.route('/', methods=['POST'])
def index():
    file = request.files['file']
    stream = file.stream.read()

    print(stream)
    print(type(stream))

    # vid_reader = imageio.get_reader(stream, 'ffmpeg')
    # print(vid_reader)
    # for img in vid_reader.iter_data():
    #     img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    #     cv2.imshow("ss", img_bgr)
    #     cv2.waitKey(1)
    #     print(img_bgr)
    return "OK"


server = pywsgi.WSGIServer(('localhost', 8082), app)
server.serve_forever()
