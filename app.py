import random

from flask import Flask, Response,request
import cv2

# configuration
app = Flask(__name__)  # flask app
video_capture = cv2.VideoCapture(0)  # get video stream from default camera
waiting_time = 0  # seconds
waiting_people = 0  # number of people waiting


@app.route('/')
def home():
    url = request.url_root

    return f"""
    This is the backend server of the canteen monitering system.<br>
    To access the video stream, go to {url}video_feed.<br>
    To get the estimated waiting time and number of people waiting, go to {url}estimated_waiting_info.
    """


@app.route("/estimated_waiting_info")
def waiting_info():
    global waiting_time, waiting_people
    res = {
        "waiting_time": waiting_time,  # times measured in seconds
        "waiting_people": waiting_people  # number of people waiting
    }
    return res


def process_video(frame):
    # 将视频转换为灰度, 替换成标记模型处理每一帧
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 设置waiting time 和waiting people， 替换成排队人数和预测时间
    global waiting_time, waiting_people
    waiting_time = random.randint(0, 100)
    waiting_people = random.randint(0, 100)
    return gray


@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            ret, frame = video_capture.read()
            if not ret:
                break
            frame = process_video(frame)
            ret, jpeg = cv2.imencode('.jpg', frame)
            if not ret:
                break
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
