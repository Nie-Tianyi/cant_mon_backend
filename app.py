from flask import Flask, Response
import cv2

app = Flask(__name__)
video_capture = cv2.VideoCapture(0)  # 从默认摄像头获取视频


def process_video(video):
    # 将视频转换为灰度
    gray = cv2.cvtColor(video, cv2.COLOR_BGR2GRAY)
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
