from flask import Flask, Response, request
from flask_cors import CORS
import cv2
import numpy as np
import random
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# configuration
video_capture = cv2.VideoCapture(0)  # get video stream from default camera
waiting_time = 0  # minutes
waiting_people = 0  # number of people waiting

app = Flask(__name__)  # flask app
CORS(app)

# Load YOLOv3 model
net = cv2.dnn.readNet("./darknet/yolov3.weights", "./darknet/cfg/yolov3.cfg")
# Get layer names
layer_names = net.getLayerNames()
# Get unconnected output layers
output_layers_indices = net.getUnconnectedOutLayers()
print(output_layers_indices)
# Convert indices to layer names
output_layers = [layer_names[i - 1] for i in output_layers_indices]
# Print the processed output layers
print("Processed output layers:", output_layers)

faceNet = cv2.dnn.readNet("deploy.prototxt", "res10_300x300_ssd_iter_140000.caffemodel")
maskNet = load_model("mask_detector.model")


def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            if face.any():
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)


def count_students(frame):
    student_count = 0
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Initialize lists to store bounding boxes and confidence scores
    boxes = []
    confidences = []

    # Process outs to count number of students and detect bounding boxes and confidence scores
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming class_id 0 corresponds to person in COCO dataset
                student_count += 1
                # Get coordinates of bounding box
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                # Calculate top-left corner
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                # Add bounding box coordinates and confidence score to lists
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # Apply non-maximum suppression to remove redundant bounding boxes
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    filtered_boxes = [boxes[i] for i in indices]

    return len(filtered_boxes), filtered_boxes


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
        "waiting_time": waiting_time,  # times measured in minutes
        "waiting_people": waiting_people  # number of people waiting
    }
    return res


def process_video(frame):
    # observer design pattern, update waiting time and waiting people when the frame is processed
    global waiting_time, waiting_people

    # Resize the frame to a smaller size
    resized_frame = cv2.resize(frame, (320, 240))

    # Detect students and bounding boxes
    student_count, boxes = count_students(resized_frame)

    # Draw bounding boxes on frame
    for box in boxes:
        x, y, w, h = box
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    (locs, preds) = detect_and_predict_mask(resized_frame, faceNet, maskNet)

    staff_num = 0
    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        if label == "Mask":
            staff_num += 1
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(resized_frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(resized_frame, (startX, startY), (endX, endY), color, 2)

    labeled_frame = resized_frame  # labeled_frame is the frame with bounding boxes

    waiting_people = student_count - staff_num # 更新等待人数

    waiting_time = waiting_people * 30  # 更新等待时间

    return labeled_frame  # 返回处理后的帧


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


@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
    return response


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
