import cv2
import numpy as np
from flask import Flask, jsonify, redirect
import threading

app = Flask(__name__)

# Global variables to store detection results
player_gesture = None
counter_gesture = None

model_cfg = "custom-yolov4-tiny-detector.cfg"  # Path to your YOLOv4-tiny cfg file
model_weights = "custom-yolov4-tiny-detector_last (5).weights"  # Path to your YOLOv4-tiny weights file

# Load the network using OpenCV's DNN module
net = cv2.dnn.readNet(model_weights, model_cfg)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Try initializing the webcam with different backends
cap = None
for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2]:
    cap = cv2.VideoCapture(0, backend)
    if cap.isOpened():
        print(f"Webcam initialized with backend: {backend}")
        break
else:
    raise RuntimeError("Failed to initialize webcam. Check your camera or drivers.")

gesture_map = {0: "Rock", 1: "Paper", 2: "Scissors"}

# Function to detect gestures
def detect_gesture():
    global player_gesture, counter_gesture
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from webcam.")
            continue

        # Process the frame for YOLO
        blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outputs = net.forward(output_layers)

        # Post-process the results (detection)
        boxes, confidences, class_ids = [], [], []
        height, width = frame.shape[:2]

        for out in outputs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.4:  # Confidence threshold for detections
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    # Append bounding box details
                    boxes.append([center_x, center_y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        # Non-Maximum Suppression to eliminate redundant boxes
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        # Update gestures
        if len(class_ids) == 1:
            player_gesture = gesture_map[class_ids[0]]
            counter_gesture = "Scissors"
            if player_gesture == "Rock":
                counter_gesture = "Paper"
            elif player_gesture == "Paper":
                counter_gesture = "Scissors"
            elif player_gesture == "Scissors":
                counter_gesture = "Rock"

        # Exit on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Flask route to provide detection results
@app.route('/detect', methods=['GET'])
def detect():
    global player_gesture, counter_gesture
    if player_gesture is None or counter_gesture is None:
        return jsonify({"error": "No gesture detected yet."})
    return jsonify({"player_gesture": player_gesture, "counter_gesture": counter_gesture})

# Add a root route to avoid 404 errors
@app.route('/')
def index():
    return "Rock Paper Scissors Backend is running."

if __name__ == "__main__":
    # Start gesture detection in a separate thread
    detection_thread = threading.Thread(target=detect_gesture)
    detection_thread.start()

    # Run the Flask app
    app.run(host="0.0.0.0", port=5000)
