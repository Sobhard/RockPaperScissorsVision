import cv2
import numpy as np
import time
from flask import Flask, jsonify


model_cfg = "custom-yolov4-tiny-detector.cfg"  
model_weights = "custom-yolov4-tiny-detector_last (5).weights"  


net = cv2.dnn.readNet(model_weights, model_cfg)


if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("GPU available! Using CUDA backend.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA) 
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
else:
    print("GPU not available. Using CPU backend.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) 
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

gesture_map = {
    0: "Paper",
    1: "Rock",
    2: "Scissors"
}

app = Flask(__name__)

@app.route('/detect', methods=['GET'])
def detect_gesture():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        return jsonify({"error": "Failed to capture frame"}), 500

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outputs = net.forward(output_layers)

    class_ids = []
    for out in outputs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                class_ids.append(class_id)

    if len(class_ids) == 1:
        player_gesture = gesture_map[class_ids[0]]
        counter_gesture = "Scissors"
        if player_gesture == "Rock":
            counter_gesture = "Paper"
        elif player_gesture == "Paper":
            counter_gesture = "Scissors"
        elif player_gesture == "Scissors":
            counter_gesture = "Rock"
        return jsonify({"player_gesture": player_gesture, "counter_gesture": counter_gesture})

    return jsonify({"error": "No gesture detected"}), 400

if __name__ == '__main__':
    app.run(debug=True)