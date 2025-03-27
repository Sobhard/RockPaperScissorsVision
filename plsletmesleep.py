import cv2
import numpy as np
import time
import threading

# Specify paths to YOLOv4-tiny model
model_cfg = "custom-yolov4-tiny-detector.cfg"  # Path to your YOLOv4-tiny cfg file
model_weights = "custom-yolov4-tiny-detector_last (5).weights"  # Path to your YOLOv4-tiny weights file

# Load the network using OpenCV's DNN module
net = cv2.dnn.readNet(model_weights, model_cfg)

# Check if GPU is available and set backend and target to CUDA
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print("GPU available! Using CUDA backend.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)  # Set backend to CUDA
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)  # Set target to CUDA
else:
    print("GPU not available. Using CPU backend.")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)  # Default CPU backend
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)  # Default CPU target

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

gesture_map = {
    0: "Paper",
    1: "Rock",
    2: "Scissors"
}

# Open webcam using a specific backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows with DirectShow

# Counter move
counter_gesture = "Nothing"

# Thread Event for synchronization
shoot_event = threading.Event()

# Function to simulate terminal output (Rock, Paper, Scissors)
def show_gesture_output():
    gestures = ["ROCK", "PAPER", "SCISSORS"]
    for gesture in gestures:
        print(gesture)
        time.sleep(0.5)
    
    print("SHOOT")
    time.sleep(0.5)  # Let the system have some time to react before shooting

    shoot_event.set()  # Trigger the event to proceed with counter gesture

# Function to handle gesture detection
def detect_gesture():
    global counter_gesture
    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

                if confidence > 0.5:
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)

                    boxes.append([center_x, center_y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(gesture_map[class_ids[i]])
                cv2.rectangle(frame, (x - (w//2), y - (h//2)), (x + (w//2), y + (h//2)), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)

        if len(class_ids) == 1:
            opponent_gesture = gesture_map[class_ids[0]]
            counter_gesture = "Scissors"

            if opponent_gesture == "Rock":
                counter_gesture = "Paper"
            elif opponent_gesture == "Paper":
                counter_gesture = "Scissors"
            elif opponent_gesture == "Scissors":
                counter_gesture = "Rock"

        if shoot_event.is_set(): 
            print(f"Counter Move: {counter_gesture}")

        if shoot_event.is_set():
            break

# Start terminal output in a separate thread
output_thread = threading.Thread(target=show_gesture_output)
output_thread.start()

# Start gesture detection in the main thread
detect_gesture()

# Release resources
cap.release()
cv2.destroyAllWindows()