import cv2
import numpy as np

model_cfg = "custom-yolov4-tiny-detector.cfg"  # Path to your YOLOv4-tiny cfg file
model_weights = "custom-yolov4-tiny-detector_last (5).weights"  # Path to your YOLOv4-tiny weights file

# Load the network using OpenCV's DNN module
net = cv2.dnn.readNet(model_weights, model_cfg)

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# Open webcam using a specific backend
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # For Windows with DirectShow

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

            if confidence > 0.4:  # Confidence thresh4old for detections
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

    # Draw bounding boxes and labels
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(class_ids[i])
            cv2.rectangle(frame, (x - (w//2), y - (h//2)), (x + (w//2), y + (h//2)), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("Frame", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()