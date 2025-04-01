import tensorflow as tf
import cv2
import time as t
import numpy as np
import matplotlib.pyplot as plt

class RPSClassification:

    def __init__(self):
        self.model = tf.lite.Interpreter(model_path="model.tflite")
        self.model.allocate_tensors()

        self.cam =  cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 50)  # Set width
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 50)  # Set height

    def getFrame(self):

        foundFrame = False

        while not foundFrame:
            foundFrame, frame = self.cam.read()

        return frame
    
    #shapes the img for the model/coverts uint8 to float32
    def processImg(self, img):
        img = cv2.resize(img, (50, 50))
        img = np.float32(img / 255.0) #love numpy for this
        img = np.expand_dims(img, axis=0)
        return img

    #returns the output data for a single img
    def testInference(self, path):
        img = cv2.imread(path)
        img = self.processImg(img)

        #set the input image of the model
        input_details = self.model.get_input_details()
        self.model.set_tensor(input_details[0]['index'], img)
        
        self.model.invoke()
        
        return self.model.get_tensor(self.model.get_output_details()[0]['index'])
    
    #returns best prediction, the frame used for that prediction, and the FPS
    def predictFromCamera_Timeout(self, time):

        #time in seconds
        start_time = t.time()
        bestPrediction = np.zeros(shape=(1, 3), dtype=np.float32)
        bestframe = self.getFrame()

        counter = 0

        while(t.time() < start_time + time):

            frame = self.getFrame()
            temp = frame
            frame = self.processImg(frame)

            counter += 1

            self.model.set_tensor(self.model.get_input_details()[0]['index'], frame)
            self.model.invoke()
            output = self.model.get_tensor(self.model.get_output_details()[0]['index'])

            if np.max(output) > np.max(bestPrediction):
                bestPrediction = output
                bestframe = temp
        
        FPS = counter/time
        return (bestPrediction, bestframe, FPS)

    def releaseCamAndDestroy(self):
        self.cam.release()
        cv2.destroyAllWindows()


#TESTING
# classifier = RPSClassification()


# # You can also perform inference on the frame and show the result if needed
# pred, frame, FPS = classifier.predictFromCamera_Timeout(1)
# print(pred)
# display_image(frame)
# print(FPS)

# while(True):
#     cv2.imshow("webcam", classifier.getFrame())
    
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# classifier.releaseCamAndDestroy()
    

