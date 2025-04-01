from flask import Flask, jsonify
from flask_cors import CORS  # Import CORS
from RPSClassifcation import RPSClassification
import random

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

classifier = RPSClassification()

GESTURES = ['Rock', 'Paper', 'Scissors']

@app.route('/detect', methods=['GET'])
def detect_gesture():
    try:
        print("Received request for gesture detection.")
        prediction, _, _ = classifier.predictFromCamera_Timeout(1)
        player_gesture = GESTURES[prediction.argmax()]
        if (player_gesture == 'Rock'):
            counter_gesture = 'Paper'
        elif (player_gesture == 'Paper'):
            counter_gesture = 'Scissors'
        elif (player_gesture == 'Scissors'):
            counter_gesture = 'Rock'
        counter_gesture = random.choice(GESTURES)

        print(f"Player Gesture: {player_gesture}, Counter Gesture: {counter_gesture}")
        return jsonify({
            'player_gesture': player_gesture,
            'counter_gesture': counter_gesture
        })
    except Exception as e:
        print(f"Error during gesture detection: {e}")
        return jsonify({'error': str(e)})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    print("Shutting down backend and releasing resources.")
    classifier.releaseCamAndDestroy()
    return jsonify({'message': 'Camera released and backend shutting down.'})

if __name__ == '__main__':
    print("Starting backend server on http://127.0.0.1:5000")
    app.run(host='127.0.0.1', port=5000)
