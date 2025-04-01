from flask import Flask, jsonify
from RPSClassifcation import RPSClassification
import random

app = Flask(__name__)
classifier = RPSClassification()

GESTURES = ['Rock', 'Paper', 'Scissors']

@app.route('/detect', methods=['GET'])
def detect_gesture():
    try:
        print("Received request for gesture detection")  # Add logging
        prediction, _, _ = classifier.predictFromCamera_Timeout(1)
        player_gesture = GESTURES[prediction.argmax()]
        counter_gesture = random.choice(GESTURES)

        print(f"Player Gesture: {player_gesture}, Counter Gesture: {counter_gesture}")  # Add logging
        return jsonify({
            'player_gesture': player_gesture,
            'counter_gesture': counter_gesture
        })
    except Exception as e:
        print(f"Error: {e}")  # Add logging
        return jsonify({'error': str(e)})

@app.route('/shutdown', methods=['POST'])
def shutdown():
    classifier.releaseCamAndDestroy()
    return jsonify({'message': 'Camera released and backend shutting down.'})

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000)  # Bind to localhost
