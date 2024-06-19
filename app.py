# app.py
from flask import Flask, render_template, Response
import cv2
import os
print("Current working directory:", os.getcwd())
from sound_with_Text import HandGestureRecognition

app = Flask(__name__)
sound_with_Text = HandGestureRecognition()

cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gesture = sound_with_Text.process_frame(frame)
        if gesture:
            cv2.putText(frame, f'Sign: {gesture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, cv2.LINE_AA)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)