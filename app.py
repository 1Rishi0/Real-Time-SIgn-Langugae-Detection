from flask import Flask, render_template, Response, request, jsonify, redirect, url_for, session
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from googletrans import Translator
from gtts import gTTS
import os
from flask_pymongo import PyMongo
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'secret_key'  # Change this to a strong secret key

# MongoDB Configuration
app.config["MONGO_URI"] = "mongodb+srv://msrishitha:Rishitha*@cluster0.8dtknaa.mongodb.net/myDatabase"
mongo = PyMongo(app)

# Load the trained model    
model = tf.keras.models.load_model('enhanced_model.h5')

# Labels for prediction output
labels = [str(i) for i in range(10)] + [chr(i) for i in range(97, 123)]

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
translator = Translator()

sentence = ""
added_signs = set()
last_sign = None
hand_present = False
detection_running = False  # Control detection state

def generate_frames():
    global sentence, last_sign, added_signs, hand_present, detection_running
    cap = cv2.VideoCapture(0)

    detection_threshold = 4  # Number of frames to confirm a letter
    detection_count = 0
    confirmed_sign = None

    while detection_running:
        success, frame = cap.read()
        if not success:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)
        hand_present = bool(results.multi_hand_landmarks)

        if hand_present:
            all_hand_landmarks = []
            for hand_landmarks in results.multi_hand_landmarks:
                hand_data = [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
                all_hand_landmarks.append(np.array(hand_data).flatten())

            if len(all_hand_landmarks) == 1:
                all_hand_landmarks.append([0] * 63)
            elif len(all_hand_landmarks) > 2:
                all_hand_landmarks = all_hand_landmarks[:2]

            combined_landmarks = np.concatenate(all_hand_landmarks).reshape(1, -1)
            prediction = model.predict(combined_landmarks)
            sign_label = labels[np.argmax(prediction)]

            # Stabilize detection using a frame threshold
            if sign_label == last_sign:
                detection_count += 1
            else:
                detection_count = 0

            if detection_count >= detection_threshold:
                confirmed_sign = sign_label
                detection_count = 0
                if confirmed_sign not in added_signs:
                    sentence += confirmed_sign
                    added_signs.add(confirmed_sign)

            last_sign = sign_label

            for hand_landmarks in results.multi_hand_landmarks:
                mp.solutions.drawing_utils.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Display detected sign and sentence
        cv2.putText(frame, f"Detected: {confirmed_sign if confirmed_sign else '...'}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text
        cv2.putText(frame, f"Sentence: {sentence.strip()}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)  # Black text

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if mongo.db.users.find_one({"username": username}):
            return jsonify({"message": "User already exists"}), 400

        hashed_password = generate_password_hash(password)
        mongo.db.users.insert_one({"username": username, "password": hashed_password})
        return redirect(url_for('signin'))

    return render_template('signup.html')

@app.route('/signin', methods=['GET', 'POST'])
def signin():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        user = mongo.db.users.find_one({"username": username})
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            return redirect(url_for('index'))
        return jsonify({"message": "Invalid credentials"}), 401

    return render_template('signin.html')

@app.route('/index')
def index():
    if 'user' not in session:
        return redirect(url_for('signin'))
    return render_template('index.html', sentence=sentence)

@app.route('/video')
def video():
    global detection_running
    detection_running = True
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/stop', methods=['POST'])
def stop():
    global detection_running
    detection_running = False
    return jsonify({"message": "Detection stopped"})

@app.route('/reset', methods=['POST'])
def reset():
    global sentence, added_signs, last_sign
    sentence = ""
    added_signs.clear()
    last_sign = None
    return jsonify({"message": "Reset successful"})

@app.route('/translate', methods=['POST'])
def translate():
    global sentence
    translated_sentence = translator.translate(sentence, src='en', dest='hi').text
    tts_en = gTTS(text=sentence, lang='en') 
    tts_hi = gTTS(text=translated_sentence, lang='hi')
    tts_en.save("static/english.mp3")
    tts_hi.save("static/hindi.mp3")
    return jsonify({"translated": translated_sentence, "english_audio": "static/english.mp3", "hindi_audio": "static/hindi.mp3"})

@app.route('/space', methods=['POST'])
def add_space():
    global sentence
    sentence += " "
    return jsonify({"message": "Space added", "sentence": sentence})

@app.route('/backspace', methods=['POST'])
def remove_last_character():
    global sentence
    sentence = sentence[:-1]
    return jsonify({"message": "Last character removed", "sentence": sentence})

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
