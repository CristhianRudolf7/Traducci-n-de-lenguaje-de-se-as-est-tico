from flask import Flask, render_template, Response, jsonify
import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import joblib
import math
import time

app = Flask(__name__)

# Inicializa MediaPipe y el modelo
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
model_extra = joblib.load('extraTrees.pkl')

# Variables globales para el manejo de la predicción
current_prediction = {"prediccion": "N/A", "precision": 0.0}
accumulated_text = ""  # Variable para almacenar la frase acumulada
last_letter = ""  # Variable para rastrear la última letra agregada

def generate_frames():
    global current_prediction, accumulated_text, last_letter
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        angulos = []
        coordenadas = []
        success, frame = cap.read()
        if not success:
            continue

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = results.multi_hand_landmarks[0].landmark

            coordenadas = []
            angulos = []
            try:
                for i in range(21):
                    punto = landmarks[i]
                    x, y, z = punto.x, punto.y, punto.z
                    coordenadas.append(np.array([x, y, z]))

                for i in range(21):
                    if i == 0 or i in [4, 8, 12, 16, 20]:
                        a = b = c = 0
                    elif i in [5, 9, 13, 17]:
                        a = np.linalg.norm(coordenadas[0] - coordenadas[i + 1])
                        b = np.linalg.norm(coordenadas[0] - coordenadas[i])
                        c = np.linalg.norm(coordenadas[i + 1] - coordenadas[i])
                    else:
                        a = np.linalg.norm(coordenadas[i - 1] - coordenadas[i + 1])
                        b = np.linalg.norm(coordenadas[i - 1] - coordenadas[i])
                        c = np.linalg.norm(coordenadas[i + 1] - coordenadas[i])

                    angulo = 0
                    if a != 0 and b != 0 and c != 0:
                        cos = (b**2 + c**2 - a**2) / (2 * b * c)
                        if -1 < cos < 1:
                            angulo = round(degrees(abs(acos(cos))))
                        angulos.append(angulo)

                x = landmarks[9].x - landmarks[0].x
                y = (1 - landmarks[9].y) - (1 - landmarks[0].y)
                angulo_radianes = math.atan2(y, x)
                angulo_grados = math.degrees(angulo_radianes)

                if angulo_grados < 0:
                    angulo_grados += 360
                angulos.append(angulo_grados)

                # Realiza la predicción si hay suficientes ángulos calculados
                if len(angulos) == 16:
                    angulos = np.array(angulos).reshape(1, -1)
                    pred = model_extra.predict(angulos)
                    letra = pred[0]

                    prob = model_extra.predict_proba(angulos)
                    precision = np.max(prob) * 100

                    # Evita agregar la misma letra consecutivamente
                    if letra != last_letter:
                        accumulated_text += letra
                        last_letter = letra
                    if precision > 75:
                        time.sleep(1.5)
                        # Actualiza la predicción actual con la frase acumulada
                        current_prediction = {"prediccion": accumulated_text, "precision": precision}

            except Exception as e:
                print("No se encontraron manos")
                print(f"Error: {e}")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/bi')
def bi():
    return render_template('bi.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_prediction')
def get_prediction():
    return jsonify(current_prediction)

@app.route('/clear_text')
def clear_text():
    global accumulated_text, last_letter
    accumulated_text = ""
    last_letter = ""
    current_prediction["prediccion"] = ""
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True)