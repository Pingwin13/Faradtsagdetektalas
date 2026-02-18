import cv2
import mediapipe as mp
import math
import numpy as np
import time
import winsound
from collections import deque
import threading
import json
from datetime import datetime

# Naplozasi beallitasok
log_filename = "faradtsagnaplo.json"
last_logged_status = None
log_data = []

def save_to_json(data):
    with open(log_filename, 'w', encoding= 'utf-8') as f:
        json.dump(data, f)

# Beallitasok es Fuggvenyek

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

def play_alarm():
    winsound.Beep(2000,600)

def distance_3d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)


def eye_aspect_ratio_3d(landmarks, eye_indices):
    p = [landmarks[i] for i in eye_indices]
    return (distance_3d(p[1], p[5]) + distance_3d(p[2], p[4])) / (2.0 * distance_3d(p[0], p[3]))


def mouth_aspect_ratio(landmarks):
    vertical = distance_3d(landmarks[13], landmarks[14])
    horizontal = distance_3d(landmarks[61], landmarks[291])
    if horizontal == 0: return 0
    return vertical / horizontal


def get_euler_angles(rvec, tvec):
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)


# Seged fuggveny arc kozephez
def get_face_center(landmarks):
    xs = [pt[0] for pt in landmarks]
    ys = [pt[1] for pt in landmarks]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))


# Landmark Indexek

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
PNP_IMAGE_POINTS_IDX = [1, 152, 33, 263, 61, 291]
PNP_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
], dtype="double")
FACE_OUTLINE = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
                103, 67, 109]
MOUTH_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
MOUTH_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Parameterek

cap = cv2.VideoCapture(0)
ear_history = deque(maxlen=5)

#Kalibracios valtozok
is_calibrated = False
calibration_frames = 0
MAX_CALIBRATION_FRAMES = 100
calibration_ear_values = []
EAR_THRESHOLD = 0.31  # Amennyiben nem sikerul kalibralni ez az alapertek.

#Szem idozites
eye_closed_frame_counter = 0
EYE_CLOSED_FRAMES_THRESHOLD = 20
MICROSLEEP_FRAMES = 45

# Pislogas szamlalo valtozok
blink_count = 0
blink_ready = True
start_time = time.time()
blinks_per_minute = 0

# DIST & posture thresholds
DIST_THRESHOLD = 200
FACE_LOST_THRESHOLD = 220
MAR_THRESHOLD = 0.6
YAWN_FRAMES_THRESHOLD = 20
yawn_frame_counter = 0

locked_face_center = None

# Fociklus

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape

    #Kalibracios uzenet
    if not is_calibrated:
        cv2.putText(frame, "KERLEK NEZZ A KAMERABA!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Kalibracio... {int((calibration_frames / MAX_CALIBRATION_FRAMES) * 100)}%", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.rectangle(frame, (w // 2 - 100, h // 2 - 120), (w // 2 + 100, h // 2 + 120), (255, 255, 255), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark]

            # Alap EAR szamitas
            left_ear_val = eye_aspect_ratio_3d(landmarks, LEFT_EYE)
            right_ear_val = eye_aspect_ratio_3d(landmarks, RIGHT_EYE)
            avg_ear = (left_ear_val + right_ear_val) / 2.0

            # Fejtartas kiszamitasa
            image_points = np.array([(landmarks[i][0], landmarks[i][1]) for i in PNP_IMAGE_POINTS_IDX], dtype="double")
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4, 1))
            success, rvec, tvec = cv2.solvePnP(PNP_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)
            yaw_angle, pitch_angle, roll_angle = 0, 0, 0
            if success:
                pitch_angle, yaw_angle, roll_angle = get_euler_angles(rvec, tvec)

            # EAR korrekcio fejmozgashoz
            correction = 1.0 - (abs(yaw_angle) / 90) * 0.35
            corrected_ear = avg_ear / correction

            # Pislogas szamlalo
            if corrected_ear < EAR_THRESHOLD:
                if blink_ready:
                    blink_count += 1
                    blink_ready = False
            else:
                blink_ready = True

            # BPM
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                blinks_per_minute = (blink_count / elapsed_time) * 60
            if elapsed_time > 60:
                start_time = time.time()
                blink_count = 0

            # Kalibracio
            if not is_calibrated:
                calibration_frames += 1
                calibration_ear_values.append(corrected_ear)
                if calibration_frames >= MAX_CALIBRATION_FRAMES:
                    avg_open_eye = sum(calibration_ear_values) / len(calibration_ear_values)
                    EAR_THRESHOLD = avg_open_eye * 0.75
                    is_calibrated = True
                    locked_face_center = get_face_center(landmarks)
                continue

            # Arckovetes
            face_center = get_face_center(landmarks)
            if locked_face_center is None:
                locked_face_center = face_center
            dist = math.hypot(face_center[0] - locked_face_center[0], face_center[1] - locked_face_center[1])

            # Mindig frissul EAR history
            ear_history.append(corrected_ear)
            smoothed_ear = sum(ear_history) / len(ear_history)

            # Fatigue logika
            if dist > FACE_LOST_THRESHOLD:
                fatigue_status = "ARC NINCS POZICIOBAN"
                fatigue_color = (128, 128, 128)
                eye_closed_frame_counter = 0
                yawn_frame_counter = 0
            else:
                if smoothed_ear < EAR_THRESHOLD:
                    eye_closed_frame_counter += 1
                else:
                    eye_closed_frame_counter = 0

                fatigue_status = "Eber"
                fatigue_color = (0, 255, 0)
                eye_status = "Nyitva"
                eye_color = (0, 255, 0)

                if eye_closed_frame_counter >= MICROSLEEP_FRAMES:
                    fatigue_status = "MICROSLEEP!"
                    fatigue_color = (0, 0, 255)
                    eye_status = "Csukva"
                    eye_color = (0, 0, 255)

                    alarm_active = any(t.name == "alarm_thread" for t in threading.enumerate())

                    if not alarm_active:
                        threading.Thread(target=play_alarm, name="alarm_thread", daemon=True).start()

                elif eye_closed_frame_counter > EYE_CLOSED_FRAMES_THRESHOLD:
                    fatigue_status = "Szem csukva"
                    fatigue_color = (0, 0, 255)
                    eye_status = "Csukva"
                    eye_color = (0, 0, 255)
                else:
                    # BPM es asitas csak eber allapotban
                    if blinks_per_minute > 40:
                        fatigue_status = "Szemfaradtsag (Magas BPM)"
                        fatigue_color = (0, 165, 255)

                    mar = mouth_aspect_ratio(landmarks)
                    if mar > MAR_THRESHOLD:
                        yawn_frame_counter += 1
                    else:
                        yawn_frame_counter = 0

                    if yawn_frame_counter > YAWN_FRAMES_THRESHOLD:
                        fatigue_status = "Asitas"
                        fatigue_color = (0, 0, 255)
            if fatigue_status != last_logged_status:
                status = ["Szem csukva", "MICROSLEEP!", "Asitas", "Szemfaradtsag (Magas BPM)"]
                if fatigue_status in status or (fatigue_status == "Eber" and last_logged_status in status):
                    event = {
                        "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Status": fatigue_status,
                        "Eye": round(float(smoothed_ear), 2),
                    }
                    log_data.append(event)
                    save_to_json(log_data)

                last_logged_status = fatigue_status


            # Kiirasok
            cv2.putText(frame, fatigue_status, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, fatigue_color, 2)
            cv2.putText(frame, f"Szem: {eye_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 2)
            cv2.putText(frame, f"EAR: {smoothed_ear:.2f} (Lim: {EAR_THRESHOLD:.2f})", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"Pislogas: {blink_count}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)
            cv2.putText(frame, f"BPM: {int(blinks_per_minute)}", (w - 200, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (255, 255, 255), 2)

            # Kirajzolasok
            face_outline_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in FACE_OUTLINE]
            cv2.polylines(frame, [np.array(face_outline_pts)], True, (255, 255, 0), 1)

            mouth_outer_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in MOUTH_OUTER]
            mouth_inner_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in MOUTH_INNER]
            cv2.polylines(frame, [np.array(mouth_outer_pts)], True, (0, 200, 200), 1)
            cv2.polylines(frame, [np.array(mouth_inner_pts)], True, (0, 200, 200), 1)

            for idx in LEFT_EYE + RIGHT_EYE:
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 2, eye_color, -1)

    cv2.imshow("Fáradtság Detektálás", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
