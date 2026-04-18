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
import os

# Konfiguráció és naplózás
os.makedirs("Vizuális adatok", exist_ok=True)
log_filename = "Vizuális adatok/fatigue_log.json"
last_logged_status = None
log_data = []

# MediaPipe Face Mesh inicializálása, finomított landmarkokkal a pontosabb EAR számításhoz
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)

# Landmark Indexek ( MediaPipe specifikus)
Left_eye = [33, 160, 158, 133, 153, 144]
Right_eye = [362, 385, 387, 263, 373, 380]
Mouth_top = 13
Mouth_bottom = 14
Mouth_left = 61
Mouth_right = 291
PnP_image_points_idx = [1, 152, 33, 263, 61, 291] # Orr, ál, szemzugok, szájszélek
PnP_model_points = np.array([
    (0.0, 0.0, 0.0),            # Orrhegy
    (0.0, -330.0, -65.0),       # Ál
    (-225.0, 170.0, -135.0),    # Bal szem bal sarka
    (225.0, 170.0, -135.0),     # Jobb szem jobb sarka
    (-150.0, -150.0, -125.0),   # Bal szájszél
    (150.0, -150.0, -125.0)     # Jobb szájszél
], dtype="double")

# vizualizációs kontúrok indexei
Face_outline = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361,
                288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149,
                150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54,
                103, 67, 109]
Mouth_outer = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308]
Mouth_inner = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308]

# Kamera beállítási pont, 0 = beépített kamera, 1,2 = USB-n keresztül csatlakoztatott kamera
cap = cv2.VideoCapture(0)
# Mozgóátlag a zajszűréshez
ear_history = deque(maxlen=5)

# Kalibrációs fázis változói
is_calibrated = False
calibration_frames = 0
Max_calibration_frames = 100
calibration_ear_values = []
EAR_threshold = 0.25  # Default érték, ha nem sikerül a kalibráció

eye_closed_start_time = None  # Eltárolja az időbélyeget, amikor lecsukódott a szem
closed_duration = 0.0   #lecsukott szem eltelt idő másodpercben
# Időzítési küszöbök a szem csukottsági állapotának megállapításához (másodpercben)
Microsleep_time = 3.0
Sleep_time = 15.0

# Statisztikai változók
blink_count = deque()
blink_ready = True
mar = 0.0
MAR_threshold = 0.5
Yawn_frames_threshold = 60
Yawn_frame_counter = 0

# Fej dőlés
Pitch_threshold = 25
#Yaw_threshold = 30
Roll_threshold = 20
Head_tilt_frames = 30
Head_tilt_frame_counter = 0
calibration_pitch = []
calibration_yaw = []
calibration_roll = []
baseline_pitch = 0.0
baseline_yaw = 0.0
baseline_roll = 0.0

locked_face_center = None

# Segédfüggvények
def save_to_json(data):
    with open(log_filename, 'w', encoding= 'utf-8') as f:
        json.dump(data, f)

# Riasztás megszólaltatása külön szálon, hogy ne akassza meg a videófolyamot
def play_alarm():
    winsound.Beep(3000,600)

def trigger_alarm():
    alarm_active = any(t.name == "alarm_thread" for t in threading.enumerate())
    if not alarm_active:
        threading.Thread(target=play_alarm, name="alarm_thread", daemon=True).start()

# Euklideszi távolság számítása 3D térben
def distance_3d(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2)

# EAR kiszámítása Soukupova és Cech képlete alapján a distance_3d függvény ötvözésével
# függőleges távolságok átlaga / vízszintes távolság
def eye_aspect_ratio_3d(lm_list, eye_indices):
    p = [lm_list[i] for i in eye_indices]
    return (distance_3d(p[1], p[5]) + distance_3d(p[2], p[4])) / (2.0 * distance_3d(p[0], p[3]))

# MAR számítása az ásítás detektálásához a distance_3d függvény ötvözésével
def mouth_aspect_ratio(lm_list):
    vertical = distance_3d(lm_list[Mouth_top], lm_list[Mouth_bottom])
    horizontal = distance_3d(lm_list[Mouth_left], lm_list[Mouth_right])
    if horizontal == 0: return 0
    return vertical / horizontal

# A rotációs mátrixból kinyeri a fej dőlésszögeit (Pitch, Yaw, Roll)
# Segít az EAR korrekciójában, ha a felhasználó nem pont a kamerába néz
def get_euler_angles(rotation_vec, translation_vec):
    rmat, _ = cv2.Rodrigues(rotation_vec)
    singularcheck = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = singularcheck < 1e-6

    if not singular:
        pitch = math.atan2(rmat[2, 1], rmat[2, 2])
        yaw = math.atan2(-rmat[2, 0], singularcheck)
        roll = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        pitch = math.atan2(-rmat[1, 2], rmat[1, 1])
        yaw = math.atan2(-rmat[2, 0], singularcheck)
        roll = 0

    face_distance = np.linalg.norm(translation_vec)
    return np.degrees(pitch), np.degrees(yaw), np.degrees(roll), face_distance

# Az arc mértani középpontjának meghatározása a pozíciókövetéshez
def get_face_center(lm_list):
    xs = [pt[0] for pt in lm_list]
    ys = [pt[1] for pt in lm_list]
    return int(sum(xs) / len(xs)), int(sum(ys) / len(ys))

# Főciklus

while True:
    live, frame = cap.read()
    if not live:
        break

    eye_status = "Ismeretlen"
    eye_color = (255, 255, 255)
    rel_pitch = 0.0
    rel_yaw = 0.0
    rel_roll = 0.0
    pitch_angle = 0.0
    yaw_angle = 0.0
    roll_angle = 0.0

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)
    h, w, _ = frame.shape
    frame_diagonal = math.sqrt(w ** 2 + h ** 2)
    Face_lost_threshold = frame_diagonal * 0.30

    # Kalibrációs instrukciók
    if not is_calibrated:
        cv2.putText(frame, "Look at the camera!", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
        cv2.putText(frame, f"Calibration... {int((calibration_frames / Max_calibration_frames) * 100)}%", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
        cv2.rectangle(frame, (w // 2 - 100, h // 2 - 120), (w // 2 + 100, h // 2 + 120), (255, 255, 255), 2)

    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark]

            # EAR és Orientáció számítás
            left_ear_val = eye_aspect_ratio_3d(landmarks, Left_eye)
            right_ear_val = eye_aspect_ratio_3d(landmarks, Right_eye)
            avg_ear = (left_ear_val + right_ear_val) / 2.0

            # EPnP algoritmus futtatása a fej dőlésszögének meghatározásához
            image_points = np.array([(landmarks[i][0], landmarks[i][1]) for i in PnP_image_points_idx], dtype="double")
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]],
                                      [0, focal_length, center[1]],
                                      [0, 0, 1]], dtype="double")
            dist_coeffs = np.zeros((4, 1))
            success, rvec, tvec = cv2.solvePnP(PnP_model_points, image_points, camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_EPNP)
            yaw_angle, pitch_angle, roll_angle = 0, 0, 0
            if success:
                pitch_angle, yaw_angle, roll_angle, dist = get_euler_angles(rvec, tvec)

            # EAR korrekció, ha elfordítjuk a fejünket, a szem perspektivikusan szűkebbnek tűnik.
            # Ezt matematikai korrekcióval kompenzálom a téves riasztások elkerülése végett.
            correction_yaw = 1.0 - (abs(yaw_angle) / 90) * 0.35
            correction_pitch = 1.0 - (abs(pitch_angle) / 90) * 0.35
            full_correction = max(0.5, correction_yaw * correction_pitch)
            corrected_ear = avg_ear / full_correction

            # Automatizált kalibráció futtatása
            if not is_calibrated:
                calibration_frames += 1
                calibration_ear_values.append(corrected_ear)
                calibration_pitch.append(pitch_angle)
                calibration_yaw.append(yaw_angle)
                calibration_roll.append(roll_angle)

                if calibration_frames >= Max_calibration_frames:
                    avg_open_eye = sum(calibration_ear_values) / len(calibration_ear_values)
                    EAR_threshold = avg_open_eye * 0.75
                    baseline_pitch = sum(calibration_pitch) / len(calibration_pitch)
                    baseline_yaw = sum(calibration_yaw) / len(calibration_yaw)
                    baseline_roll = sum(calibration_roll) / len(calibration_roll)
                    is_calibrated = True
                    locked_face_center = get_face_center(landmarks)
                continue
            # EAR simítás mozgóátlaggal
            ear_history.append(corrected_ear)
            smoothed_ear = sum(ear_history) / len(ear_history)

            # Pislogás detektálás (állapotgép alapú: Ready -> Blink -> Reset)
            if corrected_ear < EAR_threshold:
                if blink_ready:
                    blink_count.append(time.time())
                    blink_ready = False
            else:
                blink_ready = True

            # Blinks per minute dinamikus számítása
            now = time.time()
            while blink_count and now - blink_count[0] > 60:
                blink_count.popleft()
            blinks_per_minute = len(blink_count)

            # Arckövetés (távolság az eredeti kalibrált középponthoz képest)
            face_center = get_face_center(landmarks)
            if locked_face_center is None:
                locked_face_center = face_center
            dist = math.hypot(face_center[0] - locked_face_center[0], face_center[1] - locked_face_center[1])

            # Fáradtság logikai döntési fa
            if dist > Face_lost_threshold:
                fatigue_status = "Lost face"
                fatigue_color = (128, 128, 128)
                eye_closed_frame_counter = 0
                Yawn_frame_counter = 0
            else:
                if smoothed_ear < EAR_threshold:
                    if eye_closed_start_time is None:
                        eye_closed_start_time = time.time()
                    closed_duration = time.time() - eye_closed_start_time
                else:
                    eye_closed_start_time = None
                    closed_duration = 0.0

                rel_pitch = pitch_angle - baseline_pitch
                rel_yaw = yaw_angle - baseline_yaw
                rel_roll = roll_angle - baseline_roll
                # Fejdőlés
                if smoothed_ear < EAR_threshold and (abs(rel_pitch) > Pitch_threshold or abs(rel_roll) > Roll_threshold):
                    Head_tilt_frame_counter += 1
                else:
                    Head_tilt_frame_counter = 0

                # Alapértelmezett állapot
                fatigue_status = "Awake"
                fatigue_color = (0, 255, 0)
                eye_status = "Open"
                eye_color = (0, 255, 0)

                #Kritikus állapotok ellenőrzése (hierarchikus prioritás)
                if closed_duration > Sleep_time:
                    fatigue_status = "Sleep"
                    fatigue_color = (0, 0, 255)
                    eye_status = "Close"
                    eye_color = (0, 0, 255)

                    trigger_alarm()


                elif closed_duration>= Microsleep_time:
                    fatigue_status = "Microsleep"
                    fatigue_color = (0, 0, 255)
                    eye_status = "Close"
                    eye_color = (0, 0, 255)

                    trigger_alarm()

                elif Head_tilt_frame_counter > Head_tilt_frames:
                    fatigue_status = "Head tilt"
                    fatigue_color = (0, 165, 255)

                    trigger_alarm()

                elif corrected_ear < EAR_threshold:
                    fatigue_status = "Blink"
                    fatigue_color = (0, 0, 255)
                    eye_status = "Close"
                    eye_color = (0, 0, 255)

                else:
                    # Kiegészítő fáradtsági jelek pislogásszám és ásítás
                    if blinks_per_minute > 40:
                        fatigue_status = "High BPM"
                        fatigue_color = (0, 165, 255)

                    mar = mouth_aspect_ratio(landmarks)
                    if mar > MAR_threshold:
                        Yawn_frame_counter += 1
                    else:
                        Yawn_frame_counter = 0

                    if Yawn_frame_counter > Yawn_frames_threshold:
                        fatigue_status = "Yawn"
                        fatigue_color = (0, 0, 255)

            # Eseményvezérelt naplózás, csak ha az állapot megváltozik
            if fatigue_status != last_logged_status:
                status = ["Head tilt", "Microsleep", "Yawn", "High BPM", "Sleep", "Blink"]
                if fatigue_status in status or (fatigue_status == "Awake" and last_logged_status in status):
                    event = {
                        "DateTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Status": fatigue_status,
                        "EAR": round(float(corrected_ear), 2),
                    }
                    log_data.append(event)
                    save_to_json(log_data)

                last_logged_status = fatigue_status

            # Vizuális statisztikai megjelenítés a teszteléshez
            cv2.putText(frame, fatigue_status, (30, 170), cv2.FONT_HERSHEY_SIMPLEX, 1, fatigue_color, 2)
            cv2.putText(frame, f"Eye: {eye_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 2)
            cv2.putText(frame, f"EAR: {corrected_ear:.2f} (Lim: {EAR_threshold:.2f})", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"CD: {closed_duration:.1f} sec", (30, 110), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        fatigue_color, 2)
            cv2.putText(frame, f"MAR: {mar:.2f} (Lim: {MAR_threshold:.2f})", (30, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(frame, f"Blinks / min: {len(blink_count)}", (w - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.putText(frame, f"Pitch(rel): {rel_pitch:.1f}", (w - 200, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.putText(frame, f"Yaw(rel): {rel_yaw:.1f}", (w - 200, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255),2)
            cv2.putText(frame, f"Roll(rel): {rel_roll:.1f}", (w - 200, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 255), 2)
            cv2.putText(frame,f"Time: {datetime.now()}", (30,450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

            # Kontúrok kirajzolása az arcra, vizuális visszajelzés a megfelelő működésről
            face_outline_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in Face_outline]
            cv2.polylines(frame, [np.array(face_outline_pts)], True, (255, 255, 0), 1)

            mouth_outer_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in Mouth_outer]
            mouth_inner_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in Mouth_inner]
            cv2.polylines(frame, [np.array(mouth_outer_pts)], True, (0, 200, 200), 1)
            cv2.polylines(frame, [np.array(mouth_inner_pts)], True, (0, 200, 200), 1)

            for idx in Left_eye + Right_eye:
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (int(x), int(y)), 2, eye_color, -1)

    # Megjelenítés és kilépés kezelése
    cv2.imshow("Fatigue detector ", frame)
    if cv2.waitKey(30) & 0xFF == 27: #ESC gomb
        break

cap.release()
cv2.destroyAllWindows()
