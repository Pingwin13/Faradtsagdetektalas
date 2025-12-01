import cv2
import mediapipe as mp
import math
import numpy as np
from collections import deque

# Beállítások és Függvények

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)


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


def get_euler_angles(rvec, tvec, camera_matrix, dist_coeffs):
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
UPPER_LIP_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
UPPER_LIP_INNER = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]

# Paraméterek

cap = cv2.VideoCapture(0)
ear_history = deque(maxlen=5)

#Kalibrációs vátozók
is_calibrated = False
calibration_frames = 0
MAX_CALIBRATION_FRAMES = 100
calibration_ear_values = []
EAR_THRESHOLD = 0.15  # Amennyiben nem sikerül kalibrálni ez az alapérték.

#Szem időzítés
eye_closed_frame_counter = 0
EYE_CLOSED_FRAMES_THRESHOLD = 3


DIST_THRESHOLD = 100
PITCH_DROP_THRESHOLD = -15
ROLL_SLUMP_THRESHOLD = 20
MAR_THRESHOLD = 0.6
YAWN_FRAMES_THRESHOLD = 20
yawn_frame_counter = 0

locked_face_center = None

# Főciklus

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    h, w, _ = frame.shape

    #Kalibrációs üzenet
    if not is_calibrated:
        cv2.putText(frame, "KERLEK NEZZ A KAMERABA!", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, f"Kalibracio... {int((calibration_frames / MAX_CALIBRATION_FRAMES) * 100)}%", (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.rectangle(frame, (w // 2 - 100, h // 2 - 120), (w // 2 + 100, h // 2 + 120), (255, 255, 255), 2)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = [(lm.x * w, lm.y * h, lm.z * w) for lm in face_landmarks.landmark]

            # EAR Számítás
            left_ear_val = eye_aspect_ratio_3d(landmarks, LEFT_EYE)
            right_ear_val = eye_aspect_ratio_3d(landmarks, RIGHT_EYE)
            avg_ear = (left_ear_val + right_ear_val) / 2.0

            # Yaw szög kell a korrekcióhoz
            image_points = np.array([(landmarks[i][0], landmarks[i][1]) for i in PNP_IMAGE_POINTS_IDX], dtype="double")
            focal_length = w
            center = (w / 2, h / 2)
            camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                                     dtype="double")
            dist_coeffs = np.zeros((4, 1))
            success, rvec, tvec = cv2.solvePnP(PNP_MODEL_POINTS, image_points, camera_matrix, dist_coeffs,
                                               flags=cv2.SOLVEPNP_ITERATIVE)

            yaw_angle = 0
            pitch_angle = 0
            roll_angle = 0
            if success:
                pitch_angle, yaw_angle, roll_angle = get_euler_angles(rvec, tvec, camera_matrix, dist_coeffs)

            correction = 1.0 - (abs(yaw_angle) / 90) * 0.35
            corrected_ear = avg_ear / correction

            #Detektálás

            if not is_calibrated:
                #Adatgyűjtés
                calibration_frames += 1
                calibration_ear_values.append(corrected_ear)

                # Ha letelt az idő
                if calibration_frames >= MAX_CALIBRATION_FRAMES:
                    # Átlag kiszámolása
                    avg_open_eye = sum(calibration_ear_values) / len(calibration_ear_values)
                    EAR_THRESHOLD = avg_open_eye * 0.75
                    is_calibrated = True
                    print(f"Kalibráció kész! Átlag EAR: {avg_open_eye:.3f}, Küszöb: {EAR_THRESHOLD:.3f}")
                    x_coords = [pt[0] for pt in landmarks]
                    y_coords = [pt[1] for pt in landmarks]
                    locked_face_center = (int(sum(x_coords) / len(x_coords)), int(sum(y_coords) / len(y_coords)))

            else:

                # Fej követés
                x_coords = [pt[0] for pt in landmarks]
                y_coords = [pt[1] for pt in landmarks]
                face_center = (int(sum(x_coords) / len(x_coords)), int(sum(y_coords) / len(y_coords)))

                # Ha véletlen nincs lockolt center (hiba esetén), beállítjuk
                if locked_face_center is None: locked_face_center = face_center

                dist = math.hypot(face_center[0] - locked_face_center[0], face_center[1] - locked_face_center[1])

                if dist < DIST_THRESHOLD:
                    cv2.circle(frame, face_center, 8, (0, 255, 0), -1)

                    # EAR Simítás
                    ear_history.append(corrected_ear)
                    smoothed_ear = sum(ear_history) / len(ear_history)

                    if smoothed_ear < EAR_THRESHOLD:
                        eye_closed_frame_counter += 1
                    else:
                        eye_closed_frame_counter = 0

                    if eye_closed_frame_counter > EYE_CLOSED_FRAMES_THRESHOLD:
                        eye_status = "Csukva"
                        eye_color = (0, 0, 255)
                    else:
                        eye_status = "Nyitva"
                        eye_color = (0, 255, 0)

                    # Fáradtság állapotok
                    fatigue_status = "Eber"
                    fatigue_color = (0, 255, 0)

                    if pitch_angle < PITCH_DROP_THRESHOLD:
                        fatigue_status = "Bolinto fej"
                        fatigue_color = (0, 0, 255)
                    elif abs(roll_angle) > ROLL_SLUMP_THRESHOLD:
                        fatigue_status = "Oldalra dolt fej"
                        fatigue_color = (0, 0, 255)

                    mar = mouth_aspect_ratio(landmarks)
                    if mar > MAR_THRESHOLD:
                        yawn_frame_counter += 1
                    else:
                        yawn_frame_counter = 0

                    if yawn_frame_counter > YAWN_FRAMES_THRESHOLD:
                        if fatigue_status == "Eber":
                            fatigue_status = "Asitas"
                            fatigue_color = (0, 0, 255)

                    # Kiirasok
                    cv2.putText(frame, fatigue_status, (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, fatigue_color, 2)
                    cv2.putText(frame, f"Szem: {eye_status}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, eye_color, 2)

                    cv2.putText(frame, f"EAR: {smoothed_ear:.2f} (Lim: {EAR_THRESHOLD:.2f})", (30, 80),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

                    # Kirajzolasok
                    face_outline_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in FACE_OUTLINE]
                    cv2.polylines(frame, [np.array(face_outline_pts)], True, (255, 255, 0), 1)

                    mouth_outer_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in MOUTH_OUTER]
                    mouth_inner_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in MOUTH_INNER]
                    cv2.polylines(frame, [np.array(mouth_outer_pts)], True, (0, 200, 200), 1)
                    cv2.polylines(frame, [np.array(mouth_inner_pts)], True, (0, 200, 200), 1)

                    upper_outer_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in UPPER_LIP_OUTER]
                    upper_inner_pts = [(int(landmarks[i][0]), int(landmarks[i][1])) for i in UPPER_LIP_INNER]
                    cv2.polylines(frame, [np.array(upper_outer_pts)], False, (0, 200, 200), 1)
                    cv2.polylines(frame, [np.array(upper_inner_pts)], False, (0, 200, 200), 1)

                    for idx in LEFT_EYE + RIGHT_EYE:
                        x, y, _ = landmarks[idx]
                        cv2.circle(frame, (int(x), int(y)), 2, eye_color, -1)

    cv2.imshow("Fáradtság Detektálás", frame)
    if cv2.waitKey(30) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()