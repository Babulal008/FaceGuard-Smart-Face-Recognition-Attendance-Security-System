import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime
import os

# Initialize webcam
cap = cv2.VideoCapture(0)

# Create directories if not exist
os.makedirs('known_faces', exist_ok=True)
os.makedirs('unknown_faces', exist_ok=True)

# Connect to SQLite database
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Create attendance table
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    date TEXT,
    time TEXT,
    status TEXT
)''')

known_face_encodings = []
known_face_names = []

def load_known_faces():
    known_face_encodings.clear()
    known_face_names.clear()
    for filename in os.listdir('known_faces'):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            path = os.path.join('known_faces', filename)
            img = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(os.path.splitext(filename)[0])

def already_marked_today(name):
    today = datetime.now().date()
    cursor.execute("SELECT * FROM attendance WHERE name=? AND date=?", (name, str(today)))
    return cursor.fetchone() is not None

def log_attendance(name, status="Arrived"):
    if not already_marked_today(name):
        now = datetime.now()
        cursor.execute("INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
                       (name, now.date(), now.strftime("%H:%M:%S"), status))
        conn.commit()
        print(f"[INFO] Attendance logged for {name}.")

def alert_unknown_face(face_image, index):
    filename = f"unknown_faces/Unknown_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{index}.jpg"
    cv2.imwrite(filename, face_image)
    print(f"[ALERT] Unknown face captured and saved as {filename}.")

def query_attendance(name):
    cursor.execute("SELECT * FROM attendance WHERE name=?", (name,))
    results = cursor.fetchall()
    if results:
        for r in results:
            print(f"Name: {r[1]}, Date: {r[2]}, Time: {r[3]}, Status: {r[4]}")
    else:
        print(f"No records found for {name}.")

def face_recognition_system():
    load_known_faces()
    print(f"[INFO] Loaded {len(known_face_names)} known faces.")

    unknown_faces_logged = set()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = small_frame[:, :, ::-1]

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for index, face_encoding in enumerate(face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            if True in matches:
                best_match_index = matches.index(True)
                name = known_face_names[best_match_index]
                log_attendance(name)
            else:
                if index not in unknown_faces_logged:
                    top, right, bottom, left = face_locations[index]
                    face_img = frame[top*4:bottom*4, left*4:right*4]
                    alert_unknown_face(face_img, index)
                    unknown_faces_logged.add(index)

            face_names.append(name)

        # Draw labels and boxes
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        # Display live face count
        cv2.putText(frame, f"Faces: {len(face_locations)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

        cv2.imshow("FaceGuard - Attendance System", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

def cleanup():
    cap.release()
    conn.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        face_recognition_system()
    except KeyboardInterrupt:
        print("[EXIT] Interrupted by user.")
    finally:
        cleanup()
        query_attendance("Person 1")  # Example query
