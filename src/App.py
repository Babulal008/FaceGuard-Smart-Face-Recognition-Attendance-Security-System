import cv2
import face_recognition
import numpy as np
import sqlite3
from datetime import datetime
import os

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Connect to SQLite database (or create it if it doesn't exist)
conn = sqlite3.connect('attendance.db')
cursor = conn.cursor()

# Create the table for storing attendance if it doesn't exist
cursor.execute('''CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    date TEXT,
                    time TEXT,
                    status TEXT)''')

# Create a directory for storing known faces and their encodings
known_face_encodings = []
known_face_names = []

# Load known faces and their names from a directory
def load_known_faces():
    global known_face_encodings, known_face_names
    if not os.path.exists('known_faces'):
        os.makedirs('known_faces')
    
    # Load known faces from the "known_faces" directory
    for filename in os.listdir('known_faces'):
        if filename.endswith(".jpg"):
            img = face_recognition.load_image_file(f"known_faces/{filename}")
            encoding = face_recognition.face_encodings(img)[0]
            known_face_encodings.append(encoding)
            known_face_names.append(filename.split(".")[0])

# Record attendance in the database
def log_attendance(name, status="Arrived"):
    current_date = datetime.now().date()
    current_time = datetime.now().strftime('%H:%M:%S')
    cursor.execute("INSERT INTO attendance (name, date, time, status) VALUES (?, ?, ?, ?)",
                   (name, current_date, current_time, status))
    conn.commit()

# Real-time alert for unknown faces
def alert_unknown_face(name):
    print(f"ALERT: Unknown face detected! Name: {name}")

# Image Capture for New Users
def capture_new_user(name):
    print(f"New face detected! Capturing image of {name} for future recognition.")
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(f'known_faces/{name}.jpg', frame)

# Face recognition loop
def face_recognition_system():
    load_known_faces()  # Load known faces at the beginning
    face_locations = []
    face_encodings = []
    face_names = []
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Convert the image from BGR to RGB (as face_recognition uses RGB)
        rgb_frame = frame[:, :, ::-1]

        # Find all the faces in the current frame
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

        # Loop through each face found in the frame
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

            name = "Unknown"

            # If a match is found, use the known name
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]

                # Log attendance if a known face is detected
                log_attendance(name)
                print(f"Attendance recorded for: {name}")
            else:
                # If the face is not recognized, capture it as a new user
                name = "New User"
                alert_unknown_face(name)
                capture_new_user(name)

            # Draw a rectangle around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Label the face
            cv2.putText(frame, name, (left, bottom + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the resulting frame
        cv2.imshow("Face Recognition System", frame)

        # Press 'q' to quit the loop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Close the database connection and the webcam when done
def cleanup():
    conn.close()
    cap.release()
    cv2.destroyAllWindows()

# Query attendance for a specific person
def query_attendance(name):
    cursor.execute("SELECT * FROM attendance WHERE name=?", (name,))
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            print(f"Name: {row[1]}, Date: {row[2]}, Time: {row[3]}, Status: {row[4]}")
    else:
        print(f"No attendance found for {name}.")

if __name__ == "__main__":
    try:
        face_recognition_system()
    except KeyboardInterrupt:
        print("\nProgram exited.")
    finally:
        cleanup()

    # Sample query after exiting the loop:
    # Query the attendance for a person (e.g., "Person 1")
    query_attendance("Person 1")
