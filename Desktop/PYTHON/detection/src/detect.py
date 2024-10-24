import cv2
import mediapipe as mp
import face_recognition
import os
import subprocess

# Initialize MediaPipe and OpenCV components
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Initialize face recognition
known_face_encodings = []
known_face_names = []

# Load known faces from the assets folder
def load_known_faces(known_faces_dir):
    for image_name in os.listdir(known_faces_dir):
        image_path = os.path.join(known_faces_dir, image_name)
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        name = os.path.splitext(image_name)[0]  # Use filename as name
        known_face_encodings.append(encoding)
        known_face_names.append(name)

# Load Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize background subtractor for motion detection
fgbg = cv2.createBackgroundSubtractorMOG2()

def detect_gestures(hand_landmarks):
    if hand_landmarks:
        # Example: Detect pinch gesture
        thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
        distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5
        
        if distance < 0.05:  # Example threshold for pinch
            print("Pinch gesture detected")
            adjust_volume()

def adjust_volume(increase=True):
    """Increase or decrease system volume in real-time."""
    volume_step = 5  # Volume step in percentage
    
    # Command to get current volume level
    cmd_get_volume = ['amixer', 'get', 'Master']
    
    # Get current volume level
    result = subprocess.run(cmd_get_volume, stdout=subprocess.PIPE)
    output = result.stdout.decode()

    # Parse current volume percentage
    volume_str = output.split('[')[1].split('%')[0]
    current_volume = int(volume_str)
    
    # Determine new volume level
    if increase:
        new_volume = min(current_volume + volume_step, 100)
    else:
        new_volume = max(current_volume - volume_step, 0)
    
    # Set new volume level
    cmd_set_volume = ['amixer', 'set', 'Master', f'{new_volume}%']
    subprocess.run(cmd_set_volume)
    
    # Print volume change for debugging
    print(f"Volume changed to {new_volume}%")

def recognize_faces(frame, face_locations):
    face_encodings = face_recognition.face_encodings(frame, face_locations)
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)
    return face_names

def main():
    # Load known faces
    load_known_faces('assets/known_faces')

    # Open the default webcam
    cap = cv2.VideoCapture(0)
   

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Motion detection
        fgmask = fgbg.apply(frame)
        _, thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)

        # Face detection
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        face_locations = [(y, x+w, y+h, x) for (x, y, w, h) in faces]

        # Facial recognition
        face_names = recognize_faces(frame, face_locations)
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
            cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        # Hand gesture detection
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                detect_gestures(hand_landmarks)

        # Display results
        cv2.imshow('Motion Detection', thresh)
        cv2.imshow('Face and Hand Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
