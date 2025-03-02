import cv2
import imutils
from imutils import face_utils
import dlib
from scipy.spatial import distance
from datetime import datetime
import numpy as np


def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


thresh = 0.25
frame_check = 20
lStart, lEnd = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
rStart, rEnd = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
p = "c:/Users/Nilesh Chhabra/Desktop/hack-n-win/shape_predictor_68_face_landmarks.dat"
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit(1)

# Load OpenCV's pre-trained Haar cascade for face detection (alternative to dlib)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

sleep_start_time = None
awake_start_time = datetime.now()
sleep_detected = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame = imutils.resize(frame, width=450)
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Ensure it's 8-bit single-channel grayscale image (uint8)
    gray = np.asarray(gray, dtype=np.uint8)

    # Convert to RGB image
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Ensure the image is contiguous in memory
    gray = np.ascontiguousarray(gray, dtype=np.uint8)
    rgb_frame = np.ascontiguousarray(rgb_frame, dtype=np.uint8)

    # Detect faces using OpenCV's Haar Cascade classifier
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        print("No faces detected.")
    else:
        for (x, y, w, h) in faces:
            # Draw rectangle around detected face
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Use dlib to detect facial landmarks for the eyes
            gray_face = gray[y:y + h, x:x + w]
            rgb_face = rgb_frame[y:y + h, x:x + w]
            try:
                detector = dlib.get_frontal_face_detector()
                predictor = dlib.shape_predictor(p)
                subjects = detector(rgb_face, 0)

                for subject in subjects:
                    shape = predictor(gray_face, subject)
                    shape = face_utils.shape_to_np(shape)
                    leftEye = shape[lStart:lEnd]
                    rightEye = shape[rStart:rEnd]
                    leftEar = eye_aspect_ratio(leftEye)
                    rightEar = eye_aspect_ratio(rightEye)
                    ear = (leftEar + rightEar) / 2.0

                    if ear < thresh:
                        sleep_start_time = sleep_start_time or datetime.now()
                        awake_start_time = None
                    else:
                        if sleep_start_time:
                            sleep_detected = True
                            sleep_start_time = None
                            awake_start_time = awake_start_time or datetime.now()

                    if awake_start_time:
                        awake_duration = datetime.now() - awake_start_time
                        if awake_duration.total_seconds() >= 1:  # Reduced duration to switch to AWAKE state
                            awake_start_time = None
                            sleep_detected = False  # Reset sleep detection when eyes are open

            except Exception as e:
                print(f"Error during landmark detection: {e}")

    if sleep_detected:
        cv2.putText(frame, "SLEEPING", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    else:
        cv2.putText(frame, "AWAKE", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("frame", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
cap.release()
