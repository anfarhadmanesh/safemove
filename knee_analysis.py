import cv2
import logging
from mediapipe import Image, ImageFormat
import math
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
model_path = "pose_landmarker_lite.task"
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO, 
    num_poses=1
)

detector = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    # a =Hip، b =Knee، c =Ankle
    angle = math.degrees(math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0]))
    angle = abs(angle)
    if angle > 180.0:
        angle = 360.0 - angle
    return angle

video_path = 'test.mp4' 
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    logging.error("Error!")
    exit()
logging.info("Running... Press 'q' to exit the video")
cv2.namedWindow('Biomechanics Analysis', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Biomechanics Analysis', 800, 600)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break


    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = Image(image_format=ImageFormat.SRGB, data=rgb_frame)  
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))
    detection_result = detector.detect_for_video(mp_image, timestamp_ms)

    if detection_result.pose_landmarks:
            landmarks = detection_result.pose_landmarks[0]
            h, w, _ = frame.shape            
            hip = [landmarks[24].x * w, landmarks[24].y * h]
            knee = [landmarks[26].x * w, landmarks[26].y * h]
            ankle = [landmarks[28].x * w, landmarks[28].y * h]

            angle = calculate_angle(hip, knee, ankle)      

            cv2.putText(frame, f"Knee Angle: {int(angle)} deg", 
                        (int(knee[0]), int(knee[1]) - 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            for point in [hip, knee, ankle]:
                cv2.circle(frame, (int(point[0]), int(point[1])), 5, (0, 0, 255), -1)
    cv2.imshow('Biomechanics Analysis', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

logging.info("Video finished. Press any key to exit")
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()