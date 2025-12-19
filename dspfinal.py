from ultralytics import YOLO
import cv2
import datetime
import os
from collections import deque
import pygame

#parameters
MONITOR_START_HOUR = 12
MONITOR_END_HOUR = 18  
BUFFER_SIZE = 10         
DETECTION_THRESHOLD = 8 
SAVE_COOLDOWN_FRAMES = 80 

#initialization
pygame.mixer.init()
try:
    alarm_sound = pygame.mixer.Sound("alarm.mp3") 
except:
    print("WARNING: 'alarm.mp3' not found. Audio disabled.")
    alarm_sound = None

if not os.path.exists('detections'):
    os.makedirs('detections')

#create the CSV header if the file doesn't exist
if not os.path.exists("detection_log.csv"):
    with open("detection_log.csv", "w", newline='') as f:
        f.write("Timestamp,Signal_Strength,Image_Path\n")

model = YOLO("yolo-Weights/yolov8n.pt")
cap = cv2.VideoCapture(0)

signal_window = deque(maxlen=BUFFER_SIZE)
frames_since_last_save = SAVE_COOLDOWN_FRAMES 
alarm_active = False # State variable to track audio

def is_monitoring_time():
    now = datetime.datetime.now().hour
    if MONITOR_START_HOUR > MONITOR_END_HOUR:
        return now >= MONITOR_START_HOUR or now <= MONITOR_END_HOUR
    return MONITOR_START_HOUR <= now <= MONITOR_END_HOUR

while True:
    success, img = cap.read()
    if not success: break
    
    active_monitoring = is_monitoring_time()
    person_in_this_frame = False

    if active_monitoring:
        results = model(img, verbose=False)
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0 and box.conf[0] > 0.5:
                    person_in_this_frame = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 2)

        #signal processing
        signal_window.append(1 if person_in_this_frame else 0)
        current_signal_strength = sum(signal_window)
        
        #alarm logic
        if current_signal_strength >= DETECTION_THRESHOLD:
            # start alarm if not already running when signal strength over detection treshold
            if not alarm_active and alarm_sound:
                alarm_sound.play(loops=-1) # loop indefinitely
                alarm_active = True
            
            # save img and log csv
            if frames_since_last_save >= SAVE_COOLDOWN_FRAMES:
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                file_path = f"detections/intruder_{timestamp}.jpg"
                
                # save img
                cv2.imwrite(file_path, img)
                
                # log to csv
                try:
                    with open("detection_log.csv", "a", newline='') as f:
                        f.write(f"{timestamp},{current_signal_strength},{file_path}\n")
                        f.flush()     
                        os.fsync(f.fileno()) 
                    print(f"[LOGGED] {timestamp} - Signal: {current_signal_strength}")
                except Exception as e:
                    print(f"CSV Error: {e} (Is the file open in Excel?)")

                frames_since_last_save = 0 
        
        else:
            # when signal is low, stop alarm if it is running
            if alarm_active and alarm_sound:
                alarm_sound.stop()
                alarm_active = False

        frames_since_last_save += 1

        #ui
        cv2.rectangle(img, (0, 0), (320, 130), (0, 0, 0), -1)
        
        status_text = "ALARM TRIGGERED" if alarm_active else "MONITORING ACTIVE"
        status_color = (0, 0, 255) if alarm_active else (0, 255, 0)
        
        cv2.putText(img, status_text, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        cv2.putText(img, f"Signal Filter: {current_signal_strength}/{BUFFER_SIZE}", (15, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        #signal bar
        bar_width = int((current_signal_strength / BUFFER_SIZE) * 280)
        bar_fill_color = (0, 0, 255) if alarm_active else (255, 255, 0)
        cv2.rectangle(img, (15, 95), (295, 115), (50, 50, 50), -1)
        cv2.rectangle(img, (15, 95), (15 + bar_width, 115), bar_fill_color, -1)

    else:
        #stop alarm if hours change to outside active hours while it's ringing
        if alarm_active and alarm_sound:
            alarm_sound.stop()
            alarm_active = False
            
        img = cv2.addWeighted(img, 0.3, img, 0, 0)
        cv2.putText(img, "SYSTEM IDLE", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        signal_window.clear()

    cv2.imshow('Human Detector', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
pygame.mixer.quit()