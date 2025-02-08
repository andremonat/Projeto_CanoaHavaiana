import cv2

import numpy as np
import math
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# img = cv2.imread("./images/mulher.jpg")
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# img = cv2.resize(img, (500,500), interpolation= cv2.INTER_LINEAR)

# imgplot = plt.imshow(img)
# plt.show()

# pose = mp_pose.Pose()
# results = pose.process(img)

# #results.pose_landmarks.landmark
# mp_drawing.draw_landmarks(
        
#         img,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
# for i , item  in enumerate(results.pose_landmarks.landmark):
#     print(i,item)
#     x = 500*item.x
#     y = 500*item.y
    
    
#     cv2.putText(img, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
#     #cv2.putText(img, i, (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)


# imgplot = plt.imshow(img)
# plt.show()

video = cv2.VideoCapture(0)
video.set(cv2.CAP_PROP_FPS, 10)
video.set(3,500)
video.set(4,500) # 200*200

global counter
counter=0
global qstage 
stage = 'initial'

def calculate_angle(start,middle,end):
    
  
    a = np.array(start) # First
    b = np.array(middle) # Mid
    c = np.array(end) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
    
    return angle


while(True):
    #inside while loop
    _,frame = video.read()
    pose= mp_pose.Pose()
    #frame = cv2.resize(frame, (200,200), interpolation= cv2.INTER_LINEAR)
    #pose detection logic
    with mp_pose.Pose(
        static_image_mode=True, min_detection_confidence=0.5, model_complexity=2) as pose:
        results = pose.process(frame)
        mp_drawing.draw_landmarks(      
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        try:
            for i , item  in enumerate(results.pose_landmarks.landmark):
                x = 500*item.x
                y = 500*item.y
                cv2.putText(frame, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
        
            start,mid,end = [],[],[]
            try:
                for i , item  in enumerate(results.pose_landmarks.landmark):
                    #global counter
                    #global stage
                    x =  500*item.x
                    y =  500*item.y
                    if (i == 12):
                        start.append(item.x)
                        start.append(item.y)
                        cv2.putText(frame, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    elif(i== 14):
                        mid.append(item.x)
                        mid.append(item.y)
                        cv2.putText(frame, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                    elif(i == 16):
                        end.append(item.x)
                        end.append(item.y)
                        cv2.putText(frame, str(i), (int(x),int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        angle = calculate_angle(start,mid,end)
                        #print(angle)
                        cv2.putText(frame, str(angle), (70,70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)
                        cv2.putText(frame, str(counter), (50,50), cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 0), 3, cv2.LINE_AA)
                        if(angle > 120 ):
                            stage = 'down'
                        if angle < 60 and stage == 'down':
                            counter = counter + 1
                            stage = 'up'
                            #print(counter) 
            except:
                pass
        except:
            pass
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("counter=",counter)
        break

video.release()
cv2.destroyAllWindows()

