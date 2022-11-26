import cv2
import re
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import ctypes

mp_face_mesh = mp.solutions.face_mesh



# 自作関数
# landmarks は 478 x 3 の二次元 numpy配列
# index と 場所の関係は「https://www.google.com/search?q=mediapipe+facemesh+index+and+position」等で調べる
def draw_landmarks(image, landmarks):
    FACE_TRBL = [ 10,234,152, 454] # 顔の上右下左 の頂点
    EYE_LEFT  = [469,470,471, 472] # https://note.com/kitazaki/n/nbf4fd1e84271
    EYE_RIGHT = [476,475,474, 477]
    EYE_LR    = [468, 473]
    h, w, _ = image.shape

    c = (255,255,255)
    for idx in FACE_TRBL:
        p = landmarks[idx]
        cv2.circle(image, (int(p[0]*w), int(p[1]*h)), 5, c, thickness=2)

    c = (255,0,0)
    for idx in EYE_LEFT:
        p = landmarks[idx]
        cv2.circle(image, (int(p[0]*w), int(p[1]*h)), 2, c, thickness=2)

    c = (0,0,255)
    for idx in EYE_RIGHT:
        p = landmarks[idx]
        cv2.circle(image, (int(p[0]*w), int(p[1]*h)), 2, c, thickness=2)

    c = (0,255,255)
    for idx in EYE_LR:
        p = landmarks[idx]
        cv2.circle(image, (int(p[0]*w), int(p[1]*h)), 4, c, thickness=2)





def main():
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh( max_num_faces=1,
                                       refine_landmarks=True,
                                       min_detection_confidence=0.5,
                                       min_tracking_confidence=0.5)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("cant read frame..")
            continue
        h,w,_ = image.shape

        #RGB画像を利用して landmarkを推論
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results.multi_face_landmarks is not None:

            # landmarkを描く (自作関数)
            face_landmarks = results.multi_face_landmarks[0].landmark # 最初の顔のみ
            landmarks = []
            for lm in face_landmarks:
                landmarks.append([lm.x,lm.y,lm.z])
                cv2.circle(image, (int(lm.x*w), int(lm.y*h)), 2, (255,0,0), thickness=1)
            landmarks = np.array(landmarks) # note : 478 x 3
            draw_landmarks(image, landmarks)

            cv2.imshow('image', image)
            cv2.waitKey(30)

            #graph描画
            ax.cla()
            ax.set_xlabel('X Label'); ax.axes.set_xlim3d(left  = 0.0, right=1.0)
            ax.set_ylabel('Y Label'); ax.axes.set_ylim3d(bottom= 0.0, top=1.0)
            ax.set_zlabel('Z Label'); ax.axes.set_zlim3d(bottom=-0.5, top=0.5)
            ax.scatter(landmarks[:,0], landmarks[:,1], landmarks[:,2])
            plt.draw()
            plt.pause(0.005)

            if bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000) : break


    cap.release()
    face_mesh.close()








if __name__ == "__main__":

    main()
