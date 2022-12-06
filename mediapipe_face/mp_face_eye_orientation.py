import cv2
import re
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import ctypes
import math

mp_face_mesh = mp.solutions.face_mesh

def normalize(v):
    distance = np.linalg.norm(v)
    if distance != 0:
        v = v / distance
    return v


def RotX(angle):
    return np.array([[1, 0, 0],
                     [0, np.cos(angle), -np.sin(angle)],
                     [0, np.sin(angle),  np.cos(angle)]])

def RotY(angle):
    return np.array([[np.cos(angle), 0, np.sin(angle)],
                     [0, 1, 0],
                     [-np.sin(angle), 0, np.cos(angle)]])


# point : normalized 2d point
# dir   : found direction (normalized)
def draw_line(image, width, height, pt, dir, color):
    p1 = (int(width*pt[0]), int(height*pt[1]))
    p2 = (p1[0] + int(100*dir[0]), p1[1]+int(100*dir[1]))
    cv2.line(image, p1, p2, color, 2)


def calc_face_direction(frame, landmarks, vis_marker = False):
    #7個の三角形の外積の平均を顔の向きとする
    #indexについては https://zenn.dev/dami/articles/a1fdbb34e0be93
    triangles = [[152, 33, 263],
                 [  6,152, 136], [6,365,152],
                 [  6,136,  21], [6,251,365],
                 [  6, 21,  10], [6, 10,251]]
    Cs = [(255,0,0),(0,255,0),(0,0,255), (255,255,0),(0,255,255),(255,0,255), (255,255,255)]

    h, w, _ = frame.shape
    nose = landmarks[4]

    dir = np.array([0.0, 0.0, 0.0])
    for i, idx in enumerate(triangles):
        x0, x1, x2 = landmarks[idx[0]], landmarks[idx[1]], landmarks[idx[2]],
        d = normalize(np.cross(x2 - x0, x1 - x0))
        dir = dir + d

        if vis_marker :
            draw_line(frame, w,h, nose, d, Cs[i])
            cv2.circle(frame, (int(x0[0]*w), int(x0[1]*h)), 2, (0,0,255), thickness=2)
            cv2.circle(frame, (int(x1[0]*w), int(x1[1]*h)), 2, (0,0,255), thickness=2)
            cv2.circle(frame, (int(x2[0]*w), int(x2[1]*h)), 2, (0,0,255), thickness=2)

    dir = normalize(dir)
    yangle = - math.asin(dir[0]) #回転角度(y軸) Ry
    xangle =   math.asin(dir[1]) #回転角度(x軸) Rx   - Ry Rx (0,0,-1)
    # note dir = np.dot(RotY(yangle), np.dot(RotX(xangle), np.array([0,0,-1]) ))
    return xangle, yangle


# ---------------------------------------
#          27                257
#    130  468  243     463   473   359
#         23                 253
def calc_iris_orientation(image, landmarks, is_left = True, vis_marker=False):
    if is_left:
        idx = [473, 463, 257, 359, 253]  # iris and four points (外側)
        #idx = [473, 263, 362, 374, 386] # iris and four points (内側、瞼の影響受ける)
    else :
        idx = [468, 243, 27, 130, 23] # iris and four points

    h, w, _ = image.shape
    iris = [landmarks[idx[0]][0]*w, landmarks[idx[0]][1]*h]
    p1   = [landmarks[idx[1]][0]*w, landmarks[idx[1]][1]*h]
    p2   = [landmarks[idx[2]][0]*w, landmarks[idx[2]][1]*h]
    p3   = [landmarks[idx[3]][0]*w, landmarks[idx[3]][1]*h]
    p4   = [landmarks[idx[4]][0]*w, landmarks[idx[4]][1]*h]
    xmin, xmax = min(p1[0], p2[0], p3[0], p4[0]), max(p1[0], p2[0], p3[0], p4[0])
    ymin, ymax = min(p1[1], p2[1], p3[1], p4[1]), max(p1[1], p2[1], p3[1], p4[1])
    #cx = (xmin + xmax) * 0.5
    #cy = (ymin + ymax) * 0.5
    cx = (p2[0] + p4[0]) * 0.5 # 瞳孔の中心位置（もう少し正しく指定すべき）
    cy = (p1[1] + p3[1]) * 0.5
    x = (iris[0] - cx) / (xmax - xmin) * 2 # 瞳孔の中心空のずれ [-1.1] に正規化 :
    y = (iris[1] - cy) / (ymax - ymin) * 2
    #x = x / np.sqrt(2) # [-1,1]を[-45度, 45度]に対応付けるなら sqrt(2)で割る
    #y = y / np.sqrt(2)
    x = max(-1, min(1, x)) #clamp
    y = max(-1, min(1, y))

    yangle = - math.asin(x) #回転角度(y軸)
    xangle =   math.asin(y) #回転角度(x軸)

    if vis_marker :
        cv2.circle(image, (int(iris[0]), int(iris[1])), 2, (0,255,0), thickness=2)
        cv2.circle(image, (int(p1[0]), int(p1[1])), 2, (0,0,255), thickness=2)
        cv2.circle(image, (int(p2[0]), int(p2[1])), 2, (0,0,255), thickness=2)
        cv2.circle(image, (int(p3[0]), int(p3[1])), 2, (0,0,255), thickness=2)
        cv2.circle(image, (int(p4[0]), int(p4[1])), 2, (0,0,255), thickness=2)
        cv2.circle(image, (int(cx), int(cy)), 2, (255,255,255), thickness=2)
        """
        Rx = RotX(xangle)
        Ry = RotY(yangle)
        d = np.dot(Ry, np.dot(Rx, np.array([0.,0.,-1.])))
        cv2.line(image, (int(iris[0]), int(iris[1])),
                        (int(iris[0]+d[0]*100), int(iris[1]+d[1]*100)), (255,0,0), 2)
        """
    return xangle, yangle


def estimate_face_eye_orientation(image, landmarks, vis=True):
    f_xangle, f_yangle = calc_face_direction(image, landmarks)
    l_xangle, l_yangle = calc_iris_orientation(image, landmarks, True , True)
    r_xangle, r_yangle = calc_iris_orientation(image, landmarks, False, True)

    R_face  = np.dot(RotY(f_yangle), RotX(f_xangle))
    R_left  = np.dot(RotY(l_yangle), RotX(l_xangle))
    R_right = np.dot(RotY(r_yangle), RotX(r_xangle))
    dir_face  = np.dot(R_face, np.array([0,0,-1]))
    dir_left  = np.dot(R_face, np.dot(R_left , np.array([0,0,-1])))
    dir_right = np.dot(R_face, np.dot(R_right, np.array([0,0,-1])))

    h, w, _ = image.shape
    nose, right_iris, left_iris = landmarks[4], landmarks[468], landmarks[473]
    draw_line(image, w, h, nose      , dir_face , (255,128,255))
    draw_line(image, w, h, left_iris , dir_left , (255,  0,  0))
    draw_line(image, w, h, right_iris, dir_right, (   0, 0,255))



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

        #RGB画像を利用して landmarkを推論
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = face_mesh.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if results.multi_face_landmarks is None:
            continue

        # landmarkを描く (自作関数)
        face_landmarks = results.multi_face_landmarks[0].landmark # 最初の顔のみ
        landmarks = []
        for lm in face_landmarks:
            landmarks.append([lm.x,lm.y,lm.z])
        landmarks = np.array(landmarks) # note : 478 x 3

        estimate_face_eye_orientation(image, landmarks, vis=True)

        ax.cla()
        ax.set_xlabel('X Label'); ax.axes.set_xlim3d(left=0.0, right=1.0)
        ax.set_ylabel('Y Label'); ax.axes.set_ylim3d(bottom=0.0, top=1.0)
        ax.set_zlabel('Z Label'); ax.axes.set_zlim3d(bottom=-0.5, top=0.5)
        ax.scatter(landmarks[:,0], landmarks[:,1], landmarks[:,2])
        plt.draw()
        plt.pause(0.001)

        cv2.imshow('image', image)
        cv2.waitKey(50)

        if bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000):break


    cap.release()
    face_mesh.close()








if __name__ == "__main__":

    main()
