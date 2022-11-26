import cv2
import re
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
import ctypes

mp_face_mesh = mp.solutions.face_mesh

def normalize(v):
    distance = np.linalg.norm(v)
    if distance != 0:
        v = v / distance
    return v


# point : normalized 2d point
# dir   : found direction (normalized)
def draw_line(image, width, height, pt, dir, color):
    p1 = (int(width*pt[0]), int(height*pt[1]))
    p2 = (p1[0] + int(100*dir[0]), p1[1]+int(100*dir[1]))
    cv2.line(image, p1, p2, color, 2)


def calc_face_direction1(frame, landmarks, vis_marker = False):
    idx = [152, 33, 263] #4鼻 152顎 33右目 263左目
    x0, x1, x2 = landmarks[idx[0]], landmarks[idx[1]], landmarks[idx[2]],
    d = normalize(np.cross(x2 - x0, x1 - x0))

    h, w, _ = frame.shape
    nose = landmarks[4]
    draw_line(frame, w,h, nose, d, (0, 0, 255))
    print("aa", d)
    if vis_marker:
        cv2.circle(frame, (int(x0[0]*w), int(x0[1]*h)), 2, (0,0,255), thickness=2)
        cv2.circle(frame, (int(x1[0]*w), int(x1[1]*h)), 2, (0,0,255), thickness=2)
        cv2.circle(frame, (int(x2[0]*w), int(x2[1]*h)), 2, (0,0,255), thickness=2)
    return d


def calc_face_direction2(frame, landmarks, vis_marker = False):
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
    draw_line(frame, w, h, nose, dir, (255,128,255))
    print("bb", dir)

    return dir


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
        #dir1 = calc_face_direction1(image, landmarks)
        dir2 = calc_face_direction2(image, landmarks)

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
