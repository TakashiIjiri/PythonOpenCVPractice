import cv2
import re
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh
mp_drawing_styles = mp.solutions.drawing_styles


# mp_drawingを利用してface_landmarksを可視化
def draw_landmarks_mputil(image, results):
    if not results.multi_face_landmarks:
        return
    for face_landmarks in results.multi_face_landmarks:
        #mesh
        mp_drawing.draw_landmarks( image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        #輪郭と目と口と眉毛
        mp_drawing.draw_landmarks( image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_CONTOURS,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())
        # 黒目
        mp_drawing.draw_landmarks( image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_IRISES,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style())


# 自作関数
# landmarks は 478 x 3 の二次元 numpy配列
# index と 場所の関係は「https://www.google.com/search?q=mediapipe+facemesh+index+and+position」等で調べる
def draw_landmarks(image, landmarks):
    FACE_TRBL = [ 10,234,152, 454] # 顔の上右下左 の頂点
    EYE_LEFT  = [469,470,471, 472] # https://note.com/kitazaki/n/nbf4fd1e84271
    EYE_RIGHT = [476,475,474, 477]
    EYE_LR    = [ 468, 473]
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

        if results.multi_face_landmarks is not None:
            # landmarkを描く (配布されている関数を利用)
            draw_landmarks_mputil(image, results)

            # landmarkを描く (自作関数)
            face_landmarks = results.multi_face_landmarks[0].landmark # 最初の顔のみ
            landmarks = []
            for lm in face_landmarks:
                landmarks.append([lm.x,lm.y,lm.z])
            landmarks = np.array(landmarks) # note : 478 x 3
            draw_landmarks(image, landmarks)

            cv2.imshow('image', image)
            if cv2.waitKey(30) & 0xFF == 27:
              break



    cap.release()
    face_mesh.close()
    """

    """
















if __name__ == "__main__":
    main()
