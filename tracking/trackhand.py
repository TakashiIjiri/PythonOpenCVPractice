
import sys
from pathlib import Path

import cv2
import numpy as np

from rtmlib import Hand, draw_skeleton
import mediapipe as mp
from ultralytics import YOLO



#
# Hand Tracker (RTMlib)
#

KPT_THRESHOLD = 0.1

class HandTrackRtmlib:

    def __init__(self, backend="onnxruntime", device="cpu"):
        self.hand = Hand(
            backend = backend,
            device = device,
            to_openpose = False,
        )

    def track(self, frame):
        keypoints, scores = self.hand(frame)
        return keypoints, scores

    def draw(self, frame, keypoints, scores):
        return draw_skeleton(
            frame.copy(),
            keypoints,
            scores,
            openpose_skeleton=False,
            kpt_thr=KPT_THRESHOLD,
        )

    def track_and_draw(self, frame):
        keypoints, scores = self.track(frame)
        result = self.draw(frame, keypoints, scores)
        return result




    
#
# Hand Tracker (MediaPipe)
#
MM_HANDDETECTION_CONFIDENCE = 0.1
MM_HANDPRESENCE_FONFIDENCE = 0.1
class HandTrackMediapipe:

    def __init__(self):

        model_path = "hand_landmarker.task"
        model_path = Path(__file__).resolve().parent / model_path

        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options = mp.tasks.BaseOptions( model_asset_path=str(model_path)),
            running_mode = mp.tasks.vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence = MM_HANDDETECTION_CONFIDENCE,
            min_hand_presence_confidence = MM_HANDPRESENCE_FONFIDENCE,
        )

        self.tracker = mp.tasks.vision.HandLandmarker.create_from_options(options)


    def track(self, frame):
        height, width = frame.shape[:2]
        mp_image = mp.Image( 
            image_format = mp.ImageFormat.SRGB, 
            data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) )
        result = self.tracker.detect(mp_image)

        keypoints = []
        for landmarks in result.hand_landmarks:
            points = [ [lm.x * width, lm.y * height] for lm in landmarks ]
            keypoints.append(points)

        return np.array(keypoints, dtype=np.float32).reshape(-1, 21, 2)


    def draw(self, frame, keypoints):

        result = frame.copy()

        connections = [
            (0,  1), (1,  2), ( 2,  3), (3,  4),
            (0,  5), (5,  6), ( 6,  7), (7,  8),
            (5,  9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13,14), (14, 15), (15, 16),
            (13,17), (17,18), (18, 19), (19, 20), (0, 17)]

        for hand_points in keypoints:
            for i, j in connections:
                p1 = tuple(hand_points[i].astype(int))
                p2 = tuple(hand_points[j].astype(int))
                cv2.line(result, p1, p2, (0, 255, 0), 2)

            for point in hand_points:
                p = tuple(point.astype(int))
                cv2.circle(result, p, 4, (0, 0, 255), -1)

        return result


    def track_and_draw(self, frame):
        keypoints = self.track(frame)
        return self.draw(frame, keypoints)


    def close(self):
        self.tracker.close()



        

#
# Hand Tracker (YOLO)
#

YOLO_CONFIDENCE_THRESHOLD = 0.1

class HandTrackYolo:

    def __init__(self):
        model_path = "yolo26_hand_pose.pt"
        model_path = Path(__file__).resolve().parent / model_path
        self.tracker = YOLO(str(model_path))


    def track(self, frame):

        result = self.tracker.predict( 
            frame, 
            conf = YOLO_CONFIDENCE_THRESHOLD,
            verbose = False
        )[0]

        if result.keypoints is None:
            return (np.empty((0, 21, 2), dtype=np.float32),
                    np.empty((0, 21)   , dtype=np.float32))

        keypoints = result.keypoints.xy.cpu().numpy()

        if result.keypoints.conf is not None:
            scores = result.keypoints.conf.cpu().numpy()
        else:
            scores = np.ones((len(keypoints), 21), dtype=np.float32)

        return keypoints, scores


    def draw(self, frame, keypoints, scores):

        result = frame.copy()

        connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),
            (0, 5), (5, 6), (6, 7), (7, 8),
            (5, 9), (9, 10), (10, 11), (11, 12),
            (9, 13), (13, 14), (14, 15), (15, 16),
            (13, 17), (17, 18), (18, 19), (19, 20),
            (0, 17)
        ]

        for hand_points, hand_scores in zip(keypoints, scores):

            for i, j in connections:
                p1 = tuple(hand_points[i].astype(int))
                p2 = tuple(hand_points[j].astype(int))
                cv2.line(result, p1, p2, (0, 255, 0), 2)

            for point in hand_points:
                p = tuple(point.astype(int))
                cv2.circle(result, p, 4, (0, 0, 255), -1)

        return result


    def track_and_draw(self, frame):
        keypoints, scores = self.track(frame)
        return self.draw(frame, keypoints, scores)

        

#
# Camera mode
#
def camera_mode():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera")

    tracker_rtm = HandTrackRtmlib()
    tracker_mm  = HandTrackMediapipe()
    tracker_yolo = HandTrackYolo()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img1 = tracker_rtm.track_and_draw(frame)
        img2 = tracker_mm .track_and_draw(frame)
        img3 = tracker_yolo.track_and_draw(frame)

        cv2.imshow("rtmlib"   , img1)
        cv2.imshow("mediapipe", img2)
        cv2.imshow("yolo"     , img3)

        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


#
# Video mode
#
def video_mode(video_path):

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps    = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    path = Path(video_path)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out_video_rtm  = path.with_name(path.stem + "_rtm.mp4")
    out_video_mm   = path.with_name(path.stem + "_mm.mp4")
    out_video_yolo = path.with_name(path.stem + "_yolo.mp4")
    writer_rtm    = cv2.VideoWriter(str(out_video_rtm), fourcc, fps, (width, height))
    writer_mm     = cv2.VideoWriter(str(out_video_mm) , fourcc, fps, (width, height))
    writer_yolo   = cv2.VideoWriter(str(out_video_yolo), fourcc, fps, (width, height))

    tracker_rtm = HandTrackRtmlib()
    tracker_mm  = HandTrackMediapipe()
    tracker_yolo = HandTrackYolo()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        writer_rtm .write(tracker_rtm .track_and_draw(frame))
        writer_mm  .write(tracker_mm  .track_and_draw(frame))
        writer_yolo.write(tracker_yolo.track_and_draw(frame))

        if frame_idx % 100 == 0:
            print(f"Processing frame {frame_idx}")

        frame_idx += 1

    cap.release()
    writer_rtm .release()
    writer_mm  .release()
    writer_yolo.release()

#
# Main
#
def main():
    if len(sys.argv) > 1:
        video_mode(sys.argv[1])
    else:
        camera_mode()


if __name__ == "__main__":
    main()
