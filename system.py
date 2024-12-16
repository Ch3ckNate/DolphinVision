import argparse
import cv2
import os
import time

import munkres
import numpy as np
import torch.backends.cudnn as cudnn

from models.experimental import attempt_load
from utils.general import nms, scale_coords_np, bbox_iou
from munkres import Munkres
from sort import *

FPS_SCREEN = 30
FPS_UPDATE = 30

MAX_NONE_TIME = 1.00

CLS_COLS = [(0, 0, 255), (0, 255, 0), (0, 255, 255), (255, 0, 0), (255, 0, 255), (255, 255, 0), (0, 0, 0)]

binary_mode = False


def letterbox(old_shape, new_shape, stride):
    # Resize and pad image while meeting stride-multiple constraints

    r = min(min(new_shape[0] / old_shape[0], new_shape[1] / old_shape[1]), 1.0)

    new_unpad = int(round(old_shape[1] * r)), int(round(old_shape[0] * r))

    padw, padh = np.mod(new_shape[1] - new_unpad[0], stride) / 2, np.mod(new_shape[0] - new_unpad[1], stride) / 2

    return new_unpad, (padw, padh)


is_paused = not True
force_next = False


def update_window(image):
    global is_paused, binary_mode, force_next

    cv2.imshow("GUI", image)
    key_pressed = cv2.waitKeyEx(1)

    if key_pressed == ord('d'):
        arguments.debug = not arguments.debug
    elif key_pressed == ord('f'):
        cv2.setWindowProperty("GUI", cv2.WND_PROP_FULLSCREEN, 1.0 - cv2.getWindowProperty("GUI", cv2.WND_PROP_FULLSCREEN))
    elif key_pressed == ord('q'):
        cv2.destroyWindow("GUI")
    elif key_pressed == ord(' '):
        is_paused = not is_paused
    elif key_pressed == ord('b'):
        binary_mode = not binary_mode
    elif key_pressed == ord('s'):
        cv2.imwrite("frames/" + str(frame_count) + ".png", buffer_video)
    elif key_pressed == ord('r'):
        if video_record.isOpened():
            video_record.release()
        else:
            video_record.open(os.path.join("recordings", str(time.time() // 1) + ".mp4"), cv2.VideoWriter_fourcc(*'mp4v'), FPS_SCREEN, (1920, 1080))
    elif key_pressed == ord('z'):
        force_next = True


def draw_translucent_circle(img, x, y, r, color, alpha):
    tmp = img.copy()
    cv2.circle(tmp, (x, y), r, color, -1)
    cv2.addWeighted(img, 1 - alpha, tmp, alpha, 0.0, img)

def draw_translucent_rectangle(img, x1, y1, x2, y2, color, alpha):
    src1 = img[y1:y2, x1:x2]
    src2 = np.full(src1.shape, color, dtype=np.uint8)
    cv2.addWeighted(src1, 1 - alpha, src2, alpha, 0.0, src1)


def draw_body_and_face_detection(img, bx1, by1, bx2, by2, fx1, fy1, fx2, fy2, color, alpha):
    draw_translucent_rectangle(img, bx1, by1, bx2, by2, color, alpha)
    draw_translucent_rectangle(img, fx1, fy1, fx2, fy2, color, alpha)


def draw_detection(img, x1, y1, x2, y2, color):
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)


def centroid(bbox):
    return int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2)


def draw_track(img, track, color=None):
    for i in range(len(track.bbox_history) - 1):
        cv2.line(img, centroid(track.bbox_history[i]), centroid(track.bbox_history[i + 1]), CLS_COLS[int(track.detclass)] if color is None else color, thickness=1)


argument_parser = argparse.ArgumentParser()
argument_parser.add_argument('-d', '--debug', action='store_true')
argument_parser.add_argument('-v', '--video', type=str, default='0')
argument_parser.add_argument('-o', '--output', type=str, default='')
arguments = argument_parser.parse_args()

cv2.namedWindow("GUI", cv2.WINDOW_NORMAL)
cv2.resizeWindow("GUI", 1920, 1080)

video_is_camera = arguments.video.isnumeric()

    
video_export = cv2.VideoWriter()
video_record = cv2.VideoWriter()
video_stream = cv2.VideoCapture(int(arguments.video) if video_is_camera else arguments.video, cv2.CAP_MSMF, (
    cv2.CAP_PROP_FRAME_WIDTH, 1280, cv2.CAP_PROP_FRAME_HEIGHT, 720, cv2.CAP_PROP_FPS, 30) if video_is_camera else None)

# video_stream.set(cv2.CAP_PROP_POS_MSEC, 120000)

if arguments.output:
    outfile = arguments.output
    print ( "output file is : ", arguments.output)
    video_record.open(os.path.join("recordings", outfile), cv2.VideoWriter_fourcc(*'mp4v'), FPS_SCREEN, (1920, 1080))
else:
    print ( "Sending output to screen instead")
    
    
# Static allocations to be reused for efficiency.
buffer_output = np.empty((1080, 1920, 3), dtype=np.uint8)
buffer_input_face = np.full((768, 1280, 3), 114, dtype=np.uint8)
buffer_input_body = np.full((384, 640, 3), 114, dtype=np.uint8)
buffer_video = np.empty((720, 1280, 3) if video_is_camera else (1080, 1920, 3), dtype=np.uint8)

cudnn.benchmark = True         # Seek optimal algorithms for fixed size inputs
torch.set_grad_enabled(False)  # Gradient computations not needed for inference

device = torch.device('cuda:0')

model_face = attempt_load('bestface.pt', device).half()
model_body = attempt_load('bestbod3.pt', device).half()

face_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.30, tag=0)
body_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.30, tag=1)


def image_tensor(img):
    return torch.as_tensor(img[None, ..., ::-1].transpose(0, 3, 1, 2).copy(), dtype=torch.float16, device=device) / 255.0


def intersection(track0: KalmanBoxTracker, track1: KalmanBoxTracker):
    start, end = max(track0.start_time, track1.start_time), min(track0.start_time + len(track0.bbox_history), track1.start_time + len(track1.bbox_history))
    if end >= start:
        return start, end
    return None


def range_intersection(a0, a1, b0, b1):
    start, end = max(a0, b0), min(a1, b1)
    if end >= start:
        return start, end
    return None


def get_intersections(track0: KalmanBoxTracker, track1: KalmanBoxTracker):
    start, end = max(track0.start_time, track1.start_time), min(track0.start_time + len(track0.bbox_history), track1.start_time + len(track1.bbox_history))
    if end > start:
        return np.array(track0.bbox_history[start - track0.start_time:end - track0.start_time]), np.array(track1.bbox_history[start - track1.start_time:end - track1.start_time])
    return None


def create_2d_array(x, y):
    return [[0 for _ in range(y)] for _ in range(x)]


model_face(image_tensor(buffer_input_face))  # Warm up model.
model_body(image_tensor(buffer_input_body))  # Unsure if useful or needed.

frame_count = 0

all_tracks = []

while cv2.getWindowProperty("GUI", cv2.WND_PROP_VISIBLE) and video_stream.isOpened():

    if is_paused and not force_next:
        force_next = False
        update_window(buffer_output)
        continue

    success, _ = video_stream.read(buffer_video)

    if not success:
        break

    time_curr = time.time()

    # === UPDATE ===

    cv2.resize(buffer_video, (1280, 720), buffer_input_face[24:744])
    cv2.resize(buffer_video, (640, 360), buffer_input_body[12:372])

    face_detections = nms(model_face(image_tensor(buffer_input_face))[0][0], 0.8, 0.8)
    body_detections = nms(model_body(image_tensor(buffer_input_body))[0][0], 0.8, 0.8)

    face_detections = scale_coords_np(face_detections, buffer_input_face.shape, buffer_video.shape).round()
    body_detections = scale_coords_np(body_detections, buffer_input_body.shape, buffer_video.shape).round()

    face_tracker.update(face_detections)
    body_tracker.update(body_detections)

    for f in face_tracker.tracks:
        if f not in all_tracks:
            all_tracks.append(f)

    for b in body_tracker.tracks:
        if b not in all_tracks:
            all_tracks.append(b)

    # 3. Associate body tracks with face tracks.

    cost_matrix = create_2d_array(len(body_tracker.tracks), len(face_tracker.tracks))

    for i, b in enumerate(body_tracker.tracks):
        for j, f in enumerate(face_tracker.tracks):
            inter = get_intersections(b, f)
            if inter is not None:
                cost_matrix[i][j] = -np.average(calculate_iou(inter[0], inter[1]))
            else:
                cost_matrix[i][j] = munkres.DISALLOWED

    if len(body_tracker.tracks) > 0 and len(face_tracker.tracks) > 0:
        matched_indices = Munkres().compute(cost_matrix)

        # Bodies matched to a face
        for m in matched_indices:
            body_tracker.tracks[m[0]].association = face_tracker.tracks[m[1]]
            face_tracker.tracks[m[1]].association = body_tracker.tracks[m[0]]

        for b in body_tracker.tracks:
            total_counts = [0, 0, 0, 0, 0]
            for f in all_tracks:
                if f.association is b:
                    values, counts = np.unique(np.array(f.bbox_history)[:f.detections_length, 5], return_counts=True)
                    for v, c in zip(values, counts):
                        total_counts[int(v) + 1] += c if v != -1 else (c / 4)
            b.prediction = np.argmax(total_counts) - 1
            if b.association is not None:
                b.association.prediction = b.prediction

    frame_count += 1

    # === DISPLAY ===

    cv2.resize(buffer_video, (1920, 1080), buffer_output)

    if arguments.debug:
        for t in face_tracker.tracks:
            draw_detection(buffer_output, *t.bbox_history[-1][0:4], CLS_COLS[int(t.bbox_history[-1][5])])
            cv2.putText(buffer_output, str(t.id), (int(t.bbox_history[-1][0] + 5), int(t.bbox_history[-1][1] + 22)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

        for t in body_tracker.tracks:
            draw_detection(buffer_output, *t.bbox_history[-1][0:4], CLS_COLS[t.prediction])
            cv2.putText(buffer_output, str(t.id), (int(t.bbox_history[-1][0] + 5), int(t.bbox_history[-1][1] + 22)), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255))

    skipped = 0

    points = [None] * 4
    rs = [0] * 4

    for f in face_tracker.tracks:
        if f.association is not None and f.hits > 5:
            if f.association.prediction != -1:
                points[f.association.prediction] = centroid(f.bbox_history[-1])
                rs[f.association.prediction] = max(f.bbox_history[-1][2] - f.bbox_history[-1][0], f.bbox_history[-1][3] - f.bbox_history[-1][1])
        else:
            total_counts = [0, 0, 0, 0, 0]
            values, counts = np.unique(np.array(f.bbox_history)[:f.detections_length, 5], return_counts=True)
            for v, c in zip(values, counts):
                total_counts[int(v) + 1] += c
            prediction = np.argmax(total_counts) - 1
            if prediction >= 0 and points[prediction] is None and f.hits > 15:
                points[prediction] = centroid(f.bbox_history[-1])
                rs[prediction] = max(f.bbox_history[-1][2] - f.bbox_history[-1][0], f.bbox_history[-1][3] - f.bbox_history[-1][1])
        #    points[f.prediction] = centroid(f.bbox_history[-1])


    for b in body_tracker.tracks:
        if b.prediction != -1 and points[b.prediction] is None:
            cx, cy = ((b.bbox_history[-1][0] + b.bbox_history[-1][2]) / 2), b.bbox_history[-1][1]

            (w1, h1), _ = cv2.getTextSize(model_face.names[b.prediction].upper(), cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)
            (w2, h2), _ = cv2.getTextSize(model_face.names[b.prediction].upper(), cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)
            cv2.putText(buffer_output, model_face.names[b.prediction], (int(cx - w1 / 2), int(cy + h1 / 2)), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 255, 255), 5)
            cv2.putText(buffer_output, model_face.names[b.prediction], (int(cx - w2 / 2), int(cy + h2 / 2)), cv2.FONT_HERSHEY_DUPLEX, 3.0, CLS_COLS[b.prediction], 2)



    for i, p in enumerate(points):
        if p is not None:
            draw_translucent_circle(buffer_output, int(p[0]), int(p[1]), int(rs[i] / 2), CLS_COLS[i], 0.15)
            cv2.circle(buffer_output, (int(p[0]), int(p[1])), int(rs[i] / 2), CLS_COLS[i], 2)
            (w1, h1), _ = cv2.getTextSize(model_face.names[i].upper(), cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)
            (w2, h2), _ = cv2.getTextSize(model_face.names[i].upper(), cv2.FONT_HERSHEY_DUPLEX, 3.0, 4)
            cv2.putText(buffer_output, model_face.names[i], (int(p[0] - w1 / 2), int(p[1] + h1 / 2) - int(rs[i] / 2)), cv2.FONT_HERSHEY_DUPLEX, 3.0, (255, 255, 255), 5)
            cv2.putText(buffer_output, model_face.names[i], (int(p[0] - w2 / 2), int(p[1] + h2 / 2) - int(rs[i] / 2)), cv2.FONT_HERSHEY_DUPLEX, 3.0, CLS_COLS[i], 2)

    if arguments.debug:
        for i, t in enumerate(all_tracks):
            y = (i - skipped) * 22 - 1
            x = t.start_time

            #if t.detections_length < 10:
            #    skipped += 1
            #    continue

            if t.tag == 0:
                for j, d in enumerate(t.bbox_history):
                    cv2.line(buffer_output, (x + j, y), (x + j, y + 21), CLS_COLS[int(d[5])], 1)
                cv2.putText(buffer_output, str(t.id) + ("" if t.association is None else "->" + str(t.association.id)), (x + 5, y + 16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (32, 128, 256), 2)
            else:
                draw_translucent_rectangle(buffer_output, x, y, x + len(t.bbox_history), y + 21, CLS_COLS[t.prediction], 0.25)
                cv2.putText(buffer_output, str(t.id) + ("" if t.association is None else "->" + str(t.association.id)), (x + 5, y + 16), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 255, 255), 2)
            if not t.active:
                cv2.line(buffer_output, (x, y + 10), (x + len(t.bbox_history), y + 10), (128, 128, 128), 1)
                cv2.line(buffer_output, (x, y + 11), (x + len(t.bbox_history), y + 11), (128, 128, 128), 1)
                cv2.line(buffer_output, (x, y + 12), (x + len(t.bbox_history), y + 12), (128, 128, 128), 1)

    if video_record.isOpened():
        video_record.write(buffer_output)
        cv2.circle(buffer_output, (132, 132), 32, (0, 0, 255), -1)

    force_next = False
    update_window(buffer_output)
    pass

cv2.destroyAllWindows()

video_stream.release()

if video_record.isOpened():
    video_record.release()