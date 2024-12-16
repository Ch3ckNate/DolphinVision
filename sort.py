import sys

import numpy as np
from typing import MutableSequence

import torch
from filterpy.kalman import KalmanFilter

from scipy.optimize import linear_sum_assignment

from utils.general import bbox_iou


def convert_bbox_to_z(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return np.array([bbox[0] + w / 2, bbox[1] + h / 2, w * h, w / h]).reshape((4, 1))


def convert_x_to_bbox(x):
    wd2 = np.sqrt(x[2] * x[3]) / 2
    hd2 = x[2] / (4 * wd2)
    return np.array([x[0] - wd2, x[1] - hd2, x[0] + wd2, x[1] + hd2]).reshape((1, 4))


class KalmanBoxTracker(object):
    def __init__(self, bbox, tag):
        self.kf = KalmanFilter(dim_x=7, dim_z=4)
        self.kf.F = np.array([[1, 0, 0, 0, 1, 0, 0], [0, 1, 0, 0, 0, 1, 0],
                              [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0],
                              [0, 0, 0, 0, 0, 0, 1]])
        self.kf.H = np.array([[1, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0]])
        self.kf.R[2:, 2:] *= 10 * 1.0
        self.kf.P[4:, 4:] *= 5000
        self.kf.P *= 10
        self.kf.Q[-1, -1] *= 0.5 * 1
        self.kf.Q[4:, 4:] *= 0.5
        self.kf.x[:4] = convert_bbox_to_z(bbox)

        self.tag = tag
        self.time_since_update = 0
        self.bbox_history = [bbox]
        self.hits = 0
        self.hit_streak = 0
        self.age = 0
        self.active = True
        self.prediction = -1
        self.association: KalmanBoxTracker | None = None
        self.last_predicted = False

        self.start_time = 0
        self.committed = False
        self.id = None
        self.full_track = None
        self.detections_length = 1

    def useful_history(self):
        return self.bbox_history[:self.detections_length]
        
    def update(self, bbox):
        self.time_since_update = 0
        self.hits += 1
        self.hit_streak += 1
        self.kf.update(convert_bbox_to_z(bbox))
        self.bbox_history[-1] = bbox
        self.detections_length = len(self.bbox_history)
        self.last_predicted = False

    def predict(self):
        if (self.kf.x[6] + self.kf.x[2]) <= 0:
            self.kf.x[6] *= 0.0
        self.kf.predict()

        prediction = convert_x_to_bbox(self.kf.x)[0]
        bbox = np.array([prediction[0], prediction[1], prediction[2], prediction[3], 0, -1])

        self.age += 1
        if self.time_since_update > 0:
            self.hit_streak = 0
        self.time_since_update += 1

        self.bbox_history.append(bbox)
        self.last_predicted = True

        return self.bbox_history[-1]

    def get_state(self):
        return np.concatenate((convert_x_to_bbox(self.kf.x), np.expand_dims(np.array([57]), 0)), axis=1)


def iou_cost_matrix(old, new):
    new = np.expand_dims(new, 1)
    old = np.expand_dims(old, 0)

    wh = (np.maximum(0, np.minimum(new[..., 2], old[..., 2]) - np.maximum(new[..., 0], old[..., 0])) *
          np.maximum(0, np.minimum(new[..., 3], old[..., 3]) - np.maximum(new[..., 1], old[..., 1])))

    return wh / ((new[..., 2] - new[..., 0]) * (new[..., 3] - new[..., 1]) +
                 (old[..., 2] - old[..., 0]) * (old[..., 3] - old[..., 1]) - wh)


def associate_detections_to_trackers(detections, trackers, iou_threshold):
    if len(trackers) == 0:
        return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

    iou_matrix = iou_cost_matrix(trackers, detections)

    if min(iou_matrix.shape) > 0:
        a = (iou_matrix > iou_threshold).astype(np.int32)
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
            matched_indices = np.stack(np.where(a), axis=1)
        else:
            matched_indices = np.array(linear_sum_assignment(-iou_matrix)).T
    else:
        matched_indices = np.empty(shape=(0, 2))
    
    unmatched_detections = []
    for d, det in enumerate(detections):
        if d not in matched_indices[:, 0]:
            unmatched_detections.append(d)
    
    unmatched_trackers = []
    for t, trk in enumerate(trackers):
        if t not in matched_indices[:, 1]:
            unmatched_trackers.append(t)
    
    # filter out matched with low IOU
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections.append(m[0])
            unmatched_trackers.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
    
    if len(matches) == 0:
        matches = np.empty((0, 2), dtype=int)
    else:
        matches = np.concatenate(matches, axis=0)
        
    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


def calculate_iou(boxA, boxB):
    # Get coordinates of intersection rectangle
    xA = np.maximum(boxA[:, 0], boxB[:, 0])
    yA = np.maximum(boxA[:, 1], boxB[:, 1])
    xB = np.minimum(boxA[:, 2], boxB[:, 2])
    yB = np.minimum(boxA[:, 3], boxB[:, 3])

    # Calculate intersection area
    interArea = np.maximum(0, xB - xA) * np.maximum(0, yB - yA)

    # Calculate area of each box
    boxAArea = (boxA[:, 2] - boxA[:, 0]) * (boxA[:, 3] - boxA[:, 1])
    boxBArea = (boxB[:, 2] - boxB[:, 0]) * (boxB[:, 3] - boxB[:, 1])

    # Calculate IoU
    return interArea / (boxAArea + boxBArea - interArea)


def mass_iou(bboxes0: np.ndarray, bboxes1: np.ndarray):

    ret = np.zeros((len(bboxes0), 1))

    for i in range(len(ret)):
        ret[i] = bbox_iou(torch.tensor(bboxes0[i]), torch.tensor(bboxes1[i]), x1y1x2y2=True, CIoU=False)
    return ret


class Sort(object):
    def __init__(self, max_age, min_hits, iou_threshold, tag):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: MutableSequence[KalmanBoxTracker] = []
        self.frame_count = 0
        self.next_id = 0
        self.tag = tag

    def get_hit_trackers(self):
        ret = []
        for t in self.tracks:
            if t.hits >= self.min_hits:
                ret.append(t)
        return ret

    def reset(self):
        self.frame_count = 0
        self.tracks = []
        self.next_id = 0
        
    def update(self, dets):
        self.frame_count += 1
        
        # Get predicted locations from existing trackers
        trks = np.zeros((len(self.tracks), 6))
        for t, trk in enumerate(trks):
            pos = self.tracks[t].predict()
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0, 0]

        matched, unmatched_dets, _ = associate_detections_to_trackers(dets, trks, self.iou_threshold)
        
        # Update matched trackers with assigned detections
        for m in matched:
            self.tracks[m[1]].update(dets[m[0], :])
            
        # Create and initialize new trackers for unmatched detections
        for i in unmatched_dets:
            self.tracks.append(KalmanBoxTracker(dets[i, :], self.tag))
            self.tracks[-1].id = self.next_id
            self.tracks[-1].start_time = self.frame_count - 1
            self.next_id += 1

        ret = []
        i = len(self.tracks)
        for trk in reversed(self.tracks):
            if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
                c = (trk.get_state()[0], [trk.id + 1])
                ret.append(np.concatenate(c).reshape(1, -1))
            i -= 1
            if trk.time_since_update > self.max_age:
                trk.active = False
                self.tracks.pop(i)
        if len(ret) > 0:
            return np.concatenate(ret)
        return np.empty((0, 6))
