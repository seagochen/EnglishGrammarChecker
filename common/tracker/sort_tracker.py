from typing import List

import numpy as np
from scipy.optimize import linear_sum_assignment

from common.tracker.tracker_manager import TrackerManager
from common.yolo.yolo_results import Yolo



class SortTracker:
    def __init__(self, max_age=5, min_hits=1, iou_threshold=0.3, max_objects=1000):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.frame_count = 0  # 帧计数器
        self.tracker_manager = TrackerManager(max_age, max_objects)  # 用 TrackerManager 管理追踪器

    def update(self, detections: List[Yolo]):
        self.frame_count += 1
        trks = self.tracker_manager.get_tracker_states()  # 获取当前所有追踪器的状态

        # 将检测结果与追踪器进行匹配
        matched, unmatched_dets, unmatched_trks = self._associate_detections_to_trackers(detections, trks)

        detection_map = {}  # 追踪器 ID 与检测结果索引的映射
        for m in matched:
            self.tracker_manager.trackers[m[1]].update(detections[m[0]][:4])  # 更新匹配上的追踪器
            detection_map[self.tracker_manager.trackers[m[1]].id] = m[0]

        for i in unmatched_dets:
            tracker = self.tracker_manager.add_tracker(detections[i][:4])  # 添加新的追踪器
            detection_map[tracker.id] = i

        self.tracker_manager.update_trackers()  # 更新追踪器状态

        # 构造返回的结果
        ret = []
        for tracker in self.tracker_manager.trackers:
            bbox = tracker.get_state()
            detection_idx = detection_map.get(tracker.id)
            if detection_idx is not None and detection_idx < len(detections):
                kpts_combined = detections[detection_idx][4:]  # 获取检测的关键点信息
                ret.append([tracker.id] + bbox + kpts_combined.tolist())
        return np.array(ret)

    def _associate_detections_to_trackers(self, detections, trackers):
        if len(trackers) == 0:
            return np.empty((0, 2), dtype=int), np.arange(len(detections)), np.empty((0, 5), dtype=int)

        iou_matrix = np.zeros((len(detections), len(trackers)), dtype=np.float32)
        for d, det in enumerate(detections):
            for t, trk in enumerate(trackers):
                iou_matrix[d, t] = self._iou(det[:4], trk[:4])

        matched_indices = linear_sum_assignment(-iou_matrix)
        matched_indices = np.array(list(zip(*matched_indices)))

        if matched_indices.size == 0:
            matched_indices = np.empty((0, 2), dtype=int)

        unmatched_detections = []
        for d in range(len(detections)):
            if d not in matched_indices[:, 0]:
                unmatched_detections.append(d)

        unmatched_trackers = []
        for t in range(len(trackers)):
            if t not in matched_indices[:, 1]:
                unmatched_trackers.append(t)

        matches = []
        for m in matched_indices:
            if iou_matrix[m[0], m[1]] < self.iou_threshold:
                unmatched_detections.append(m[0])
                unmatched_trackers.append(m[1])
            else:
                matches.append(m.reshape(1, 2))

        if len(matches) == 0:
            matches = np.empty((0, 2), dtype=int)
        else:
            matches = np.concatenate(matches, axis=0)

        return matches, np.array(unmatched_detections), np.array(unmatched_trackers)

    @staticmethod
    def _iou(bb_test, bb_gt):
        xx1 = max(bb_test[0], bb_gt[0])
        yy1 = max(bb_test[1], bb_gt[1])
        xx2 = min(bb_test[2], bb_gt[2])
        yy2 = min(bb_test[3], bb_gt[3])

        union_w = max(0., xx2 - xx1)
        union_h = max(0., yy2 - yy1)
        intersection = union_w * union_h
        area_bb_test = (bb_test[2] - bb_test[0]) * (bb_test[3] - bb_test[1])
        area_bb_gt = (bb_gt[2] - bb_gt[0]) * (bb_gt[3] - bb_gt[1])
        union = area_bb_test + area_bb_gt - intersection
        return intersection / (union + 1e-6)
