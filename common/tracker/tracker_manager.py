from typing import List

import numpy as np

from common.tracker.kalman_filter import KalmanFilter


class TrackerManager:
    def __init__(self, max_age: int, max_objects: int):
        self.max_age = max_age
        self.trackers = []  # 存储所有活跃的追踪器
        self.available_ids = set()  # 可用ID的池子，用于重用ID

        # 初始化可用ID池子
        for i in range(max_objects):
            self.available_ids.add(i)

    def _get_next_id(self):
        """获取下一个可用的ID"""
        if self.available_ids:
            return self.available_ids.pop()
        else:
            raise Exception("No available ID left!")

    def _release_id(self, trk_id):
        """将ID放入可用池子"""
        self.available_ids.add(trk_id)

    def add_tracker(self, detection: List[float]):
        """为未匹配的检测结果创建新的追踪器"""
        new_id = self._get_next_id()
        tracker = KalmanFilter(new_id, detection)
        self.trackers.append(tracker)
        return tracker

    def update_trackers(self):
        """更新所有追踪器状态，删除超时未更新的追踪器"""
        active_trackers = []
        for tracker in self.trackers:
            if tracker.time_since_update <= self.max_age:
                active_trackers.append(tracker)
            else:
                self._release_id(tracker.id)  # 释放超时未更新的ID
        self.trackers = active_trackers

    def get_tracker_states(self):
        """返回所有追踪器的状态"""
        states = []
        for tracker in self.trackers:
            pos = tracker.predict()
            if not np.any(np.isnan(pos)):  # 过滤NaN值
                states.append([*tracker.get_state(), tracker.id])
            else:
                self._release_id(tracker.id)  # 释放NaN状态的追踪器ID
        return np.array(states)

