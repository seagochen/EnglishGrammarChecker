import math
from common.yolo.yolo_results import YoloPose, YoloPoint

# 定义姿势类型
class PoseType:
    Standing = "Standing"
    Squatting = "Squatting"
    Bending = "Bending"
    Unknown = "Unknown"

def calculate_angle(p1, p2, p3):
    """
    计算 p1 -> p2 -> p3 之间的夹角（弧度），以度数返回。
    """
    a = math.sqrt((p2.x - p1.x) ** 2 + (p2.y - p1.y) ** 2)
    b = math.sqrt((p2.x - p3.x) ** 2 + (p2.y - p3.y) ** 2)
    c = math.sqrt((p3.x - p1.x) ** 2 + (p3.y - p1.y) ** 2)
    if a * b == 0:  # 避免除以零
        return 0
    # 余弦定理计算角度
    angle_rad = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    return math.degrees(angle_rad)


def determine_body_angle(yolo_pose: YoloPose) -> float:
    """
    肩ー腰ー膝の関節「かんせつ」形成される接続線を使用して人体の角度を推定します。

    :param yolo_pose: YoloPose 对象，包含人体关键点的信息。
    :return: 返回姿势类型 "Standing", "Squatting" 或 "Bending"
    """
    # 提取关键点坐标
    shoulder_left = yolo_pose.pts[5]
    shoulder_right = yolo_pose.pts[6]
    hip_left = yolo_pose.pts[11]
    hip_right = yolo_pose.pts[12]
    knee_left = yolo_pose.pts[13]
    knee_right = yolo_pose.pts[14]

    # 计算肩膀、髋部和膝盖的平均点，生成 YoloPoint 实例
    shoulder_avg = YoloPoint(
        x=(shoulder_left.x + shoulder_right.x) / 2,
        y=(shoulder_left.y + shoulder_right.y) / 2,
        conf=0  # 置信度在此不需要，设为0即可
    )
    hip_avg = YoloPoint(
        x=(hip_left.x + hip_right.x) / 2,
        y=(hip_left.y + hip_right.y) / 2,
        conf=0
    )
    knee_avg = YoloPoint(
        x=(knee_left.x + knee_right.x) / 2,
        y=(knee_left.y + knee_right.y) / 2,
        conf=0
    )

    # 计算肩膀-髋部-膝盖之间的角度
    angle = calculate_angle(shoulder_avg, hip_avg, knee_avg)
    return angle


def determine_body_pose(yolo_pose: YoloPose, alpha=70.0, beta=140.0) -> str:

    # 体の角度を測る「はかる」
    angle = determine_body_angle(yolo_pose)

    # 閾値「しきいち」を使用して、姿勢を決定「けってい」します。
    if angle > beta:  # 接近180°角时，通常表示站立
        return PoseType.Standing
    elif alpha < angle <= beta:  # 角度在60到160之间表示弯腰
        return PoseType.Bending
    elif angle <= alpha:  # 角度小于60度通常表示蹲下
        return PoseType.Squatting
    else:
        return PoseType.Unknown  # 未知姿势