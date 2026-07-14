import os
import cv2
import csv
import ctypes
import time
import numpy as np
import mediapipe as mp
  
from niryo_one_tcp_client import *
from niryo_one_tcp_client.enums import *


ROBOT_IP = "192.168.1.104"

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
VIDEO_FOLDER = PROJECT_FOLDER
CSV_FOLDER = os.path.join(PROJECT_FOLDER, "recorded_robot_paths")

WINDOW_NAME = "Niryo One Arm Tracking"

SIDEBAR_WIDTH = 320
VIDEO_WIDTH = 760
VIDEO_HEIGHT = 430
PROGRESS_HEIGHT = 46

RECORDED_VIDEO_WIDTH = 640
RECORDED_VIDEO_HEIGHT = 360

X_FIXED = 0.24

Y_MIN = -0.18
Y_MAX = 0.18

Z_MIN = 0.16
Z_MAX = 0.34

ROLL = -1.57
PITCH = 0.0
YAW = 0.0

SEND_EVERY_N_FRAMES = 10
MIN_MOVE_DISTANCE = 0.010
FRAME_SKIP = 2

MOTION_GAIN = 1.5
MOTION_GAIN_MIN = 0.5
MOTION_GAIN_MAX = 3.0

USE_RECORDED_DATA_IF_AVAILABLE = True

VIDEO_REFRESH_EVERY_N_FRAMES = 10
VIDEO_REFRESH_INTERVAL_SECONDS = 1.0

VIDEO_LIST_START_Y = 68
VIDEO_ROW_HEIGHT = 16
VIDEO_LIST_END_Y = 294
MAX_VISIBLE_VIDEO_ROWS = (VIDEO_LIST_END_Y - VIDEO_LIST_START_Y) // VIDEO_ROW_HEIGHT

gripper_speed = 1000
gripper_used = RobotTool.GRIPPER_1

video_files = []
selected_index = 0
hovered_index = -1
pending_video_change = False
video_scroll_offset = 0

DELETE_BUTTON = {
    "x1": 20,
    "y1": 346,
    "x2": 300,
    "y2": 382,
    "clicked": False
}

OPEN_GRIPPER_BUTTON = {
    "x1": 20,
    "y1": 392,
    "x2": 145,
    "y2": 428,
    "clicked": False
}

CLOSE_GRIPPER_BUTTON = {
    "x1": 155,
    "y1": 392,
    "x2": 300,
    "y2": 428,
    "clicked": False
}


def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))


def update_motion_gain(value):
    global MOTION_GAIN
    MOTION_GAIN = max(value / 10.0, MOTION_GAIN_MIN)


def set_hand_cursor():
    try:
        ctypes.windll.user32.LoadCursorW(None, 32649)
    except:
        pass


def distance_3d(p1, p2):
    return np.sqrt(
        (p1[0] - p2[0]) ** 2 +
        (p1[1] - p2[1]) ** 2 +
        (p1[2] - p2[2]) ** 2
    )


def get_video_files():
    videos = []
    scan_folder = os.path.abspath(VIDEO_FOLDER)

    try:
        for file in os.listdir(scan_folder):
            full_path = os.path.join(scan_folder, file)

            if os.path.isfile(full_path) and file.lower().endswith(".mp4"):
                videos.append(file)
    except Exception as e:
        print(f"Video folder scan warning: {e}")
        return []

    return sorted(videos, key=lambda name: name.lower())


def refresh_video_files(current_video=None, force_select_new=False):
    global video_files, selected_index, hovered_index, pending_video_change, video_scroll_offset

    new_files = get_video_files()
    old_files = video_files.copy()
    added_files = [file for file in new_files if file not in old_files]

    if new_files == video_files and not force_select_new:
        return

    video_files = new_files

    print("Scanning video folder:", os.path.abspath(VIDEO_FOLDER))
    print(f"Detected {len(video_files)} mp4 video(s):")

    for file in video_files:
        print(f"  {file}")

    if not video_files:
        selected_index = 0
        hovered_index = -1
        pending_video_change = False
        video_scroll_offset = 0
        print("No videos found after refresh.")
        return

    if added_files:
        newest_video = added_files[-1]
        selected_index = video_files.index(newest_video)
        pending_video_change = True
        print(f"New video detected and selected: {newest_video}")

    elif force_select_new:
        if current_video in video_files:
            selected_index = video_files.index(current_video)
        else:
            selected_index = 0
            pending_video_change = True
        print("Video list manually refreshed.")

    else:
        if current_video in video_files:
            selected_index = video_files.index(current_video)
        else:
            selected_index = 0
            pending_video_change = True
        print("Video list refreshed.")

    if hovered_index >= len(video_files):
        hovered_index = -1

    if selected_index < video_scroll_offset:
        video_scroll_offset = selected_index
    elif selected_index >= video_scroll_offset + MAX_VISIBLE_VIDEO_ROWS:
        video_scroll_offset = selected_index - MAX_VISIBLE_VIDEO_ROWS + 1

    video_scroll_offset = int(max(0, min(video_scroll_offset, max(0, len(video_files) - MAX_VISIBLE_VIDEO_ROWS))))


def get_csv_path(video_name):
    os.makedirs(CSV_FOLDER, exist_ok=True)
    name = os.path.splitext(video_name)[0]
    return os.path.join(CSV_FOLDER, f"{name}_robot_path.csv")


def video_has_recording(video_name):
    csv_path = get_csv_path(video_name)
    return os.path.exists(csv_path) and os.path.getsize(csv_path) > 100


def delete_current_recording(current_video):
    csv_path = get_csv_path(current_video)

    if os.path.exists(csv_path):
        os.remove(csv_path)
        print(f"Deleted recording for {current_video}")
    else:
        print(f"No recording found for {current_video}")


def image_to_robot(hand_x, hand_y, frame_width, frame_height):
    norm_x = hand_x / frame_width
    norm_y = hand_y / frame_height

    centred_x = norm_x - 0.5
    centred_y = 0.5 - norm_y

    amplified_x = 0.5 + centred_x * MOTION_GAIN
    amplified_y = 0.5 + centred_y * MOTION_GAIN

    amplified_x = clamp(amplified_x, 0.0, 1.0)
    amplified_y = clamp(amplified_y, 0.0, 1.0)

    robot_x = X_FIXED
    robot_y = Y_MIN + amplified_x * (Y_MAX - Y_MIN)
    robot_z = Z_MIN + amplified_y * (Z_MAX - Z_MIN)

    robot_y = clamp(robot_y, Y_MIN, Y_MAX)
    robot_z = clamp(robot_z, Z_MIN, Z_MAX)

    return robot_x, robot_y, robot_z


def get_row_from_mouse_y(y):
    visible_index = (y - VIDEO_LIST_START_Y) // VIDEO_ROW_HEIGHT

    if 0 <= visible_index < MAX_VISIBLE_VIDEO_ROWS:
        real_index = video_scroll_offset + visible_index

        if 0 <= real_index < len(video_files):
            return real_index

    return -1


def mouse_callback(event, x, y, flags, param):
    global selected_index, hovered_index, pending_video_change, video_scroll_offset

    if x < SIDEBAR_WIDTH:
        hovered_index = get_row_from_mouse_y(y)
    else:
        hovered_index = -1

    if event == cv2.EVENT_MOUSEWHEEL:
        if flags > 0:
            scroll_video_list(-1)
        else:
            scroll_video_list(1)
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        if DELETE_BUTTON["x1"] <= x <= DELETE_BUTTON["x2"] and DELETE_BUTTON["y1"] <= y <= DELETE_BUTTON["y2"]:
            DELETE_BUTTON["clicked"] = True
            return

        if OPEN_GRIPPER_BUTTON["x1"] <= x <= OPEN_GRIPPER_BUTTON["x2"] and OPEN_GRIPPER_BUTTON["y1"] <= y <= OPEN_GRIPPER_BUTTON["y2"]:
            OPEN_GRIPPER_BUTTON["clicked"] = True
            return

        if CLOSE_GRIPPER_BUTTON["x1"] <= x <= CLOSE_GRIPPER_BUTTON["x2"] and CLOSE_GRIPPER_BUTTON["y1"] <= y <= CLOSE_GRIPPER_BUTTON["y2"]:
            CLOSE_GRIPPER_BUTTON["clicked"] = True
            return

        if hovered_index != -1:
            selected_index = hovered_index
            pending_video_change = True


def scroll_video_list(direction):
    global video_scroll_offset

    max_offset = max(0, len(video_files) - MAX_VISIBLE_VIDEO_ROWS)
    video_scroll_offset = int(clamp(video_scroll_offset + direction, 0, max_offset))


def draw_progress_bar(frame, timestamp_ms, total_ms):
    progress_area = np.zeros((PROGRESS_HEIGHT, VIDEO_WIDTH, 3), dtype=np.uint8)
    progress_area[:] = (18, 18, 18)

    progress = clamp(timestamp_ms / total_ms, 0.0, 1.0) if total_ms > 0 else 0.0

    bar_x1 = 24
    bar_y1 = 18
    bar_x2 = VIDEO_WIDTH - 150
    bar_y2 = 30

    cv2.rectangle(progress_area, (bar_x1, bar_y1), (bar_x2, bar_y2), (70, 70, 70), -1)

    fill_x = int(bar_x1 + progress * (bar_x2 - bar_x1))

    cv2.rectangle(progress_area, (bar_x1, bar_y1), (fill_x, bar_y2), (0, 255, 255), -1)

    current_sec = timestamp_ms / 1000
    total_sec = total_ms / 1000 if total_ms > 0 else 0

    cv2.putText(
        progress_area,
        f"{current_sec:.1f}s / {total_sec:.1f}s",
        (VIDEO_WIDTH - 130, 29),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (245, 245, 245),
        1
    )

    return np.vstack((frame, progress_area))


def draw_status_box(frame, playback_mode, timestamp_ms, robot_x, robot_y, robot_z):
    mode_text = "CSV PLAYBACK" if playback_mode else "RECORDING"

    cv2.rectangle(frame, (15, 12), (VIDEO_WIDTH - 15, 52), (30, 30, 30), -1)

    status = f"{mode_text}   |   t = {timestamp_ms:.0f} ms"

    if robot_x is not None:
        status += f"   |   x = {robot_x:.3f}   y = {robot_y:.3f}   z = {robot_z:.3f}"

    cv2.putText(
        frame,
        status,
        (28, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 255),
        2
    )


def draw_sidebar(frame, current_video, playback_mode):
    full_height = VIDEO_HEIGHT + PROGRESS_HEIGHT
    sidebar = np.zeros((full_height, SIDEBAR_WIDTH, 3), dtype=np.uint8)
    sidebar[:] = (10, 10, 10)

    cv2.rectangle(sidebar, (0, 0), (SIDEBAR_WIDTH - 1, full_height - 1), (55, 55, 55), 1)

    cv2.putText(
        sidebar,
        "DETECTED VIDEOS",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2
    )

    if len(video_files) > MAX_VISIBLE_VIDEO_ROWS:
        list_info = f"Showing {video_scroll_offset + 1}-{min(video_scroll_offset + MAX_VISIBLE_VIDEO_ROWS, len(video_files))} of {len(video_files)}"
    else:
        list_info = f"{len(video_files)} video(s)"

    cv2.putText(sidebar, list_info, (20, 62),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

    visible_videos = video_files[video_scroll_offset:video_scroll_offset + MAX_VISIBLE_VIDEO_ROWS]

    for visible_i, video in enumerate(visible_videos):
        i = video_scroll_offset + visible_i
        y = VIDEO_LIST_START_Y + visible_i * VIDEO_ROW_HEIGHT
        box_y1 = y - 12
        box_y2 = y + 3

        if i == selected_index:
            cv2.rectangle(sidebar, (15, box_y1), (SIDEBAR_WIDTH - 15, box_y2), (70, 70, 70), -1)

        if i == hovered_index:
            cv2.rectangle(sidebar, (15, box_y1), (SIDEBAR_WIDTH - 15, box_y2), (45, 90, 90), -1)

        if video_has_recording(video):
            cv2.circle(sidebar, (30, y - 4), 4, (0, 0, 255), -1)

        if i == hovered_index:
            colour = (0, 255, 0)
        elif video == current_video:
            colour = (0, 255, 255)
        else:
            colour = (235, 235, 235)

        cv2.putText(
            sidebar,
            video[:29],
            (50, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            colour,
            1
        )

    cv2.line(sidebar, (20, 298), (SIDEBAR_WIDTH - 20, 298), (90, 90, 90), 1)

    mode_text = "CSV PLAYBACK" if playback_mode else "RECORDING"

    cv2.putText(sidebar, "Mode:", (20, 316),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1)

    cv2.putText(sidebar, mode_text, (105, 316),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

    cv2.putText(sidebar, "Gain:", (20, 336),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1)

    cv2.putText(sidebar, f"{MOTION_GAIN:.1f}x", (105, 336),
                cv2.FONT_HERSHEY_SIMPLEX, 0.46, (0, 255, 255), 1)

    cv2.rectangle(
        sidebar,
        (DELETE_BUTTON["x1"], DELETE_BUTTON["y1"]),
        (DELETE_BUTTON["x2"], DELETE_BUTTON["y2"]),
        (0, 0, 180),
        -1
    )

    cv2.rectangle(
        sidebar,
        (DELETE_BUTTON["x1"], DELETE_BUTTON["y1"]),
        (DELETE_BUTTON["x2"], DELETE_BUTTON["y2"]),
        (255, 255, 255),
        1
    )

    cv2.putText(
        sidebar,
        "DELETE + RECORD AGAIN",
        (37, 370),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2
    )

    cv2.rectangle(
        sidebar,
        (OPEN_GRIPPER_BUTTON["x1"], OPEN_GRIPPER_BUTTON["y1"]),
        (OPEN_GRIPPER_BUTTON["x2"], OPEN_GRIPPER_BUTTON["y2"]),
        (0, 140, 0),
        -1
    )

    cv2.rectangle(
        sidebar,
        (OPEN_GRIPPER_BUTTON["x1"], OPEN_GRIPPER_BUTTON["y1"]),
        (OPEN_GRIPPER_BUTTON["x2"], OPEN_GRIPPER_BUTTON["y2"]),
        (255, 255, 255),
        1
    )

    cv2.putText(
        sidebar,
        "OPEN",
        (48, 416),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.rectangle(
        sidebar,
        (CLOSE_GRIPPER_BUTTON["x1"], CLOSE_GRIPPER_BUTTON["y1"]),
        (CLOSE_GRIPPER_BUTTON["x2"], CLOSE_GRIPPER_BUTTON["y2"]),
        (140, 0, 0),
        -1
    )

    cv2.rectangle(
        sidebar,
        (CLOSE_GRIPPER_BUTTON["x1"], CLOSE_GRIPPER_BUTTON["y1"]),
        (CLOSE_GRIPPER_BUTTON["x2"], CLOSE_GRIPPER_BUTTON["y2"]),
        (255, 255, 255),
        1
    )

    cv2.putText(
        sidebar,
        "CLOSE",
        (185, 416),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2
    )

    cv2.putText(sidebar, "Click video to select", (20, full_height - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)

    cv2.putText(sidebar, "Q quit | R refresh | wheel scroll", (20, full_height - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

    return np.hstack((sidebar, frame))


def load_recorded_data(csv_path):
    data = []

    if not os.path.exists(csv_path):
        return data

    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)

        for row in reader:
            data.append({
                "timestamp_ms": float(row["timestamp_ms"]),
                "frame": int(float(row["frame"])),
                "sx": int(float(row["shoulder_x"])),
                "sy": int(float(row["shoulder_y"])),
                "ex": int(float(row["elbow_x"])),
                "ey": int(float(row["elbow_y"])),
                "wx": int(float(row["wrist_x"])),
                "wy": int(float(row["wrist_y"])),
                "robot_x": float(row["robot_x"]),
                "robot_y": float(row["robot_y"]),
                "robot_z": float(row["robot_z"]),
                "visibility": float(row["visibility"])
            })

    return data


def get_record_for_time(data, timestamp_ms, last_index):
    if not data:
        return None, last_index

    while last_index + 1 < len(data):
        if data[last_index + 1]["timestamp_ms"] <= timestamp_ms:
            last_index += 1
        else:
            break

    return data[last_index], last_index


def create_csv_writer(csv_path, video_name):
    file = open(csv_path, "w", newline="")
    writer = csv.writer(file)

    writer.writerow([
        "video",
        "frame",
        "timestamp_ms",
        "video_width",
        "video_height",
        "shoulder_x", "shoulder_y",
        "elbow_x", "elbow_y",
        "wrist_x", "wrist_y",
        "visibility",
        "robot_x", "robot_y", "robot_z",
        "roll", "pitch", "yaw",
        "motion_gain"
    ])

    print(f"Recording new CSV for {video_name}")
    return file, writer


def safe_sleep_mode(robot):
    try:
        robot.move_pose(X_FIXED, 0.0, 0.25, ROLL, PITCH, YAW)
        print("Robot returned to safe pose.")
    except Exception as e:
        print(f"Safe pose warning: {e}")

    try:
        robot.close_gripper(gripper_used, gripper_speed)
    except Exception as e:
        print(f"Gripper warning: {e}")

    for method_name in ["set_learning_mode", "activate_learning_mode", "change_learning_mode"]:
        if hasattr(robot, method_name):
            try:
                getattr(robot, method_name)(True)
                print("Robot returned to sleep mode.")
                return
            except Exception as e:
                print(f"{method_name} warning: {e}")

    print("Sleep mode command not available in this Niryo client version.")


def get_left_screen_arm(lm, mp_pose, width, height):
    left_shoulder = lm[mp_pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = lm[mp_pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = lm[mp_pose.PoseLandmark.LEFT_WRIST]

    right_shoulder = lm[mp_pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = lm[mp_pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = lm[mp_pose.PoseLandmark.RIGHT_WRIST]

    if left_wrist.x < right_wrist.x:
        shoulder = left_shoulder
        elbow = left_elbow
        wrist = left_wrist
    else:
        shoulder = right_shoulder
        elbow = right_elbow
        wrist = right_wrist

    sx = int(shoulder.x * width)
    sy = int(shoulder.y * height)

    ex = int(elbow.x * width)
    ey = int(elbow.y * height)

    wx = int(wrist.x * width)
    wy = int(wrist.y * height)

    visibility = min(shoulder.visibility, elbow.visibility, wrist.visibility)

    return sx, sy, ex, ey, wx, wy, visibility


def draw_arm(frame, sx, sy, ex, ey, wx, wy):
    cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 255), 4)
    cv2.line(frame, (ex, ey), (wx, wy), (0, 255, 255), 4)

    cv2.circle(frame, (sx, sy), 6, (0, 0, 255), -1)
    cv2.circle(frame, (ex, ey), 6, (255, 0, 0), -1)
    cv2.circle(frame, (wx, wy), 8, (0, 255, 0), -1)


def go_to_next_video():
    global selected_index, pending_video_change

    selected_index += 1

    if selected_index >= len(video_files):
        selected_index = 0

    pending_video_change = True


def main():
    global video_files, pending_video_change, selected_index, video_scroll_offset

    video_files = get_video_files()

    if not video_files:
        print("No .mp4 videos found.")
        return

    robot = NiryoOneClient()

    if not robot.connect(ROBOT_IP):
        print("Failed to connect to Niryo One robot.")
        return

    print("Connected to Niryo One robot.")

    cap = None
    current_video = None
    last_sent_pose = None

    csv_file = None
    csv_writer = None
    recorded_data = []
    recorded_index = 0
    playback_mode = False

    frame_id = 0
    total_frames = 0
    fps = 0
    loop_counter = 0
    last_video_refresh_time = 0.0

    mp_pose = mp.solutions.pose

    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=0,
        smooth_landmarks=True,
        min_detection_confidence=0.35,
        min_tracking_confidence=0.35
    )

    try:
        robot.calibrate(CalibrateMode.AUTO)
        print("Calibration complete.")

        robot.change_tool(gripper_used)
        robot.close_gripper(gripper_used, gripper_speed)

        robot.move_pose(X_FIXED, 0.0, 0.25, ROLL, PITCH, YAW)
        last_sent_pose = (X_FIXED, 0.0, 0.25)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

        cv2.resizeWindow(
            WINDOW_NAME,
            SIDEBAR_WIDTH + VIDEO_WIDTH,
            VIDEO_HEIGHT + PROGRESS_HEIGHT
        )

        cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

        cv2.createTrackbar(
            "Motion Gain x10",
            WINDOW_NAME,
            int(MOTION_GAIN * 10),
            int(MOTION_GAIN_MAX * 10),
            update_motion_gain
        )

        try:
            cv2.setTrackbarMin(
                "Motion Gain x10",
                WINDOW_NAME,
                int(MOTION_GAIN_MIN * 10)
            )
        except:
            pass

        pending_video_change = True

        while True:
            set_hand_cursor()

            loop_counter += 1

            now = time.time()

            if now - last_video_refresh_time >= VIDEO_REFRESH_INTERVAL_SECONDS:
                refresh_video_files(current_video, force_select_new=False)
                last_video_refresh_time = now

            if not video_files:
                print("No .mp4 videos found.")
                break

            if selected_index >= len(video_files):
                selected_index = 0
                pending_video_change = True

            if pending_video_change:
                current_video = video_files[selected_index]
                video_path = os.path.join(VIDEO_FOLDER, current_video)
                csv_path = get_csv_path(current_video)

                if cap is not None:
                    cap.release()

                if csv_file is not None:
                    csv_file.close()
                    csv_file = None
                    csv_writer = None

                cap = cv2.VideoCapture(video_path)
                frame_id = 0
                recorded_index = 0
                last_sent_pose = (X_FIXED, 0.0, 0.25)
                pending_video_change = False

                if not cap.isOpened():
                    print(f"Error: Cannot open video {current_video}")
                    go_to_next_video()
                    continue

                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                recorded_data = []

                if USE_RECORDED_DATA_IF_AVAILABLE and os.path.exists(csv_path):
                    recorded_data = load_recorded_data(csv_path)
                    playback_mode = len(recorded_data) > 0
                    print(f"Loaded CSV path for {current_video}")
                else:
                    playback_mode = False
                    csv_file, csv_writer = create_csv_writer(csv_path, current_video)

                print(f"Loaded video: {current_video}")

            ret, frame = cap.read()

            if not ret:
                print(f"Finished video: {current_video}")
                go_to_next_video()
                continue

            frame_id += 1

            if frame_id % FRAME_SKIP != 0:
                continue

            timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

            if fps > 0 and total_frames > 0:
                total_ms = (total_frames / fps) * 1000
            else:
                total_ms = 0

            frame = cv2.resize(frame, (VIDEO_WIDTH, VIDEO_HEIGHT))
            display_frame = frame.copy()

            height, width, _ = frame.shape

            robot_x = None
            robot_y = None
            robot_z = None

            if playback_mode:
                record, recorded_index = get_record_for_time(
                    recorded_data,
                    timestamp_ms,
                    recorded_index
                )

                if record is not None:
                    scale_x = width / RECORDED_VIDEO_WIDTH
                    scale_y = height / RECORDED_VIDEO_HEIGHT

                    sx = int(record["sx"] * scale_x)
                    sy = int(record["sy"] * scale_y)

                    ex = int(record["ex"] * scale_x)
                    ey = int(record["ey"] * scale_y)

                    wx = int(record["wx"] * scale_x)
                    wy = int(record["wy"] * scale_y)

                    robot_x = record["robot_x"]
                    robot_y = record["robot_y"]
                    robot_z = record["robot_z"]

                    draw_arm(display_frame, sx, sy, ex, ey, wx, wy)

            else:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = pose.process(rgb_frame)

                if result.pose_landmarks:
                    lm = result.pose_landmarks.landmark

                    sx, sy, ex, ey, wx, wy, visibility = get_left_screen_arm(
                        lm,
                        mp_pose,
                        width,
                        height
                    )

                    if visibility > 0.15:
                        draw_arm(display_frame, sx, sy, ex, ey, wx, wy)

                        robot_x, robot_y, robot_z = image_to_robot(
                            wx,
                            wy,
                            width,
                            height
                        )

                        if csv_writer is not None:
                            csv_writer.writerow([
                                current_video,
                                frame_id,
                                timestamp_ms,
                                width,
                                height,
                                sx, sy,
                                ex, ey,
                                wx, wy,
                                visibility,
                                robot_x, robot_y, robot_z,
                                ROLL, PITCH, YAW,
                                MOTION_GAIN
                            ])
                    else:
                        cv2.putText(
                            display_frame,
                            "Arm visibility too low",
                            (28, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.70,
                            (0, 0, 255),
                            2
                        )
                else:
                    cv2.putText(
                        display_frame,
                        "No body detected",
                        (28, 90),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.70,
                        (0, 0, 255),
                        2
                    )

            draw_status_box(
                display_frame,
                playback_mode,
                timestamp_ms,
                robot_x,
                robot_y,
                robot_z
            )

            if robot_x is not None:
                current_pose = (robot_x, robot_y, robot_z)

                should_send = False

                if frame_id % SEND_EVERY_N_FRAMES == 0:
                    if last_sent_pose is None:
                        should_send = True
                    elif distance_3d(current_pose, last_sent_pose) > MIN_MOVE_DISTANCE:
                        should_send = True

                if should_send:
                    try:
                        robot.move_pose(
                            robot_x,
                            robot_y,
                            robot_z,
                            ROLL,
                            PITCH,
                            YAW
                        )

                        last_sent_pose = current_pose

                    except Exception as e:
                        print(f"Robot command skipped: {e}")

            frame_with_progress = draw_progress_bar(
                display_frame,
                timestamp_ms,
                total_ms
            )

            combined_frame = draw_sidebar(
                frame_with_progress,
                current_video,
                playback_mode
            )

            cv2.imshow(WINDOW_NAME, combined_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r") or key == ord("R"):
                refresh_video_files(current_video, force_select_new=True)
                print("Detected videos:")
                for video in video_files:
                    print(f"  {video}")
                last_video_refresh_time = time.time()

            if DELETE_BUTTON["clicked"]:
                DELETE_BUTTON["clicked"] = False

                if current_video is not None:
                    delete_current_recording(current_video)
                    pending_video_change = True

            if OPEN_GRIPPER_BUTTON["clicked"]:
                OPEN_GRIPPER_BUTTON["clicked"] = False

                try:
                    robot.open_gripper(gripper_used, gripper_speed)
                    print("Gripper opened.")
                except Exception as e:
                    print(f"Open gripper warning: {e}")

            if CLOSE_GRIPPER_BUTTON["clicked"]:
                CLOSE_GRIPPER_BUTTON["clicked"] = False

                try:
                    robot.close_gripper(gripper_used, gripper_speed)
                    print("Gripper closed.")
                except Exception as e:
                    print(f"Close gripper warning: {e}")

    except KeyboardInterrupt:
        print("Stopped by user.")

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        if cap is not None:
            cap.release()

        if csv_file is not None:
            csv_file.close()

        pose.close()
        cv2.destroyAllWindows()

        safe_sleep_mode(robot)
        robot.quit()

        print("Disconnected from robot.")


if __name__ == "__main__":
    main()
