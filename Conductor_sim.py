import os
import cv2
import csv
import ctypes
import time
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import mediapipe as mp

try:
    import mujoco
    HAS_MUJOCO = True
except Exception:
    HAS_MUJOCO = False

try:
    from niryo_one_tcp_client import *
    from niryo_one_tcp_client.enums import *
    HAS_NIRYO = True
except Exception:
    HAS_NIRYO = False

    class RobotTool:
        GRIPPER_1 = "GRIPPER_1"

    class CalibrateMode:
        AUTO = "AUTO"


SIM_MODE = True

PROJECT_FOLDER = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_FOLDER, "niryo_one")
URDF_PATH = os.path.join(MODEL_DIR, "niryo_one.urdf")

TABLE_Z = 0.0
SIM_RENDER_W = 720
SIM_RENDER_H = 540
MAX_JOINT_VELOCITY = 3.0
CONTROL_RATE_HZ = 30.0

SIM_WINDOW_NAME = "MuJoCo (offscreen)"

ROBOT_IP = "192.168.1.104"

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


class SimRobot:
    HOME_POSE = [0.0, 0.400, -1.100, 0.000, -0.800, -0.227]

    ARM_JOINT_NAMES = ("joint_1", "joint_2", "joint_3",
                       "joint_4", "joint_5", "joint_6")

    EE_LINK_CANDIDATES = ("tool_link", "hand_link", "gripper_base",
                          "base_gripper")

    NIRYO_COLOURS = {
        "base_link":     "0.20 0.20 0.22 1",
        "shoulder_link": "0.416 0.643 0.867 1",
        "arm_link":      "0.416 0.643 0.867 1",
        "elbow_link":    "0.15 0.15 0.15 1",
        "forearm_link":  "0.416 0.643 0.867 1",
        "wrist_link":    "0.15 0.15 0.15 1",
        "hand_link":     "0.15 0.15 0.15 1",
        "tool_link":     "0.15 0.15 0.15 1",
    }
    GRIPPER_BODY_RGBA = "0.30 0.30 0.30 1"
    GRIPPER_JAW_RGBA = "0.45 0.45 0.45 1"

    _SMOOTH_ALPHA = 0.85
    _GRAB_DISTANCE = 0.10

    def __init__(self, urdf_path):
        if not HAS_MUJOCO:
            raise RuntimeError("mujoco is not installed in this environment")
        self._urdf_path = Path(urdf_path)
        self._model = None
        self._data = None
        self._renderer = None
        self._joint_map = {}
        self._arm_joints = []
        self._gripper_joints = []
        self._ee_body_id = 0
        self._arm_dof_indices = []
        self._grabbed_body = -1
        self._initialize()

    @classmethod
    def _prepare_urdf(cls, urdf_path):
        tree = ET.parse(urdf_path)
        urdf_dir = urdf_path.parent.resolve()
        assets = {}
        dae_swaps = 0

        for mesh_el in tree.iter("mesh"):
            fn = mesh_el.get("filename", "")
            if not fn:
                continue

            if fn.lower().endswith(".dae"):
                dae_dir = Path(fn).parent
                stem = Path(fn).stem
                obj_fn = str(dae_dir / f"{stem}.obj")
                stl_fn = fn[:-4] + ".stl"
                collada_to_stl = stl_fn.replace("/collada/", "/stl/")

                if (urdf_dir / obj_fn).exists():
                    fn = obj_fn
                    dae_swaps += 1
                elif (urdf_dir / stl_fn).exists():
                    fn = stl_fn
                    dae_swaps += 1
                elif collada_to_stl != stl_fn and (urdf_dir / collada_to_stl).exists():
                    fn = collada_to_stl
                    dae_swaps += 1
                else:
                    print(f"No OBJ/STL for {fn} - skipping")
                    continue

            abs_path = (urdf_dir / fn).resolve()
            basename = abs_path.name
            mesh_el.set("filename", basename)
            if basename not in assets and abs_path.exists():
                assets[basename] = abs_path.read_bytes()

        cls._inject_colours(tree)

        for joint_el in tree.iter("joint"):
            p = joint_el.find("parent")
            c = joint_el.find("child")
            if (p is not None and p.get("link") == "world" and
                    c is not None and c.get("link") == "base_link"):
                o = joint_el.find("origin")
                if o is not None:
                    xyz = (o.get("xyz", "0 0 0").split() + ["0", "0", "0"])[:3]
                    o.set("xyz", f"{xyz[0]} {xyz[1]} {TABLE_Z}")

        xml_str = ET.tostring(tree.getroot(), encoding="unicode",
                              xml_declaration=False)
        if dae_swaps:
            print(f"Swapped {dae_swaps} .dae mesh references")
        print(f"Loaded {len(assets)} mesh assets into memory")
        return xml_str, assets

    @classmethod
    def _inject_colours(cls, tree):
        for link_el in tree.iter("link"):
            link_name = link_el.get("name", "").lower()
            vis_el = link_el.find("visual")
            if vis_el is None or vis_el.find("material") is not None:
                continue
            rgba = cls.NIRYO_COLOURS.get(link_name)
            if rgba is None:
                if any(k in link_name for k in ("mors", "clamp", "left_gripper", "right_gripper")):
                    rgba = cls.GRIPPER_JAW_RGBA
                elif any(k in link_name for k in ("grip", "rod", "servo", "base_gripper")):
                    rgba = cls.GRIPPER_BODY_RGBA
                else:
                    rgba = cls.NIRYO_COLOURS.get("shoulder_link")
            mat_el = ET.SubElement(vis_el, "material", name=f"mat_{link_name}")
            ET.SubElement(mat_el, "color", rgba=rgba)

    @staticmethod
    def _add_scene(mjcf_xml, assets):
        root = ET.fromstring(mjcf_xml)
        worldbody = root.find("worldbody")

        asset = root.find("asset")
        if asset is None:
            asset = ET.SubElement(root, "asset")

        ET.SubElement(asset, "texture", name="grid", type="2d",
                      builtin="checker", rgb1="0.22 0.30 0.42",
                      rgb2="0.18 0.24 0.35", width="512", height="512")
        ET.SubElement(asset, "material", name="grid_mat", texture="grid",
                      texrepeat="6 6", texuniform="true", reflectance="0.2",
                      specular="0.3", shininess="0.4")
        ET.SubElement(asset, "texture", name="skybox", type="skybox",
                      builtin="gradient", rgb1="0.20 0.28 0.45",
                      rgb2="0.35 0.45 0.60", width="512", height="3072")

        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.set("gravity", "0 0 0")

        ET.SubElement(worldbody, "geom", name="floor", type="plane",
                      size="2 2 0.1", material="grid_mat",
                      pos="0 0 0", conaffinity="1", condim="3")
        ET.SubElement(worldbody, "light", name="key", pos="0.5 -0.6 1.8",
                      dir="-0.1 0.3 -1", diffuse="0.75 0.78 0.85",
                      specular="0.3 0.3 0.35", castshadow="true")
        ET.SubElement(worldbody, "light", name="fill", pos="-0.8 0.5 1.2",
                      dir="0.5 -0.3 -0.5", diffuse="0.35 0.38 0.45",
                      specular="0.05 0.05 0.08")
        ET.SubElement(worldbody, "light", name="rim", pos="-0.3 0.8 0.8",
                      dir="0.3 -0.8 -0.3", diffuse="0.22 0.24 0.30")

        ET.SubElement(worldbody, "camera", name="front",
                      pos="0.72 -0.38 0.52",
                      mode="targetbody", target="shoulder_link",
                      fovy="45")

        visual = root.find("visual")
        if visual is None:
            visual = ET.SubElement(root, "visual")
        quality = visual.find("quality")
        if quality is None:
            quality = ET.SubElement(visual, "quality")
        quality.set("shadowsize", "4096")
        hl = visual.find("headlight")
        if hl is None:
            hl = ET.SubElement(visual, "headlight")
        hl.set("ambient", "0.15 0.16 0.20")
        hl.set("diffuse", "0.25 0.27 0.32")
        hl.set("specular", "0.1 0.1 0.12")
        rgba_el = visual.find("rgba")
        if rgba_el is None:
            rgba_el = ET.SubElement(visual, "rgba")
        rgba_el.set("haze", "0.25 0.35 0.50 1")
        fog = visual.find("map")
        if fog is None:
            fog = ET.SubElement(visual, "map")
        fog.set("fogstart", "3")
        fog.set("fogend", "10")

        glob = visual.find("global")
        if glob is None:
            glob = ET.SubElement(visual, "global")
        glob.set("offwidth", str(SIM_RENDER_W))
        glob.set("offheight", str(SIM_RENDER_H))

        return mujoco.MjModel.from_xml_string(
            ET.tostring(root, encoding="unicode"), assets,
        )

    def _initialize(self):
        xml_str, assets = self._prepare_urdf(self._urdf_path)

        urdf_model = mujoco.MjModel.from_xml_string(xml_str, assets)
        tmp = tempfile.NamedTemporaryFile(suffix=".xml", delete=False, mode="w")
        tmp.close()
        mujoco.mj_saveLastXML(tmp.name, urdf_model)
        mjcf_xml = Path(tmp.name).read_text()
        Path(tmp.name).unlink()

        self._model = self._add_scene(mjcf_xml, assets)
        self._data = mujoco.MjData(self._model)
        self._model.dof_damping[:] = 100.0

        for i in range(self._model.njnt):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name is not None:
                self._joint_map[name] = (
                    int(self._model.jnt_qposadr[i]),
                    int(self._model.jnt_dofadr[i]),
                )

        GRIPPER_PREFIXES = ("joint_base_to_mors", "joint_base_to_clamp",
                            "left_gripper", "right_gripper")
        for name, (qadr, dadr) in self._joint_map.items():
            if name in self.ARM_JOINT_NAMES:
                self._arm_joints.append((name, qadr, dadr))
            elif any(name.startswith(p) for p in GRIPPER_PREFIXES):
                self._gripper_joints.append((name, qadr, dadr))
        self._arm_joints.sort(key=lambda x: x[0])
        self._arm_dof_indices = [dadr for _, _, dadr in self._arm_joints]

        self._ee_body_id = self._find_ee_body()

        self._renderer = mujoco.Renderer(self._model, height=SIM_RENDER_H,
                                         width=SIM_RENDER_W)

        print(f"MuJoCo model loaded: {self._urdf_path.name}")
        print("Arm joints:", [n for n, _, _ in self._arm_joints],
              " Gripper joints:", [n for n, _, _ in self._gripper_joints])

        self._go_home()

    def _find_ee_body(self):
        for name in self.EE_LINK_CANDIDATES:
            bid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                return bid
        last_name = self._arm_joints[-1][0]
        jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, last_name)
        return int(self._model.jnt_bodyid[jid])

    def _go_home(self):
        for i, (_, qadr, _) in enumerate(self._arm_joints):
            if i < len(self.HOME_POSE):
                self._data.qpos[qadr] = self.HOME_POSE[i]
        mujoco.mj_forward(self._model, self._data)
        ee = self.get_end_effector_position()
        print(f"Home pose set - EE at ({ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f})")

    def get_joint_positions(self):
        return [float(self._data.qpos[qadr]) for _, qadr, _ in self._arm_joints]

    def get_end_effector_position(self):
        return np.array(self._data.xpos[self._ee_body_id], dtype=np.float64)

    def solve_ik(self, target_position, target_orientation=None):
        qpos_save = self._data.qpos.copy()
        qvel_save = self._data.qvel.copy()

        max_iter = 100
        tol = 5e-4
        damping = 0.5
        step_scale = 0.8

        for _ in range(max_iter):
            mujoco.mj_forward(self._model, self._data)
            ee_pos = np.array(self._data.xpos[self._ee_body_id])
            error = target_position - ee_pos

            if np.linalg.norm(error) < tol:
                break

            jacp = np.zeros((3, self._model.nv))
            mujoco.mj_jac(self._model, self._data, jacp, None,
                          ee_pos, self._ee_body_id)

            J = jacp[:, self._arm_dof_indices]
            JJT = J @ J.T + damping ** 2 * np.eye(3)
            dq = step_scale * (J.T @ np.linalg.solve(JJT, error))

            for i, (name, qadr, _) in enumerate(self._arm_joints):
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                lo = self._model.jnt_range[jid, 0]
                hi = self._model.jnt_range[jid, 1]
                self._data.qpos[qadr] = float(np.clip(
                    self._data.qpos[qadr] + dq[i], lo, hi,
                ))

        result = self.get_joint_positions()

        self._data.qpos[:] = qpos_save
        self._data.qvel[:] = qvel_save
        mujoco.mj_forward(self._model, self._data)

        return result

    def set_joint_positions(self, positions):
        alpha = self._SMOOTH_ALPHA
        max_vel = MAX_JOINT_VELOCITY * 2.0
        dt = 1.0 / CONTROL_RATE_HZ

        for i, angle in enumerate(positions):
            if i < len(self._arm_joints):
                name, qadr, _ = self._arm_joints[i]
                jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
                lo = self._model.jnt_range[jid, 0]
                hi = self._model.jnt_range[jid, 1]
                target = float(np.clip(angle, lo, hi))
                current = float(self._data.qpos[qadr])
                smoothed = current + alpha * (target - current)
                delta = np.clip(smoothed - current, -max_vel * dt, max_vel * dt)
                self._data.qpos[qadr] = current + float(delta)

    def set_gripper(self, openness):
        for name, qadr, _ in self._gripper_joints:
            jid = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            lo = float(self._model.jnt_range[jid, 0])
            hi = float(self._model.jnt_range[jid, 1])
            if "right" in name.lower():
                self._data.qpos[qadr] = hi - openness * (hi - lo)
            else:
                self._data.qpos[qadr] = lo + openness * (hi - lo)
        if openness < 0.3:
            self._try_grab()
        elif openness > 0.7:
            self._release()

    def step(self):
        if self._grabbed_body >= 0:
            ee_pos = self.get_end_effector_position()
            for i in range(self._model.njnt):
                if (self._model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE and
                        self._model.jnt_bodyid[i] == self._grabbed_body):
                    qadr = int(self._model.jnt_qposadr[i])
                    self._data.qpos[qadr:qadr+3] = ee_pos - [0, 0, 0.04]
                    break
        mujoco.mj_forward(self._model, self._data)

    def _try_grab(self):
        if self._grabbed_body >= 0:
            return
        ee_pos = self.get_end_effector_position()
        for i in range(self._model.nbody):
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY, i)
            if name and name.startswith("target_"):
                obj_pos = self._data.xpos[i]
                if np.linalg.norm(ee_pos - obj_pos) < self._GRAB_DISTANCE:
                    self._grabbed_body = i
                    print(f"Grabbed {name}")
                    return

    def _release(self):
        if self._grabbed_body >= 0:
            name = mujoco.mj_id2name(self._model, mujoco.mjtObj.mjOBJ_BODY,
                                     self._grabbed_body)
            print(f"Released {name}")
            self._grabbed_body = -1

    def render_frame(self):
        cam_id = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
        if cam_id >= 0:
            self._renderer.update_scene(self._data, camera=cam_id)
        else:
            self._renderer.update_scene(self._data)
        rgb = self._renderer.render()
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # --- NiryoOneClient-compatible interface used by the conductor loop ---

    def connect(self, ip):
        return True

    def calibrate(self, mode):
        pass

    def change_tool(self, tool):
        pass

    def close_gripper(self, tool, speed):
        self.set_gripper(0.0)

    def open_gripper(self, tool, speed):
        self.set_gripper(1.0)

    def move_pose(self, x, y, z, roll, pitch, yaw):
        joints = self.solve_ik(np.array([x, y, z], dtype=np.float64))
        self.set_joint_positions(joints)
        self.step()
        cv2.imshow(SIM_WINDOW_NAME, self.render_frame())

    def quit(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
        print("MuJoCo backend disconnected")


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

    if SIM_MODE:
        robot = SimRobot(URDF_PATH)
    else:
        robot = NiryoOneClient()

    if not robot.connect(ROBOT_IP):
        print("Failed to connect to Niryo One robot.")
        return

    print("Simulation ready." if SIM_MODE else "Connected to Niryo One robot.")

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
