import json
import numpy as np
import os
from tqdm import tqdm


def path_parser(root_path, data_source='TagBA'):
    full_path = root_path
    path_dict = dict()
    if data_source == 'TagBA':
        path_dict['camera_pose'] = os.path.join(
            full_path, 'TagBA', 'CameraTrajectory-BA.txt')
        path_dict['cam_intrinsic'] = os.path.join(
            full_path, 'camera_intrinsics.json')
    elif data_source == 'ARKit':
        path_dict['camera_pose'] = os.path.join(full_path, 'SyncedPoses.txt')
        path_dict['cam_intrinsic'] = os.path.join(full_path, 'Frames.txt')
    elif data_source == 'SenseAR':
        path_dict['camera_pose'] = os.path.join(full_path, 'frame_pose.csv')
        path_dict['cam_intrinsic'] = os.path.join(
            full_path, 'device_parameter.txt')
    return path_dict


def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])


def load_camera_pose(cam_pose_dir, use_homogenous=True, data_source='TagBA'):
    if cam_pose_dir is not None and os.path.isfile(cam_pose_dir):
        pass
    else:
        raise FileNotFoundError("Given camera pose dir:{} not found"
                                .format(cam_pose_dir))

    from transforms3d.quaternions import quat2mat

    pose_dict = dict()

    def process(line_data_list):
        line_data = np.array(line_data_list, dtype=float)
        fid = line_data_list[0]
        trans = line_data[1:4]
        quat = line_data[4:]
        rot_mat = quat2mat(np.append(quat[-1], quat[:3]).tolist())
        if data_source == 'ARKit':
            rot_mat = rot_mat.dot(np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ]))
            rot_mat = rotx(np.pi / 2) @ rot_mat
            trans = rotx(np.pi / 2) @ trans
        trans_mat = np.zeros([3, 4])
        trans_mat[:3, :3] = rot_mat
        trans_mat[:3, 3] = trans
        if use_homogenous:
            trans_mat = np.vstack((trans_mat, [0, 0, 0, 1]))
        pose_dict[fid] = trans_mat

    print(f"data source: {data_source}")
    if data_source == 'TagBA' or data_source == 'ARKit':
        with open(cam_pose_dir, "r") as f:
            cam_pose_lines = f.readlines()
        for cam_line in cam_pose_lines:
            line_data_list = cam_line.split(" ")
            if len(line_data_list) == 0:
                continue
            process(line_data_list)
    elif data_source == 'SenseAR':
        import csv
        with open(cam_pose_dir, 'r') as f:
            reader = csv.reader(f, delimiter=' ', quotechar='|')
            for line_data_list in reader:
                if len(line_data_list) is not 8:
                    continue
                process(line_data_list)

    return pose_dict


def load_camera_intrinsic(cam_file, data_source='TagBA'):
    """Load camera parameter from file"""
    assert os.path.isfile(
        cam_file), "camera info:{} not found".format(cam_file)

    cam_dict = dict()
    if data_source == 'TagBA':
        with open(cam_file, "r") as f:
            cam_info = json.load(f)
        cam_dict['K'] = np.array([
            [cam_info['fx'], 0, cam_info['cx']],
            [0, cam_info['fy'], cam_info['cy']],
            [0, 0, 1]
        ], dtype=float)
        w = int(cam_info['horizontal_resolution'])
        h = int(cam_info['vertical_resolution'])
        cam_dict['shape'] = (w, h)
        cam_dict['dist_coeff'] = np.array(
            cam_info['distortion_coefficients'], dtype=float)
    elif data_source == 'Open3D':
        with open(cam_file, "r") as f:
            cam_info = json.load(f)
        cam_dict['K'] = np.array(
            cam_info['intrinsic_matrix'], dtype=float).reshape(3, 3).transpose()
        w = int(cam_info['width'])
        h = int(cam_info['height'])
        cam_dict['shape'] = (w, h)
        cam_dict['dist_coeff'] = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    elif data_source == 'SenseAR':
        with open(cam_file, 'r') as f:
            cam_lines = f.readlines()
            cam_dict['K'] = np.array([
                [float(cam_lines[2].split(': ')[1]), 0, float(cam_lines[4].split(': ')[1])],
                [0, float(cam_lines[3].split(': ')[1]), float(cam_lines[5].split(': ')[1])],
                [0, 0, 1]
            ], dtype=float)
    elif data_source == 'ARKit':
        with open(cam_file, "r") as f:
            cam_intrinsic_lines = f.readlines()

        cam_intrinsic_dict = dict()
        for line in cam_intrinsic_lines:
            line_data_list = [float(i) for i in line.split(',')]
            if len(line_data_list) == 0:
                continue
            cam_dict = dict()
            cam_dict['K'] = np.array([
                [line_data_list[2], 0, line_data_list[4]],
                [0, line_data_list[3], line_data_list[5]],
                [0, 0, 1]
            ], dtype=float)
            cam_intrinsic_dict[str(int(line_data_list[1])).zfill(5)] = cam_dict
        return cam_intrinsic_dict
    else:
        raise NotImplementedError(
            "Data parsing for source: {} not implemented".format(data_source))

    return cam_dict


def extract_frames(video_path, out_folder, size):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(frame_count)):
        ret, frame = cap.read()
        if ret is not True:
            break
        frame = cv2.resize(frame, size)
        cv2.imwrite(os.path.join(out_folder, str(i).zfill(5) + '.jpg'), frame)
