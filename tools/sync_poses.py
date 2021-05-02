import os
import argparse


def sync_intrinsics_and_poses(cam_file, pose_file, out_file):
    """Load camera intrinsics"""
    assert os.path.isfile(cam_file), "camera info:{} not found".format(cam_file)
    with open(cam_file, "r") as f:
        cam_intrinsic_lines = f.readlines()
    
    cam_intrinsics = []
    for line in cam_intrinsic_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_intrinsics.append([float(i) for i in line_data_list])

    """load camera poses"""
    assert os.path.isfile(pose_file), "camera info:{} not found".format(pose_file)
    with open(pose_file, "r") as f:
        cam_pose_lines = f.readlines()

    cam_poses = []
    for line in cam_pose_lines:
        line_data_list = line.split(',')
        if len(line_data_list) == 0:
            continue
        cam_poses.append([float(i) for i in line_data_list])

    
    lines = []
    ip = 0
    length = len(cam_poses)
    for i in range(len(cam_intrinsics)):
        while ip + 1< length and abs(cam_poses[ip + 1][0] - cam_intrinsics[i][0]) < abs(cam_poses[ip][0] - cam_intrinsics[i][0]):
            ip += 1
        cam_pose = cam_poses[ip][:4] + cam_poses[ip][5:] + [cam_poses[ip][4]]
        line = [str(a) for a in cam_pose]
        # line = [str(a) for a in cam_poses[ip]]
        line[0] = str(i).zfill(5)
        lines.append(' '.join(line) + '\n')
    
    dirname = os.path.dirname(out_file)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(out_file, 'w') as f:
        f.writelines(lines)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cam_file', type=str, default='../Frames.txt')
    parser.add_argument('--pose_file', type=str, default='../ARPoses.txt')
    parser.add_argument('--out_file', type=str, default='../SyncedPoses.txt')
    args = parser.parse_args()
    sync_intrinsics_and_poses(args.cam_file, args.pose_file, args.out_file)
    