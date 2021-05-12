import os
import pickle
from tqdm import tqdm

from tools.kp_reproject import *
from tools.sync_poses import *

# params
project_path = '/home/sunjiaming/Repositories/NeuralFusion/data/neucon_demo/phone_room_0'
# project_path = '/home/sunjiaming/Repositories/NeuralFusion/data/neucon_demo/conf_0'

def process_data(data_path, data_source='ARKit', window_size=9, min_angle=15, min_distance=0.1, ori_size=(1920, 1440), size=(640, 480)):
    # save image
    print('Extract images from video...')
    video_path = os.path.join(data_path, 'Frames.m4v')
    image_path = os.path.join(data_path, 'images')
    if not os.path.exists(image_path):
        os.mkdir(image_path)
    extract_frames(video_path, out_folder=image_path, size=size)

    # load intrin and extrin
    print('Load intrinsics and extrinsics')
    sync_intrinsics_and_poses(os.path.join(data_path, 'Frames.txt'), os.path.join(data_path, 'ARposes.txt'),
                            os.path.join(data_path, 'SyncedPoses.txt'))

    path_dict = path_parser(data_path, data_source=data_source)
    cam_intrinsic_dict = load_camera_intrinsic(
        path_dict['cam_intrinsic'], data_source=data_source)

    for k, v in tqdm(cam_intrinsic_dict.items(), desc='Processing camera intrinsics...'):
        cam_intrinsic_dict[k]['K'][0, :] /= (ori_size[0] / size[0])
        cam_intrinsic_dict[k]['K'][1, :] /= (ori_size[1] / size[1])
    cam_pose_dict = load_camera_pose(
        path_dict['camera_pose'], data_source=data_source)

    # save_intrinsics_extrinsics
    if not os.path.exists(os.path.join(data_path, 'poses')):
        os.mkdir(os.path.join(data_path, 'poses'))
    for k, v in tqdm(cam_pose_dict.items(), desc='Saving camera extrinsics...'):
        np.savetxt(os.path.join(data_path, 'poses', '{}.txt'.format(k)), v, delimiter=' ')

    if not os.path.exists(os.path.join(data_path, 'intrinsics')):
        os.mkdir(os.path.join(data_path, 'intrinsics'))
    for k, v in tqdm(cam_intrinsic_dict.items(), desc='Saving camera intrinsics...'):
        np.savetxt(os.path.join(data_path, 'intrinsics', '{}.txt'.format(k)), v['K'], delimiter=' ')

    # generate fragment
    fragments = []

    all_ids = []
    ids = []
    count = 0
    last_pose = None

    for id in tqdm(cam_intrinsic_dict.keys(), desc='Keyframes selection...'):
        cam_intrinsic = cam_intrinsic_dict[id]
        cam_pose = cam_pose_dict[id]

        if count == 0:
            ids.append(id)
            last_pose = cam_pose
            count += 1
        else:
            angle = np.arccos(
                ((np.linalg.inv(cam_pose[:3, :3]) @ last_pose[:3, :3] @ np.array([0, 0, 1]).T) * np.array(
                    [0, 0, 1])).sum())
            dis = np.linalg.norm(cam_pose[:3, 3] - last_pose[:3, 3])
            if angle > (min_angle / 180) * np.pi or dis > min_distance:
                ids.append(id)
                last_pose = cam_pose
                # Compute camera view frustum and extend convex hull
                count += 1
                if count == window_size:
                    all_ids.append(ids)
                    ids = []
                    count = 0

    # save fragments
    for i, ids in enumerate(tqdm(all_ids, desc='Saving fragments file...')):
        poses = []
        intrinsics = []
        for id in ids:
            # Moving down the X-Y plane in the ARKit coordinate to meet the training settings in ScanNet.
            cam_pose_dict[id][2, 3] += 1.5
            poses.append(cam_pose_dict[id])
            intrinsics.append(cam_intrinsic_dict[id]['K'])
        fragments.append({
            'scene': data_path.split('/')[-1],
            'fragment_id': i,
            'image_ids': ids,
            'extrinsics': poses,
            'intrinsics': intrinsics
        })

    with open(os.path.join(data_path, 'fragments.pkl'), 'wb') as f:
        pickle.dump(fragments, f)

if __name__ == '__main__':
    process_data(project_path)