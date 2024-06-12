from flask import Flask, request
import socket
import time
import os
import shutil
import flask
import numpy as np
import pyvista as pv
from pathlib import Path
import cv2
import open3d as o3d
import copy
from scipy.ndimage import binary_dilation
import skimage as ski
import imageio.v3 as iio
import struct
import torch
import facer
import multiprocessing
import ast
from tqdm import tqdm


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False
global timestamp
global prev_mat


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])

    target_temp = copy.deepcopy(target)
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation)  # source to target translation
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp_registration(source_path, target_path):
    # source point cloud
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source_path)
    # source_pc.points = source_path.points

    # target point cloud
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target_path)

    try:
        source_pc_prev = source_pc.transform(np.transpose(prev_mat))
        # print(np.transpose(prev_mat))
        # save_ply("source_previous.ply", source_pc_prev.points)
        # save_ply("target_after.ply", target_pc.points)

        # draw_registration_result(source_pc, target_pc, np.transpose(prev_mat))

        # point-to-point ICP registration
        threshold = 100
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pc, target_pc, threshold, np.transpose(prev_mat),
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))

        # draw_registration_result(source_pc_prev, target_pc, reg_p2p.transformation)

        source_pc_final = source_pc_prev.transform(reg_p2p.transformation)
        # save_ply("source_after.ply", source_pc_final.points)

    except Exception as e:
        print(e)
        # save_ply("target_before.ply", target_pc.points)
        # save_ply("source_initial.ply", source_pc.points)
        # initial transform matrix
        trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]])
        # draw_registration_result(source_pc, target_pc, trans_init)
        # point-to-point ICP registration
        threshold = 100
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source_pc, target_pc, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
        # draw_registration_result(source_pc, target_pc, reg_p2p.transformation)

    # source_pc_icp = source_pc.transform(reg_p2p.transformation)
    # o3d.io.write_point_cloud("output/icp.ply", source_pc_icp, write_ascii=True)

    return reg_p2p.transformation


def preprocess_point_cloud(path, voxel_size):
    print(":: Downsample with a voxel size %.3f." % voxel_size)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(path)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)

    '''
    # FGR: feature matching
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh)
    '''
    #  RANSAC: feature matching
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))

    source_pc_icp = source_down.transform(result.transformation)

    return source_pc_icp, result.transformation


def translate(gt_arr, scan_arr):
    gt_x, gt_y, gt_z = calc_mean(gt_arr)
    bf_x, bf_y, bf_z = calc_mean(scan_arr)

    trans_x = bf_x - gt_x
    trans_y = bf_y - gt_y
    trans_z = bf_z - gt_z

    trans_mat = np.array([[1, 0, 0, -trans_x], [0, 1, 0, -trans_y], [0, 0, 1, -trans_z], [0, 0, 0, 1]])

    after = []
    for xyz in scan_arr:
        after.append([float(xyz[0]) - trans_x, float(xyz[1]) - trans_y, float(xyz[2]) - trans_z])

    return np.array(after), trans_mat


def calc_mean(arr):
    mean_x = 0
    mean_y = 0
    mean_z = 0

    for xyz in arr:
        mean_x += float(xyz[0])
        mean_y += float(xyz[1])
        mean_z += float(xyz[2])

    mean_x /= len(arr)
    mean_y /= len(arr)
    mean_z /= len(arr)

    return mean_x, mean_y, mean_z


def teeth_detection(PV_img):
    '''
    PV_img: jpg 형태의 RGB 이미지 경로
    '''
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    image = facer.hwc2bchw(facer.read_hwc(PV_img)).to(device=device)  # image: 1 x 3 x h x w

    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    with torch.inference_mode():
        faces = face_detector(image)

    face_parser = facer.face_parser('farl/lapa/448', device=device)  # optional "farl/celebm/448"
    with torch.inference_mode():
        faces = face_parser(image, faces)

    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)  # nfaces x nclasses x h x w
    n_classes = seg_probs.size(1)
    vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    # print(type(faces))
    # print(faces)

    facer.show_bhw(vis_img)
    facer.show_bchw(facer.draw_bchw(image, faces))

    vis_array = vis_img.cpu().numpy()
    # print(type(vis_array))

    # plt.imshow(vis_array)
    # plt.show()
    # unique = np.unique(vis_array)
    # print(unique)

    new_vis_array = vis_array.squeeze()
    # plt.imshow(new_vis_array)
    # plt.show()

    # imageio.imwrite('label_map.jpg', vis_array)
    return new_vis_array


def seg_only_face(map, point_x, point_y):
    new_map = np.squeeze(map)
    face_idx = np.nonzero(new_map)
    # print(face_idx[0])
    # print(face_idx[1])
    x = round(point_x)
    y = round(point_y)
    if y in face_idx[0]:
        if x in face_idx[1]:
            # print('yes')
            return True
        else:
            return False
    else:
        return False


def check_points_in_label(shared_list, rvec, tvec, intrinsic_matrix, width, label_map, point):
    temp_xy, _ = cv2.projectPoints(point, rvec, tvec, intrinsic_matrix, None)
    temp_xy = np.squeeze(temp_xy, axis=0)
    temp_xy[0, 0] = width - temp_xy[0, 0]
    temp_xy = np.floor(temp_xy.astype(int))

    if seg_only_face(label_map, temp_xy[0, 0], temp_xy[0, 1]):
        shared_list.append(point)
    else:
        pass


def project_on_pv(points, pv_img, pv2world_transform, focal_length, principal_point, label_map, cam2world_transform):
    height, width, _ = pv_img.shape

    homog_points = np.hstack((points, np.ones(len(points)).reshape((-1, 1))))
    # print(f'point shape is {points.shape} and homog point shape is {homog_points.shape}')
    world2pv_transform = np.linalg.inv(pv2world_transform)
    points_pv = (world2pv_transform @ homog_points.T).T[:, :3]

    intrinsic_matrix = np.array([[focal_length[0], 0, width - principal_point[0]], [
        0, focal_length[1], principal_point[1]], [0, 0, 1]])
    rvec = np.zeros(3)
    tvec = np.zeros(3)

    # box 안에 있는 point들 추려보기
    face_points = []
    # print(f'point cloud size is {len(points_pv)}')

    # 여기서부터 multiprocessing 적용 코드
    manager = multiprocessing.Manager()
    shared_list = manager.list()

    multiprocess_pool = multiprocessing.Pool(multiprocessing.cpu_count())
    for point in tqdm(points_pv):
        multiprocess_pool.apply_async(
            check_points_in_label(shared_list,
                                  rvec,
                                  tvec,
                                  intrinsic_matrix,
                                  width,
                                  label_map,
                                  point)
        )
    multiprocess_pool.close()
    multiprocess_pool.join()

    # for point in tqdm(points_pv):
    #     temp_xy, _ = cv2.projectPoints(point, rvec, tvec, intrinsic_matrix, None)
    #     temp_xy = np.squeeze(temp_xy, axis=0)
    #     temp_xy[0, 0] = width - temp_xy[0, 0]
    #     temp_xy = np.floor(temp_xy.astype(int))
    #
    #     if seg_only_face(label_map, temp_xy[0, 0], temp_xy[0, 1]):
    #         face_points.append(point)
    #     else:
    #         continue
    face_points = np.array(shared_list)
    # face_points = np.array(face_points)
    homog_face_points = np.hstack((face_points, np.ones(len(face_points)).reshape((-1, 1))))
    face_points_world = (pv2world_transform @ homog_face_points.T).T[:, :3]

    # save_ply('only_face.ply', face_points_world, rgb=None, cam2world_transform=cam2world_transform)

    return face_points_world


def write_bytes_to_png(bytes_path, width, height):
    print(".", end="", flush=True)

    output_path = bytes_path.replace('bytes', 'png')
    if os.path.exists(output_path):
        return

    image = []
    with open(bytes_path, 'rb') as f:
        image = np.frombuffer(f.read(), dtype=np.uint8)

    image = image.reshape((height, width, 4))

    new_image = image[:, :, :3]
    cv2.imwrite(output_path, new_image)

    # Delete '*.bytes' files
    bytes_path.unlink()


def get_width_and_height(path):
    with open(path) as f:
        lines = f.readlines()
    (_, _, width, height) = lines[0].split(',')

    return (int(width), int(height))


def convert_images(folder):
    p = multiprocessing.Pool(multiprocessing.cpu_count())
    img_folder = 'PV'
    pv_path = list(Path(folder).glob('*pv.txt'))
    assert len(list(pv_path)) == 1
    (width, height) = get_width_and_height(pv_path[0])

    paths = (Path(folder) / Path(img_folder)).glob('*bytes')
    print("Processing images")
    for path in paths:
        p.apply_async(write_bytes_to_png, (str(path), width, height))
    p.close()
    p.join()


def save_single_pcloud(path,
                       folder,
                       lut,
                       focal_lengths,
                       principal_point,
                       rig2world_transforms,
                       rig2cam,
                       pv_timestamps,
                       pv2world_transforms,
                       discard_no_rgb,
                       clamp_min,
                       clamp_max,
                       depth_path_suffix,
                       ):
    # extract the timestamp for this frame
    timestamp = extract_timestamp(path.name.replace(depth_path_suffix, ''))
    # load depth img
    img = cv2.imread(str(path), -1)
    height, width = img.shape
    assert len(lut) == width * height

    # Clamp values if requested
    if clamp_min > 0 and clamp_max > 0:
        # Convert crop values to mm
        clamp_min = clamp_min * 1000.
        clamp_max = clamp_max * 1000.
        # Clamp depth values
        img[img < clamp_min] = 0
        img[img > clamp_max] = 0

    if rig2world_transforms and (timestamp in rig2world_transforms):
        # if we have pv, get vertex colors
        # get the pv frame which is closest in time
        target_id = match_timestamp(timestamp, pv_timestamps)
        pv_ts = pv_timestamps[target_id]
        rgb_path = str(Path(folder) / Path('PV') / Path(f'{pv_ts}.png'))

        # segmentation array
        rgb_seg_array = teeth_detection(rgb_path)
        # print(type(rgb_seg_array))
        # print(rgb_seg_array.shape)
        rgb_seg_array[rgb_seg_array > 0] = 1
        face_mask = rgb_seg_array
        # face_mask = rgb_seg_array[rgb_seg_array>0] # face label mask 생성 (1504 x 846)
        # face_mask = np.array(face_mask)
        # 320 x 180 형태로 face label mask를 resizing 진행
        # print(type(face_mask))
        # print(face_mask.shape)
        pgmsize_face_mask = cv2.resize(face_mask, (0, 0), fx=1 / (4.7), fy=1 / (4.7), interpolation=cv2.INTER_NEAREST)
        # face에 해당하는 index를 추출
        # print(pgmsize_face_mask.shape)
        pgmsize_face_index = np.where(pgmsize_face_mask > 0)
        # 320 x 288 size의 face에 해당하는 값만 1인 mask를 생성
        raw_mask = np.zeros_like(img)
        raw_mask[pgmsize_face_index] = 1
        # np.pad(raw_mask, 'constant', constant_values=1)

        # 초기에 확보한 pgm 데이터에 face mask를 곱하여 face에 해당하는 pgm만 냅둠
        face_pgm = img * raw_mask
        
        # pixel dilation 방식
        structure = np.array([[False, False, True],
                              [False, True, True],
                              [False, False, True]])
        dilated_face_mask = binary_dilation(raw_mask, structure=structure, iterations=8)

        '''
        # image translation 방식
        pgm_mask = cv2.imread('inputs/pgm_mask.png', cv2.IMREAD_GRAYSCALE)

        best_iou = 0
        best_iter = 0
        copied_mask = raw_mask

        for iter in range(1, 20):
            num_rows, num_cols = copied_mask.shape[:2]
            translation_matrix = np.float32([[1, 0, 1], [0, 1, 0]])
            copied_mask = cv2.warpAffine(copied_mask, translation_matrix, (num_cols, num_rows))

            intersection = np.logical_and(pgm_mask, copied_mask)
            union = np.logical_or(pgm_mask, copied_mask)
            iou_score = np.sum(intersection) / np.sum(union)

            if iou_score > best_iou:
                best_iou = iou_score
                best_iter = iter

        print("best iteration was " + str(best_iter))
        print("best IOU was " + str(best_iou))

        num_rows, num_cols = raw_mask.shape[:2]
        translation_matrix = np.float32([[1, 0, best_iter], [0, 1, 0]])
        dilated_face_mask = cv2.warpAffine(raw_mask, translation_matrix, (num_cols, num_rows))
        '''

        dilated_face_pgm = img * dilated_face_mask

        # fig = plt.figure(figsize=(20,8))
        # fig.add_subplot(1, 8, 1)
        # plt.imshow(rgb_seg_array)
        # plt.title('Face segmentation image', fontsize=8)

        # fig.add_subplot(1, 8, 2)
        # plt.imshow(face_mask)
        # plt.title('Face mask image', fontsize=8)

        # fig.add_subplot(1, 8, 3)
        # plt.imshow(pgmsize_face_mask)
        # plt.title('Resized to pgm size (320 x 180)', fontsize=8)

        # fig.add_subplot(1, 8, 4)
        # plt.imshow(raw_mask)
        # plt.title('Face mask size (320 x 288)', fontsize=8)

        # fig.add_subplot(1, 8, 5)
        # plt.imshow(face_pgm)
        # plt.title('Final face mask applied image', fontsize=8)

        # fig.add_subplot(1, 8, 6)
        # plt.imshow(dilated_face_mask)
        # plt.title('Dilated face mask', fontsize=8)

        # fig.add_subplot(1, 8, 7)
        # plt.imshow(dilated_face_pgm)
        # plt.title('Dilated face mask applied image', fontsize=8)

        # fig.add_subplot(1, 8, 8)
        # plt.imshow(img)
        # plt.title('Original pgm image', fontsize=8)
        # plt.show()

        # Get xyz points in camera space
        # points = get_points_in_cam_space(img, lut)
        face_points = get_points_in_cam_space(dilated_face_pgm, lut)

        # if we have the transform from rig to world for this frame,
        # then put the point clouds in world space
        rig2world = rig2world_transforms[timestamp]
        # print('Transform found for timestamp %s' % timestamp)
        # xyz, cam2world_transform = cam2world(points, rig2cam, rig2world)
        face_xyz, _ = cam2world(face_points, rig2cam, rig2world)
        rgb = None
        # save_ply('test_pgm_point_cloud.ply', xyz)
        # save_ply('test_face_pgm_point_cloud.ply', face_xyz)
    #     # Project from depth to pv going via world space
    #     pv_img = cv2.imread(rgb_path)
    #     face_points_world = project_on_pv(
    #         xyz, pv_img, pv2world_transforms[target_id],
    #         focal_lengths[target_id], principal_point,
    #         rgb_seg_array, cam2world_transform)

    # face_cloud = pv.PolyData(face_points_world)
    face_cloud = pv.PolyData(face_xyz)
    return face_cloud


def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])


def load_extrinsics(extrinsics_path):
    assert Path(extrinsics_path).exists()
    mtx = np.loadtxt(str(extrinsics_path), delimiter=',').reshape((4, 4))
    return mtx


def get_points_in_cam_space(img, lut):
    img = np.tile(img.flatten().reshape((-1, 1)), (1, 3))
    points = img * lut
    remove_ids = np.where(np.sqrt(np.sum(points ** 2, axis=1)) < 1e-6)[0]
    points = np.delete(points, remove_ids, axis=0)
    points /= 1000.
    return points


def cam2world(points, rig2cam, rig2world):
    homog_points = np.hstack((points, np.ones((points.shape[0], 1))))
    cam2world_transform = rig2world @ np.linalg.inv(rig2cam)
    world_points = cam2world_transform @ homog_points.T
    return world_points.T[:, :3], cam2world_transform


def extract_timestamp(path):
    return int(path.split('.')[0])


def load_lut(lut_filename):
    with open(lut_filename, mode='rb') as depth_file:
        lut = np.frombuffer(depth_file.read(), dtype="f")
        lut = np.reshape(lut, (-1, 3))
    return lut


def load_pv_data(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()

    # The first line contains info about the intrinsics.
    # The following lines (one per frame) contain timestamp, focal length and transform PVtoWorld
    n_frames = len(lines) - 1
    frame_timestamps = np.zeros(n_frames, dtype=np.longlong)
    focal_lengths = np.zeros((n_frames, 2))
    pv2world_transforms = np.zeros((n_frames, 4, 4))

    intrinsics_ox, intrinsics_oy, \
        intrinsics_width, intrinsics_height = ast.literal_eval(lines[0])

    for i_frame, frame in enumerate(lines[1:]):
        # Row format is
        # timestamp, focal length (2), transform PVtoWorld (4x4)
        frame = frame.split(',')
        frame_timestamps[i_frame] = int(frame[0])
        focal_lengths[i_frame, 0] = float(frame[1])
        focal_lengths[i_frame, 1] = float(frame[2])
        pv2world_transforms[i_frame] = np.array(frame[3:20]).astype(float).reshape((4, 4))

    return (frame_timestamps, focal_lengths, pv2world_transforms,
            intrinsics_ox, intrinsics_oy, intrinsics_width, intrinsics_height)


def load_rig2world_transforms(path):
    transforms = {}
    with open(path, "r") as f:
        lines = [l.strip() for l in f.readlines()]
    for line in lines:
        value = line.split(",")
        timestamp = int(value[0])
        transform_entries = [float(v) for v in value[1:]]
        transform = np.array(transform_entries).reshape((4, 4))
        transforms[timestamp] = transform
    return transforms


def save_pclouds(folder,
                 sensor_name=None,
                 save_in_cam_space=False,
                 discard_no_rgb=False,
                 clamp_min=0.,
                 clamp_max=0.,
                 depth_path_suffix=''
                 ):
    print("")
    print("Saving point clouds")
    sensor_name = "Depth Long Throw"
    calib = r'{}_lut.bin'.format(sensor_name)
    extrinsics = r'{}_extrinsics.txt'.format(sensor_name)
    rig2world = r'{}_rig2world.txt'.format(sensor_name)
    calib_path = Path(folder) / Path(calib)
    rig2campath = Path(folder) / Path(extrinsics)
    rig2world_path = Path(folder) / Path(rig2world) if not save_in_cam_space else ''

    pv_info_path = sorted(Path(folder).glob(r'*pv.txt'))
    (pv_timestamps, focal_lengths, pv2world_transforms, ox,
     oy, _, _) = load_pv_data(list(pv_info_path)[0])
    principal_point = np.array([ox, oy])

    # lookup table to extract xyz from depth
    lut = load_lut(calib_path)

    # from camera to rig space transformation (fixed)
    rig2cam = load_extrinsics(rig2campath)

    # from rig to world transformations (one per frame)
    rig2world_transforms = load_rig2world_transforms(
        rig2world_path) if rig2world_path != '' and Path(rig2world_path).exists() else None
    depth_path = Path(folder) / Path(sensor_name)
    depth_path.mkdir(exist_ok=True)

    # Depth path suffix used for now only if we load masked AHAT
    depth_paths = sorted(depth_path.glob('*[0-9]{}.pgm'.format(depth_path_suffix)))
    assert len(list(depth_paths)) > 0

    for path in depth_paths:
        face_pc = save_single_pcloud(path,
                                     folder,
                                     lut,
                                     focal_lengths,
                                     principal_point,
                                     rig2world_transforms,
                                     rig2cam,
                                     pv_timestamps,
                                     pv2world_transforms,
                                     discard_no_rgb,
                                     clamp_min,
                                     clamp_max,
                                     depth_path_suffix,
                                     )
    return face_pc


def save_ply(output_path, points, rgb=None, cam2world_transform=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if rgb is not None:
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    pcd.estimate_normals()
    if cam2world_transform is not None:
        # Camera center
        camera_center = (cam2world_transform) @ np.array([0, 0, 0, 1])
        o3d.geometry.PointCloud.orient_normals_towards_camera_location(pcd, camera_center[:3])

    o3d.io.write_point_cloud(output_path, pcd)


def add_header(path):
    NonHeader = open(path, 'rb')
    lines = NonHeader.readlines()
    lines.insert(0, b'65535\n')
    lines.insert(0, b'320 288\n')
    lines.insert(0, b'P5\n')
    NonHeader.close()

    WithHeader = open(path, 'wb')
    WithHeader.writelines(lines)
    WithHeader.close()


def remove_background(path):
    img = iio.imread(uri=path)
    t = ski.filters.threshold_mean(img)

    if path[-6:-4] == 'ab':
        binary_mask = img > t
    else:
        binary_mask = img < t

    selection = img.copy()
    selection[~binary_mask] = 0

    iio.imwrite(uri=path, image=selection)


@app.route("/depth", methods=['POST'])
def print_depth():
    try:
        global timestamp
        timestamp = str(int(time.time()))

        shutil.rmtree('inputs/Depth Long Throw', ignore_errors=True)
        os.mkdir('inputs/Depth Long Throw')

        file1 = open('inputs/Depth Long Throw/' + str(timestamp) + ".pgm", 'wb')
        file2 = open('inputs/Depth Long Throw/' + str(timestamp) + "_ab.pgm", 'wb')
        file3 = open("inputs/Depth Long Throw_extrinsics.txt", 'w')
        file4 = open('inputs/Depth Long Throw_rig2world.txt', 'w')

        for f in request.files:
            if f == "long":
                file1.write(request.files[f].read())
                file1.close()

            elif f == "ab":
                file2.write(request.files[f].read())
                file2.close()

            elif f == "r2c":
                line1 = []
                rows = request.files[f].read().decode('utf-8').split('\n')

                row1 = rows[1].split(',')
                row2 = rows[2].split(',')
                row3 = rows[3].split(',')
                row4 = rows[4].split(',')

                line1.append(row1[0])
                line1.append(row2[0])
                line1.append(row3[0])
                line1.append(row4[0])

                line1.append(row1[1])
                line1.append(row2[1])
                line1.append(row3[1])
                line1.append(row4[1])

                line1.append(row1[2])
                line1.append(row2[2])
                line1.append(row3[2])
                line1.append(row4[2])

                line1.append(row1[3])
                line1.append(row2[3])
                line1.append(row3[3])
                line1.append(row4[3])

                file3.write(','.join(line1) + '\n')
                file3.close()

            elif f == 'r2w':
                line2 = []
                rows = request.files[f].read().decode('utf-8').split('\n')

                row1 = rows[1].split(',')
                row2 = rows[2].split(',')
                row3 = rows[3].split(',')
                row4 = rows[4].split(',')

                line2.append(row1[0])
                line2.append(row2[0])
                line2.append(row3[0])
                line2.append(row4[0])

                line2.append(row1[1])
                line2.append(row2[1])
                line2.append(row3[1])
                line2.append(row4[1])

                line2.append(row1[2])
                line2.append(row2[2])
                line2.append(row3[2])
                line2.append(row4[2])

                line2.append(row1[3])
                line2.append(row2[3])
                line2.append(row3[3])
                line2.append(row4[3])

                file4.write(str(timestamp) + ',' + ','.join(line2) + '\n')
                file4.close()

    except Exception as err:
        print("ko:", err)

    add_header('inputs/Depth Long Throw/' + str(timestamp) + ".pgm")
    add_header('inputs/Depth Long Throw/' + str(timestamp) + "_ab.pgm")
    remove_background('inputs/Depth Long Throw/' + str(timestamp) + ".pgm")
    remove_background('inputs/Depth Long Throw/' + str(timestamp) + "_ab.pgm")

    return "ok"


@app.route("/send", methods=['GET'])
def download_model():
    try:
        file = open('output/icp.obj', 'r')
        lines = file.readlines()

        mesh = {}

        verts = []
        faces = []

        for line in lines:
            if line.split(' ')[0] == 'v':
                verts.append(line.split(' ')[1])
                verts.append(line.split(' ')[2])
                verts.append(line.split(' ')[3])
            elif line.split(' ')[0] == 'f':
                faces.append(int(line.split(' ')[1].split('//')[0]) - 1)
                faces.append(int(line.split(' ')[2].split('//')[0]) - 1)
                faces.append(int(line.split(' ')[3].split('//')[0]) - 1)

        mesh['verts'] = verts
        mesh['faces'] = faces

        response_list = mesh
        response_dict = dict(response_list)

    except Exception as err:
        print("ko:", err)

    return flask.jsonify(response_dict)


@app.route("/etcf", methods=['POST'])
def print_etcf():
    try:
        file = open("inputs/inputs_pv.txt", 'w')
        line1 = []
        line2 = []

        for f in request.files:
            if f == "px" or f == "py":
                line1.append(str(struct.unpack('f', request.files[f].read())[0]))

            elif f == "IntW" or f == "IntH":
                line1.append(str(int.from_bytes(request.files[f].read(), 'little')))

            if f == "ts":
                line2.append(str(timestamp))

            elif f == "fx" or f == "fy":
                line2.append(str(struct.unpack('f', request.files[f].read())[0]))

            elif f[0] == 'M':
                line2.append(str(struct.unpack('f', request.files[f].read())[0]))

        file.write(','.join(line1) + '\n')
        file.write(','.join(line2) + '\n')
        file.close()

    except Exception as err:
        print("ko:", err)

    return "ok"


@app.route("/rgb", methods=['POST'])
def print_rgb():
    try:
        shutil.rmtree('inputs/PV', ignore_errors=True)
        os.mkdir('inputs/PV')

        for f in request.files:
            data = request.files[f].read()
            file = open('inputs/PV/' + str(timestamp) + '.bytes', 'wb')
            file.write(data)
            file.close()

    except Exception as err:
        print("ko:", err)

    path = './inputs'
    convert_images(path)
    HL2_pc = save_pclouds(path)

    scan_data = pv.read('output/Hong-face-scan_v3.stl')
    # scan_data = pv.read('output/HWJ_ST.obj')
    before = scan_data.points

    global prev_mat

    try:
        print(prev_mat is None)
        icp_transformation = icp_registration(before, HL2_pc.points)
        trans_mat = np.transpose(np.matmul(icp_transformation, np.transpose(prev_mat)))
        prev_mat = trans_mat

    except Exception as err:
        after, trans_mat = translate(HL2_pc.points, before)
        '''
        voxel_size = 0.0005
        source_down, source_fpfh = preprocess_point_cloud(before, voxel_size)
        target_down, target_fpfh = preprocess_point_cloud(HL2_pc.points, voxel_size)

        after, trans_mat = execute_global_registration(source_down, target_down,
                                                       source_fpfh, target_fpfh,
                                                       voxel_size)
        '''
        icp_transformation = icp_registration(after, HL2_pc.points)
        trans_mat = np.transpose(np.matmul(icp_transformation, trans_mat))
        prev_mat = trans_mat

        print("ko:", err)

    mat2str = []
    for row in trans_mat:
        for column in row:
            mat2str.append(str(column))

    return ','.join(mat2str)


if __name__ == "__main__":
    # path = './inputs'
    # convert_images(path)
    # HL2_pc = save_pclouds(path)

    IP = socket.gethostbyname(socket.gethostname())
    # app.run(host="192.168.0.45")
    app.run(host='0.0.0.0', debug=False)