from flask import Flask, request, jsonify, current_app, g as app_ctx
import socket
import time
import struct
import tarfile
import os
import shutil
import flask
import numpy as np
import cv2
import pyvista as pv
import multiprocessing
from pathlib import Path
import cv2
import open3d as o3d
import copy


app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.paint_uniform_color([1, 0.706, 0])

    target_temp = copy.deepcopy(target)
    target_temp.paint_uniform_color([0, 0.651, 0.929])

    source_temp.transform(transformation) # source to target translation
    o3d.visualization.draw_geometries([source_temp, target_temp])


def icp_registration(source_path, target_path):
    # source point cloud
    source_pc = o3d.geometry.PointCloud()
    source_pc.points = o3d.utility.Vector3dVector(source_path)

    # target point cloud
    target_pc = o3d.geometry.PointCloud()
    target_pc.points = o3d.utility.Vector3dVector(target_path)

    # initial transform matrix
    trans_init = np.asarray([[1.0, 0.0, 0.0, 0.0],
                             [0.0, 1.0, 0.0, 0.0],
                             [0.0, 0.0, 1.0, 0.0],
                             [0.0, 0.0, 0.0, 1.0]])
    draw_registration_result(source_pc, target_pc, trans_init)
    # point-to-point ICP registration
    threshold = 100
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_pc, target_pc, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=2000))
    draw_registration_result(source_pc, target_pc, reg_p2p.transformation)

    source_pc_icp = source_pc.transform(reg_p2p.transformation)
    o3d.io.write_point_cloud("output/icp.stl", source_pc_icp, write_ascii=True)

    return reg_p2p.transformation


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


def face_detection(ply_path):
    '''
    ply_path: point cloud가 담긴 변수
    return: (0,0,0)에서 가장 가까운 point 기준 반경 15 cm 이내 point들만 남기는 코드
    '''
    # points = pv.read(ply_path)
    cloud = pv.PolyData(ply_path)
    closest_index = cloud.find_closest_point((0, 0, 0))
    ref_point = cloud.points[closest_index]
    new_face_points = []
    for point in cloud.points:
        dist = np.linalg.norm(ref_point - point)
        if dist < 0.15:
            new_face_points.append(point)
        else:
            continue

    # memory clean
    cloud.clear_point_data()

    face_cloud = pv.PolyData(new_face_points)
    # face_cloud.plot()
    face_cloud.save('test_ply.ply')
    #new_points = pv.wrap(face_cloud.points)
    #surf = new_points.reconstruct_surface()
    #surf.save('test_obj.stl')

    return face_cloud


def save_single_pcloud(path,
                       save_in_cam_space,
                       lut,
                       rig2world_transforms,
                       rig2cam,
                       discard_no_rgb,
                       clamp_min,
                       clamp_max,
                       depth_path_suffix
                       ):
    print(".", end="", flush=True)

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

    # Get xyz points in camera space
    points = get_points_in_cam_space(img, lut)

    if rig2world_transforms and (timestamp in rig2world_transforms):
        # if we have the transform from rig to world for this frame,
        # then put the point clouds in world space
        rig2world = rig2world_transforms[timestamp]
        xyz, cam2world_transform = cam2world(points, rig2cam, rig2world)

    face_pc = face_detection(xyz)
    return face_pc


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
    # print(rig2cam)
    # print(rig2world)
    # print(cam2world_transform)
    world_points = cam2world_transform @ homog_points.T
    return world_points.T[:, :3], cam2world_transform


def extract_timestamp(path):
    return int(path.split('.')[0])


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


def load_lut(lut_filename):
    with open(lut_filename, mode='rb') as depth_file:
        lut = np.frombuffer(depth_file.read(), dtype="f")
        lut = np.reshape(lut, (-1, 3))
    return lut


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
                                     save_in_cam_space,
                                     lut,
                                     rig2world_transforms,
                                     rig2cam,
                                     discard_no_rgb,
                                     clamp_min,
                                     clamp_max,
                                     depth_path_suffix
                                     )

    return face_pc


@app.route("/depth", methods=['POST'])
def print_depth():
    try:
        shutil.rmtree('inputs/Depth Long Throw', ignore_errors=True)
        os.mkdir('inputs/Depth Long Throw')
        ts = int(time.time())
        filename = str(ts)
        file1 = open('inputs/Depth Long Throw/' + filename + ".pgm", 'wb')
        file2 = open('inputs/Depth Long Throw/' + filename + "_ab.pgm", 'wb')
        file3 = open("inputs/Depth Long Throw_extrinsics.txt", 'w')
        file4 = open('inputs/Depth Long Throw_rig2world.txt', 'w')

        for f in request.files:
            if f == "long":
                file1.write(request.files[f].read())
                file1.close()

                NonHeader = open('inputs/Depth Long Throw/' + filename + ".pgm", 'rb')
                lines = NonHeader.readlines()
                lines.insert(0, b'65535\n')
                lines.insert(0, b'320 288\n')
                lines.insert(0, b'P5\n')
                NonHeader.close()

                WithHeader = open('inputs/Depth Long Throw/' + filename + ".pgm", 'wb')
                WithHeader.writelines(lines)
                WithHeader.close()

                # img = np.frombuffer(request.files[f].read(), np.uint16).reshape(288, 320)
                # cv2.imwrite('inputs/Depth Long Throw/' + filename + ".png", (img * 5000).astype(np.uint16))

            elif f == "ab":
                file2.write(request.files[f].read())
                file2.close()

                NonHeader = open('inputs/Depth Long Throw/' + filename + "_ab.pgm", 'rb')
                lines = NonHeader.readlines()
                lines.insert(0, b'65535\n')
                lines.insert(0, b'320 288\n')
                lines.insert(0, b'P5\n')
                NonHeader.close()
                
                WithHeader = open('inputs/Depth Long Throw/' + filename + "_ab.pgm", 'wb')
                WithHeader.writelines(lines)
                WithHeader.close()

                # img = np.frombuffer(request.files[f].read(), np.uint16).reshape(288, 320)
                # cv2.imwrite('inputs/Depth Long Throw/' + filename + "_ab.png", img)

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

            elif f =='r2w':
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

                file4.write(str(ts) + ',' + ','.join(line2) + '\n')
                file4.close()

    except Exception as err:
        print("ko:", err)

    path = './inputs'
    HL2_pc = save_pclouds(path)
    scan_data = pv.read('output/Hong-face-scan_v3.stl')
    before = scan_data.points
    after, trans_mat = translate(HL2_pc.points, before)

    icp_transformation = icp_registration(after, HL2_pc.points)
    trans_mat = np.transpose(np.matmul(icp_transformation, trans_mat))

    # rot_mat_y_180 = [[-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]]
    # rot_mat_x_90 = [[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
    # rot_mat_z_90 = [[0, -1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    #rot_mat_x__90 = [[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]]
    #rot_mat_z__90 = [[0, 1, 0, 0], [-1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    #flip_y = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
    #rot_mat_xz = np.matmul(rot_mat_z__90, rot_mat_x__90)
    #rot_mat = np.matmul(flip_y, rot_mat_xz)
    #final_mat = np.matrix.transpose(np.matmul(rot_mat, trans_mat))

    mat2str = []
    for row in trans_mat:
        for column in row:
            mat2str.append(str(column))

    return ','.join(mat2str)


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


@app.route("/test", methods=['GET'])
def download_xt():
    return "0.0220382, 0.962838, 0.26918, 0.0, -0.999679, 0.0245826, -0.00608518, 0.0, -0.0124762, -0.268959, 0.963071, 0.0, -0.0594006, -0.0151882, -0.0179601, 1.0"

@app.route("/test2", methods=['GET'])
def download_txt():
    return "0.053924, -0.954192, 0.294297, 0.0, -0.998123, -0.0429405, 0.0436614, 0.0, -0.0290241, -0.296099, -0.954716, 0.0, -0.0563135, 0.0798842, -0.150462, 1.0"

if __name__ == "__main__":
    IP = socket.gethostbyname(socket.gethostname())
    # app.run(host="192.168.0.45")
    app.run(host='0.0.0.0', debug=True)
