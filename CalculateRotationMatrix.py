import argparse
import yaml
import numpy as np
import open3d as o3d
import struct
import os
import math
from scipy.spatial.transform import Rotation as R


def create_normal_arrows(plane_clouds, normal_vec):
    arrows = []

    for i, vec in enumerate(normal_vec):
        points = np.asarray(plane_clouds[i].points)
        center = np.mean(points, axis=0)

        arrow_length = 10

        # 创建箭头
        arrow = o3d.geometry.TriangleMesh.create_arrow(
            cylinder_radius=0.1,
            cone_radius=1,
            cylinder_height=arrow_length,
            cone_height=3,
            resolution=200,
        )

        # 计算旋转矩阵，使箭头指向法向量的方向
        arrow_direction = vec / np.linalg.norm(vec)
        z_axis = np.array([0, 0, 1])

        # 计算旋转轴和角度
        rotation_axis = np.cross(z_axis, arrow_direction)
        rotation_angle = np.arccos(np.dot(z_axis, arrow_direction))

        if np.linalg.norm(rotation_axis) > 0:
            rotation_axis /= np.linalg.norm(rotation_axis)
            rotation_matrix = o3d.geometry.get_rotation_matrix_from_axis_angle(
                rotation_axis * rotation_angle
            )
            arrow.rotate(rotation_matrix)

        # 设置箭头的位置
        arrow.translate(center)

        # 设置箭头颜色与平面点云颜色一致
        plane_color = np.mean(np.asarray(plane_clouds[i].colors), axis=0)
        arrow.paint_uniform_color(plane_color)

        arrows.append(arrow)

    return arrows


def rotate_custom(r):

    initial_translation = np.array([0, 0, 0])  # 初始平移不变
    initial_transformation = np.eye(4)
    initial_transformation[:3, :3] = r
    initial_transformation[:3, 3] = initial_translation

    return initial_transformation


def generate_rotation_matrix(axis, angle_degrees):
    # 将角度转换为弧度
    angle_radians = np.radians(angle_degrees)

    if axis == "x":
        rotation = R.from_euler("x", angle_radians)
    elif axis == "y":
        rotation = R.from_euler("y", angle_radians)
    elif axis == "z":
        rotation = R.from_euler("z", angle_radians)
    else:
        raise ValueError("Axis must be one of 'x', 'y', or 'z'")

    # 返回旋转矩阵
    return rotation.as_matrix()


def rotation_matrix_to_align_with_z(normal_vector):
    """计算将法向量旋转到与z轴垂直的旋转矩阵"""
    # 目标轴：z轴
    target_axis = np.array([0, 0, 1])

    # 归一化法向量和目标轴
    n = normal_vector / np.linalg.norm(normal_vector)
    t = target_axis / np.linalg.norm(target_axis)

    # 计算旋转角度（弧度制）
    angle = np.arccos(np.clip(np.dot(n, t), -1.0, 1.0))

    # 计算旋转轴
    rotation_axis = np.cross(n, t)

    # 如果旋转轴为零向量，说明法向量已经与目标轴平行或反平行
    if np.linalg.norm(rotation_axis) < 1e-8:
        return np.eye(3)  # 返回单位矩阵（无需旋转）

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Rodrigues' rotation formula to compute the rotation matrix
    K = np.array(
        [
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ]
    )
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


def rotation_matrix_to_align_with_y(normal_vector):
    """计算将法向量旋转到与z轴垂直的旋转矩阵"""
    # 目标轴：z轴
    target_axis = np.array([0, 1, 0])

    # 归一化法向量和目标轴
    n = normal_vector / np.linalg.norm(normal_vector)
    t = target_axis / np.linalg.norm(target_axis)

    # 计算旋转角度（弧度制）
    angle = np.arccos(np.clip(np.dot(n, t), -1.0, 1.0))

    # 计算旋转轴
    rotation_axis = np.cross(n, t)

    # 如果旋转轴为零向量，说明法向量已经与目标轴平行或反平行
    if np.linalg.norm(rotation_axis) < 1e-8:
        return np.eye(3)  # 返回单位矩阵（无需旋转）

    rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)

    # Rodrigues' rotation formula to compute the rotation matrix
    K = np.array(
        [
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0],
        ]
    )
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return R


# 定义解析PointCloud2数据的函数
def parse_pointcloud2(data, point_step):
    # 每个点的属性数和总步长
    num_points = len(data) // point_step
    points = []

    # 逐个解析点数据
    for i in range(num_points):
        point_data = data[i * point_step : (i + 1) * point_step]
        x, y, z, intensity = struct.unpack("fffI", point_data[:16])
        points.append((x, y, z, intensity))

    return np.array(points)


def load_point_cloud_from_csv(file_path, color):
    # Load point cloud data, skipping the first row (header)
    data = np.loadtxt(file_path, delimiter=",", dtype=np.float32, skiprows=1)

    # data = data[data[:, 1] < 0]
    # data = data[data[:, 2] > 6]
    # Create point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(data[:, :3])
    pcd.colors = o3d.utility.Vector3dVector(
        np.ones((data.shape[0], 3)) * color
    )  # Add color to point cloud
    return pcd


# 计算夹角
def cal_degree(vector_a, vector_b):
    dot_product = np.dot(vector_a, vector_b)

    # 计算模长
    magnitude_a = np.linalg.norm(vector_a)
    magnitude_b = np.linalg.norm(vector_b)

    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_a * magnitude_b)

    # 计算夹角（弧度）
    theta_radians = np.arccos(cos_theta)

    # 转换为度数
    theta_degrees = np.degrees(theta_radians)

    return theta_degrees


# 使用方法：python CalculateRotationMatrix.py config/asc/fl.yaml
"""
CalculateRotationMatrix.py: 脚本名
config/asc/fl.yaml: ASC的fl雷达配置文件
"""

if __name__ == "__main__":
    # 创建ArgumentParser对象
    parser = argparse.ArgumentParser(description="计算旋转矩阵")

    parser.add_argument("config", type=str, help="配置文件路径")

    # 解析参数
    args = parser.parse_args()

    # 读取配置文件
    with open(args.config, "r") as file:
        config = yaml.safe_load(file)

    # 读取lidar数据
    file_path = config.get("lidar_file")
    name = os.path.splitext(os.path.basename(file_path))[0]

    with open(file_path, "r") as file:
        lines = file.readlines()

    # 从文本中找到`data`字段并提取其中的点数据
    data_line = [line for line in lines if line.startswith("data:")][0]
    data = eval(data_line.split("data:")[1].strip())

    # 解析点数据
    point_step = 32  # 每个点的步长为32字节
    points = parse_pointcloud2(bytearray(data), point_step)

    # 如果你希望保存到一个CSV文件
    csv_file = os.path.splitext(file_path)[0]

    print(csv_file)
    np.savetxt(
        f"{csv_file}.csv", points, delimiter=",", header="x,y,z,intensity", comments=""
    )

    # Define file path
    parsed_lidar_flie = f"{csv_file}.csv"

    pcd = load_point_cloud_from_csv(parsed_lidar_flie, color=[0.1, 0.1, 0.1])  # Gray

    # 下采样点云
    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    # 存储所有的平面模型和对应的平面点云
    planes = []
    plane_clouds = []
    plane_threshold = 0.01  # 距离阈值
    min_plane_points = 750  # 每个平面至少需要的点数
    normal_vec = []  # 保存平面的法向量

    # 循环检测多个平面
    while True:
        # 使用 RANSAC 进行平面分割
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=plane_threshold, ransac_n=3, num_iterations=1000
        )

        # 如果检测到的平面点少于阈值，则终止分割
        if len(inliers) < min_plane_points:
            break

        # 保存平面模型和对应的平面点云
        planes.append(plane_model)
        plane_clouds.append(pcd.select_by_index(inliers))

        # o3d.visualization.draw_geometries([pcd.select_by_index(inliers)])

        # 移除已经分割出的平面点
        pcd = pcd.select_by_index(inliers, invert=True)
        normal_vec.append(plane_model[:3])

    # 可视化结果
    for i, plane_cloud in enumerate(plane_clouds):
        # 给每个平面设置不同的颜色
        color = np.random.rand(3)
        plane_cloud.paint_uniform_color(color)

    # 我们希望找到一个法向量，使得旋转后的法向量与目标轴的夹角最小
    max_angle = 180  # 存储目标轴的的夹角
    normal_vecs = None  # 存储相应的法向量
    plane = None  # 存储相应的法向量

    for i, j in zip(normal_vec, planes):
        
        print(config.get("max_angle_z"))

        if config.get("max_angle_z") == "z":
            aixs = np.array([0, 0, 1])
        elif config.get("max_angle_z") == "y":
            aixs = np.array([0, 1, 0])
        elif config.get("max_angle_z") == "x":
            aixs = np.array([1, 0, 0])
        else:
            pass

        angle = cal_degree(i, aixs)
        if angle < max_angle:
            max_angle = angle
            normal_vecs = i
            plane = j

    if config.get("max_angle_z") == "z":
        rotation_matrix = rotation_matrix_to_align_with_z(normal_vecs)
    elif config.get("max_angle_z") == "y":
        rotation_matrix = rotation_matrix_to_align_with_y(normal_vecs)
    

    fixed_rotate = config.get("fixed_rotate_angle")
    fixed_rotation_matrix = generate_rotation_matrix(fixed_rotate[0], fixed_rotate[1])

    fixed_rotation_matrix = fixed_rotation_matrix @ rotation_matrix

    r_m = fixed_rotation_matrix.flatten()
    formatted_array = "[" + ", ".join(f"{x:.5f}" for x in r_m) + "]"
    print("修正后旋转矩阵:\n", formatted_array)

    # 应用初始变换旋转雷达
    initial_transformation = rotate_custom(fixed_rotation_matrix)

    extra_rotate = config.get("extra_rotate_angle")

    extra_rotation_matrix = generate_rotation_matrix(extra_rotate[0], extra_rotate[1])

    initial_transformation = extra_rotation_matrix @ initial_transformation[:3, :3]

    print(f"增加旋转角度{extra_rotate[1]}")
    r_m = initial_transformation.flatten()
    formatted_array = "[" + ", ".join(f"{x:.5f}" for x in r_m) + "]"
    print(f"{name} 最终旋转矩阵:\n", formatted_array)

    initial_transformation = rotate_custom(initial_transformation)
    # 应用初始变换旋转雷达
    pcd.transform(initial_transformation)

    arrows = create_normal_arrows(plane_clouds, normal_vec)
    print("平面的个数为：", len(plane_clouds))

    # 创建坐标轴对象
    axis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=5.0, origin=[0, 0, 0])

    o3d.visualization.draw_geometries([*arrows, axis, pcd])
