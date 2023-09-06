# From: http://www.open3d.org/docs/release/tutorial/pipelines/global_registration.html

import open3d as o3d
import pointcloud_data_loader
import numpy as np


def execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size):
    # We use RANSAC for global registration. In each RANSAC iteration, ransac_n random points are picked from the
    # source point cloud. Their corresponding points in the target point cloud are detected by querying the
    # nearest neighbor in the 33-dimensional FPFH feature space. A pruning step takes fast pruning algorithms to
    # quickly reject false matches early.

    distance_threshold = voxel_size * 1.5
    print(":: RANSAC registration on downsampled point clouds.")
    print("   Since the downsampling voxel size is %.3f," % voxel_size)
    print("   we use a liberal distance threshold %.3f." % distance_threshold)
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
    return result


def preprocess_point_cloud(pcd, voxel_size):
    # We downsample the point cloud, estimate normals, then compute a FPFH feature for each point.
    # The FPFH feature is a 33-dimensional vector that describes the local geometric property of a point.
    # A nearest neighbor query in the 33-dimensinal space can return points with similar local geometric structures.

    print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source, target, voxel_size):
    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def perform_global_registration(source, target):
    if isinstance(source, str):
        source = o3d.io.read_point_cloud(source)
        target = o3d.io.read_point_cloud(target)

    voxel_size = 0.01  # 0.05 means 5cm for this dataset

    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)

    #pointcloud_data_loader.visualize_point_clouds(source_down, target_down)

    result_ransac = execute_global_registration(source_down, target_down, source_fpfh, target_fpfh, voxel_size)
    print(result_ransac)

    return result_ransac.transformation


if __name__ == '__main__':
    import sys

    # Get the function name from command-line argument
    function_name = sys.argv[1]

    if function_name == 'perform_global_registration':
        arg1 = str(sys.argv[2])
        arg2 = str(sys.argv[3])
        result = perform_global_registration(arg1, arg2)
        np.savetxt("tmp/tmp_reg.txt", result)
