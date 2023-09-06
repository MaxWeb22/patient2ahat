import copy
import open3d as o3d
import numpy as np
import pointcloud_data_loader


#def draw_registration_result(source, target, transformation):
#    source_temp = copy.deepcopy(source)
#    target_temp = copy.deepcopy(target)
#    source_temp.paint_uniform_color([1, 0.706, 0])
#    target_temp.paint_uniform_color([0, 0.651, 0.929])

#    source_temp.transform(transformation)
#    o3d.visualization.draw_geometries([source_temp, target_temp])


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
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
    print(":: Downsample with a voxel size %.3f." % voxel_size)
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


def prepare_dataset(source, target, voxel_size):
    print(":: Load two point clouds.")

    evaluation = o3d.pipelines.registration.evaluate_registration(source, target, 0.02, np.identity(4))
    print(evaluation)

    #pointcloud_data_loader.visualize_point_clouds(source, target)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_icp(source, target):
    voxel_size = 0.01  # 0.05 means 5cm for this dataset
    source, target, source_down, target_down, source_fpfh, target_fpfh = prepare_dataset(source, target, voxel_size)

    print("Apply Global Registration")
    result_ransac = execute_global_registration(source_down, target_down,
                                                source_fpfh, target_fpfh,
                                                voxel_size)
    print(result_ransac)
    pointcloud_data_loader.visualize_point_clouds(source_down, target_down, result_ransac.transformation)

    ############################# ICP #####################################

    threshold = 0.02  #0.02 [max_correspondence_distance]
    trans_init = result_ransac.transformation

    #draw_registration_result(source_down, target_down, trans_init)

    #print("Initial alignment")
    #evaluation = o3d.pipelines.registration.evaluate_registration(source_down, target_down, threshold, trans_init)
    #print(evaluation)

    print("Apply point-to-point ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source_down, target_down, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    print(reg_p2p)

    #pointcloud_data_loader.visualize_point_clouds(source, target, reg_p2p.transformation)
    #pointcloud_data_loader.visualize_point_clouds(source, target)

    #draw_registration_result(source, target, reg_p2p.transformation)

    #print("Apply point-to-plane ICP")
    #reg_p2l = o3d.pipelines.registration.registration_icp(
    #    source_down, target_down, threshold, trans_init,
    #    o3d.pipelines.registration.TransformationEstimationPointToPlane())
    #print(reg_p2l)
    #draw_registration_result(source_down, target_down, reg_p2l.transformation)

    #draw_registration_result(source, target, reg_p2l.transformation)

    return reg_p2p.transformation
