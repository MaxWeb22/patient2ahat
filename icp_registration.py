# From: http://www.open3d.org/docs/release/tutorial/pipelines/icp_registration.html
# and   http://www.open3d.org/docs/0.14.1/tutorial/t_pipelines/t_icp_registration.html

import copy
import time
import open3d as o3d
import numpy as np
import pointcloud_data_loader


def perform_icp_p2point_registration(source, target, trans_init=np.identity(4)):
    treg = o3d.pipelines.registration

    # Search distance for Nearest Neighbour Search [Hybrid-Search is used].
    max_correspondence_distance = 0.01

    # Select the `Estimation Method`, and `Robust Kernel` (for outlier-rejection).
    estimation = treg.TransformationEstimationPointToPoint()

    # Convergence-Criteria
    criteria = treg.ICPConvergenceCriteria(relative_fitness=0.000001, relative_rmse=0.000001, max_iteration=10000)

    print("Initial alignment")
    evaluation = treg.evaluate_registration(source, target, max_correspondence_distance, trans_init)
    print(evaluation)

    print("Apply ICP")
    s = time.time()
    reg_p2p = treg.registration_icp(source, target, max_correspondence_distance, trans_init, estimation, criteria)
    icp_time = time.time() - s

    print("Time taken by ICP: ", icp_time)
    print("Inlier Fitness: ", reg_p2p.fitness)
    print("Inlier RMSE: ", reg_p2p.inlier_rmse)

    return reg_p2p.transformation
