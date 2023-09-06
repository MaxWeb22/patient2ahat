import argparse
import copy
import os
import sys
import numpy as np
import torch
import torch.utils.data
import torchvision
import pointcloud_data_loader as pcdl
import p2a_utils

# visualize the point cloud
import open3d as o3d
# open3d>=0.13.0, otherwise, comment line below
# o3d.visualization.webrtc_server.enable_webrtc()

# sys.path.insert(0, '../')
import PointNetLK_Revisited.data_utils as data_utils
import PointNetLK_Revisited.trainer as trainer


def perform_pnlk_registration(target, source):
    if isinstance(source, str):
        source = o3d.io.read_point_cloud(source)
        target = o3d.io.read_point_cloud(target)

    args = argparse.Namespace()

    # dimension for the PointNet embedding
    args.dim_k = 1024

    # device: cuda/cpu
    #args.device = 'cuda:0'
    args.device = 'cpu'

    # maximum iterations for the LK
    args.max_iter = 5

    # embedding function: pointnet
    args.embedding = 'pointnet'

    # output log file name
    args.outfile = 'toyexample_2021_04_17'

    # specify data type: real
    args.data_type = 'real'

    # specify visualize result or not
    args.vis = False

    # specify if example data should be used
    example_data = False

    if example_data:
        # load data
        p0 = np.load('PointNetLK_Revisited/demo/p0.npy')[np.newaxis,...]
        p1 = np.load('PointNetLK_Revisited/demo/p1.npy')[np.newaxis,...]

        # get the source and target from the testset
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(p0[0])
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        source.orient_normals_to_align_with_direction()

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(p1[0])
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.orient_normals_to_align_with_direction()

        #pcdl.visualize_point_clouds(target, source)

        # randomly set the twist parameters for the ground truth pose
        x = np.array([[0.57, -0.29, 0.73, -0.37, 0.48, -0.54]])
    else:
        p0 = np.asarray(source.points)[np.newaxis,...]
        p1 = np.asarray(target.points)[np.newaxis,...]

        # randomly set the twist parameters for the ground truth pose
        x = np.array([[0.057, -0.029, 0.073, -0.037, 0.048, -0.054]])

    # set voxelization parameters
    voxel_ratio = 0.05
    voxel = 2
    max_voxel_points = 1000
    num_voxels = 8

    # construct the testing dataset
    testset = data_utils.ToyExampleData(p0, p1, voxel_ratio, voxel, max_voxel_points, num_voxels, x, args.vis)

    # create model
    dptnetlk = trainer.TrainerAnalyticalPointNetLK(args)
    model = dptnetlk.create_model()

    # specify device
    if not torch.cuda.is_available():
        args.device = 'cpu'
    args.device = torch.device(args.device)
    model.to(args.device)

    # load pre-trained model
    model.load_state_dict(torch.load('PointNetLK_Revisited/logs/model_trained_on_ModelNet40_model_best.pth', map_location='cpu'))

    # testloader
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    # begin testing
    transformation = dptnetlk.test_one_epoch(model, testloader, args.device, 'test', args.data_type, args.vis, toyexample=True)

    if args.vis:
        voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt, _, _ = testset[0]
    else:
        voxels_p0, voxel_coords_p0, voxels_p1, voxel_coords_p1, igt = testset[0]

    if example_data:
        # get the source and target from the testset
        source = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(p0[0])
        source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        source.orient_normals_to_align_with_direction()

        target = o3d.geometry.PointCloud()
        target.points = o3d.utility.Vector3dVector(p1[0])
        target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        target.orient_normals_to_align_with_direction()

        # visualize the source and target
        target.transform(igt)
        #pcdl.visualize_point_clouds(target, source)

        #pcdl.visualize_point_clouds(target, source, transformation)

    return transformation, igt


if __name__ == '__main__':
    import sys

    # Get the function name from command-line argument
    function_name = sys.argv[1]

    if function_name == 'perform_pnlk_registration':
        arg1 = str(sys.argv[2])
        arg2 = str(sys.argv[3])
        result1, result2 = perform_pnlk_registration(arg1, arg2)
        np.savetxt("tmp/tmp_reg.txt", result1)
        np.savetxt("tmp/tmp_init_reg.txt", result2)
