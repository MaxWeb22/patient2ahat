# Copyright (c) Chris Choy (chrischoy@ai.stanford.edu) and Wei Dong (weidong@andrew.cmu.edu)
#
# Please cite the following papers if you use any part of the code.
# - Christopher Choy, Wei Dong, Vladlen Koltun, Deep Global Registration, CVPR 2020
# - Christopher Choy, Jaesik Park, Vladlen Koltun, Fully Convolutional Geometric Features, ICCV 2019
# - Christopher Choy, JunYoung Gwak, Silvio Savarese, 4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks, CVPR 2019

import os
from urllib.request import urlretrieve

import open3d as o3d
from DeepGlobalRegistration.core.deep_global_registration import DeepGlobalRegistration
from DeepGlobalRegistration.config import get_config
import numpy as np
import pointcloud_data_loader as pcdl


def perform_dgr_registration(source, target):
  if isinstance(source, str):
    source = o3d.io.read_point_cloud(source)
    target = o3d.io.read_point_cloud(target)

  config = get_config()

  config.weights = "DeepGlobalRegistration/best_val_checkpoint_e300.pth"#best_val_checkpoint_e100.pth"##ResUNetBN2C-feat32-3dmatch-v0.05.pth"##"#

  # preprocessing
  pcd0 = source
  #pcd0.estimate_normals()
  pcd1 = target
  #pcd1.estimate_normals()

  # Sample Test:
  #pcd0 = pcd0.voxel_down_sample(0.05)
  #pcd1 = pcd1.voxel_down_sample(0.05)

  #pcdl.visualize_point_clouds(pcd0, pcd1)

  # registration
  dgr = DeepGlobalRegistration(config)
  T01 = dgr.register(pcd0, pcd1)

  return T01


if __name__ == '__main__':
  import sys

  # Get the function name from command-line argument
  function_name = sys.argv[1]

  if function_name == 'perform_dgr_registration':
    arg1 = str(sys.argv[2])
    arg2 = str(sys.argv[3])
    sys.argv = [sys.argv[0]]

    result = perform_dgr_registration(arg1, arg2)
    np.savetxt("tmp/tmp_reg.txt", result)