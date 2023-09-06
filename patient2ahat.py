import pointcloud_data_loader as pcdl
import icp_registration
import p2a_utils
import numpy as np
import copy
import random
import open3d as o3d

# Select if the algorithm should be called directly, or with subprocess
call_direct = False

interpreters = {
    'Anac_Python_3_7': '/home/weber/anaconda3/envs/Anac_Python_3_7/bin/python',
    'Anac_Python_fmr': '/home/weber/anaconda3/envs/Anac_Python_fmr/bin/python',
    'Anac_Python_pointnet_lk': '/home/weber/anaconda3/envs/Anac_Python_pointnet_lk/bin/python',
    'Anac_Python_3_8_dgr': '/home/weber/anaconda3/envs/Anac_Python_3_8_dgr/bin/python'
}

files = {
    'icp': 'icp_registration.py',
    'global': 'global_registration.py',
    'fmr': 'fmr_registration.py',
    'pointnetlk': 'pnlk_registration.py',
    'dgr': 'dgr_registration.py'
}

# Dictionary of algorithms
algo_list = {
    0: "bm_vis",       # Visualization of the current benchmark transformation for the patient and alignment
    1: "bm_compare",   # Compare all the benchmarks for each patient
    2: "global",       # Execute global Registration
    3: "icp",          # Execute ICP Registration
    4: "fmr",          # Execute FeatureMetricRegistration
    5: "pointnetlk",   # Execute PointnetLK Registration
    6: "dgr",          # Execute DeepGlobalRegistration
    7: "global_icp"    # Execute Global+ICP Registration
}

# List of patients
patient_list = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]

# List of alignments
alignment_list = ["front", "left", "right"]

# Select algorithm, patient, and alignment
algorithm = algo_list[6]
patient = patient_list[3]
alignment = alignment_list[0]

# Create empty registration variable
reg_result = np.eye(4)

# Load files
source, target, src_original, trgt_original = pcdl.load_files(patient, alignment)

# Use target/source randomization
src_rnd = False
trg_rnd = False

use_rnd_trans = False

# Random Transform: rotations randomly drawn from [0◦, 45◦] and translations randomly sampled from [0, 0.8]
if use_rnd_trans:
    rnd_trans_matrix = p2a_utils.random_transformation_matrix()
else:
    rnd_trans_matrix = np.identity(4)

# If randomization is True, one of the point clouds is changed to a randomly transformed point cloud
if src_rnd:
    source = copy.deepcopy(target)
    src_original = copy.deepcopy(trgt_original)
    source.transform(rnd_trans_matrix)
    src_original.transform(rnd_trans_matrix)
    source = p2a_utils.process_point_cloud(source, num_points=20000, filter_radius=0.009)
    pcdl.visualize_point_clouds(source, target)
elif trg_rnd:
    target = copy.deepcopy(source)
    source.transform(rnd_trans_matrix)
    trgt_original = copy.deepcopy(src_original)
    src_original.transform(rnd_trans_matrix)
    target = p2a_utils.process_point_cloud(target, num_points=20000, filter_radius=0.009)
    pcdl.visualize_point_clouds(source, target)
else:
    source.transform(rnd_trans_matrix)
    src_original.transform(rnd_trans_matrix)
    pcdl.visualize_point_clouds(source, target)


# Load the current benchmark for the selected patient and alignment
if algorithm == "bm_vis":
    benchmark = p2a_utils.visualize_bm_registration(patient, alignment, src_original, trgt_original)
    exit()

# Load and compare all benchmarks for all patients
elif algorithm == "bm_compare":
    for pat in patient_list:
        current_src = pcdl.load_patient(pat)
        bm_list = list()
        for align in alignment_list:
            bm_list.append(p2a_utils.load_benchmark(pat, align))
        pcdl.compare_benchmarks(current_src, bm_list[0], bm_list[1], bm_list[2])
    exit()

# Check the selected algorithm and perform the corresponding actions
if algorithm == "icp":
    # Load the best global registration for the patient and alignment till now
    reg_glob = p2a_utils.load_registration("global", patient, alignment)

    # Check, if there is already a "best" global registration
    if reg_glob is not None:
        pcdl.visualize_point_clouds(src_original, trgt_original, reg_glob)
        reg_result = icp_registration.perform_icp_p2point_registration(source, target, reg_glob)

        # Compare the icp registration to the best global registration to check, if the result is better
        pcdl.compare_registrations(src_original, trgt_original, reg_glob, reg_result)
    else:
        print("No Global Registration Benchmark existing for this Configuration")

elif algorithm == "global":
    if call_direct:
        import global_registration

        reg_result = global_registration.perform_global_registration(source, target)
    else:
        return_val = p2a_utils.execute_function(files[algorithm], interpreters["Anac_Python_3_7"],
                                                "perform_global_registration", (source, target))
        print(return_val)
        reg_result = p2a_utils.load_temp_registration()

elif algorithm == "fmr":
    if call_direct:
        import fmr_registration

        reg_result = fmr_registration.perform_fmr_registration(source, target)
    else:
        return_val = p2a_utils.execute_function(files[algorithm], interpreters["Anac_Python_fmr"],
                                                "perform_fmr_registration", (source, target))
        print(return_val)
        reg_result = p2a_utils.load_temp_registration()

elif algorithm == "pointnetlk":
    # For pointnetlk_revisited the two pointclouds need to have some degree of overlay
    # This funktion overlays the point clouds, so that the algorithm works
    source_cpy = copy.deepcopy(source)
    source_original_cpy = copy.deepcopy(src_original)
    if (src_rnd==False and trg_rnd==False) is True:
        source_cpy.transform(np.linalg.inv(rnd_trans_matrix))
        source_original_cpy.transform(np.linalg.inv(rnd_trans_matrix))
        pcdl.visualize_point_clouds(source_cpy, target)
        source_cpy.points, source_original_cpy.points, translation_vector = p2a_utils.translate_to_overlay(patient, alignment,
                                                                        source_cpy.points, source_original_cpy.points)
    else:
        source_cpy.transform(np.linalg.inv(rnd_trans_matrix))
        source_original_cpy.transform(np.linalg.inv(rnd_trans_matrix))

    pcdl.visualize_point_clouds(source_cpy, target)
    pcdl.visualize_point_clouds(source_original_cpy, target)

    if call_direct:
        import pnlk_registration

        reg_result, init_reg = pnlk_registration.perform_pnlk_registration(source_original_cpy, trgt_original)
    else:
        return_val = p2a_utils.execute_function(files[algorithm], interpreters["Anac_Python_pointnet_lk"],
                                                "perform_pnlk_registration", (source_original_cpy, trgt_original))
        print(return_val)

        reg_result = p2a_utils.load_temp_registration()
        init_reg = p2a_utils.load_temp_init_registration()

    # Create the transformation matrix (A to B)
    a_to_b = np.eye(4)

    # Calculate the transformation matrix from source origin to registration result
    if (src_rnd==False and trg_rnd==False) is True:
        a_to_b[:3, 3] = translation_vector
        a_to_d = reg_result @ np.array(init_reg) @ a_to_b @ np.linalg.inv(rnd_trans_matrix)
    else:
        a_to_d = reg_result @ np.array(init_reg) @ np.linalg.inv(rnd_trans_matrix)

    reg_result = a_to_d


elif algorithm == "dgr":
    if call_direct:
        import dgr_registration

        reg_result = dgr_registration.perform_dgr_registration(source, target)
    else:
        return_val = p2a_utils.execute_function(files[algorithm], interpreters["Anac_Python_3_8_dgr"],
                                                "perform_dgr_registration", (source, target))
        print(return_val)
        reg_result = p2a_utils.load_temp_registration()


elif algorithm == "global_icp":
    if call_direct:
        import global_registration

        reg_glob = global_registration.perform_global_registration(source, target)

        pcdl.visualize_point_clouds(src_original, trgt_original, reg_glob)
        reg_result = icp_registration.perform_icp_p2point_registration(source, target, reg_glob)

        # Compare the icp registration to the best global registration to check, if the result is better
        #pcdl.compare_registrations(src_original, trgt_original, reg_glob, reg_result)

    else:
        return_val = p2a_utils.execute_function(files["global"], interpreters["Anac_Python_3_7"],
                                                "perform_global_registration", (source, target))
        print(return_val)
        reg_glob = p2a_utils.load_temp_registration()

        pcdl.visualize_point_clouds(src_original, trgt_original, reg_glob)
        reg_result = icp_registration.perform_icp_p2point_registration(source, target, reg_glob)

        # Compare the icp registration to the best global registration to check, if the result is better
        #pcdl.compare_registrations(src_original, trgt_original, reg_glob, reg_result)


# Visualize the down sampled and preprocessed patient and scene data with the registration result
pcdl.visualize_point_clouds(source, target, reg_result)
# Visualize the original patient and scene data with the registration result
pcdl.visualize_point_clouds(src_original, trgt_original, reg_result, True, Number=0, Algo=algorithm, Pat=patient, Align=alignment)
# Save the registration result with the name, alignment and algorithm
p2a_utils.save_registration(algorithm, patient, alignment, reg_result)
# Calculate the translation and registration error with the current registration result compared to the benchmark
err_t, err_r = p2a_utils.calculate_registration_errors(p2a_utils.load_benchmark(patient, alignment), reg_result)
