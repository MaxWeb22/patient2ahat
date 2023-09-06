import pointcloud_data_loader as pcdl
import icp_registration
import p2a_utils
import numpy as np
import copy
import pandas as pd
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
    0: "global",       # Execute global Registration
    1: "fmr",          # Execute FeatureMetricRegistration
    2: "dgr",          # Execute DeepGlobalRegistration
    3: "pointnetlk"    # Execute PointnetLK Registration
}

# List of patients
patient_list = [1, 2, 4, 5, 6, 7, 8, 9, 10, 11]

# List of alignments
alignment_list = ["front", "left", "right"]

runs = 1

results = []

random_rotation = False


for patient in patient_list:
    for alignment in alignment_list:
        # Load files
        source, target, src_original, trgt_original = pcdl.load_files(patient, alignment)

        if random_rotation:
            # Random Transform: rotations randomly drawn from [0◦, 45◦] and translations randomly sampled from [0, 0.8]
            rnd_trans_matrix = p2a_utils.random_transformation_matrix()

            source.transform(rnd_trans_matrix)
            src_original.transform(rnd_trans_matrix)
            # pcdl.visualize_point_clouds(source, target)

            # perform benchmark registration
            return_val = p2a_utils.execute_function(files["global"], interpreters["Anac_Python_3_7"],
                                                    "perform_global_registration", (source, target))
            print(return_val)
            reg_glob = p2a_utils.load_temp_registration()
            bm_result = icp_registration.perform_icp_p2point_registration(source, target, reg_glob)
            p2a_utils.save_current_bm_registration(bm_result, patient, alignment)
            pcdl.visualize_point_clouds(src_original, trgt_original, bm_result, True, Algo="current_bm", Pat=patient, Align=alignment, Experiment=True)

        for algorithm in algo_list:
            for run in range(runs):
                #pcdl.visualize_point_clouds(source, target)
                #pcdl.visualize_point_clouds(src_original, trgt_original)


                # Create empty registration variable
                reg_result = np.eye(4)

                # Check the selected algorithm and perform the corresponding actions
                if algo_list[algorithm] == "global":
                    if call_direct:
                        import global_registration

                        reg_result = global_registration.perform_global_registration(source, target)
                    else:
                        return_val = p2a_utils.execute_function(files[algo_list[algorithm]], interpreters["Anac_Python_3_7"],
                                                                "perform_global_registration", (source, target))
                        print(return_val)
                        reg_result = p2a_utils.load_temp_registration()

                elif algo_list[algorithm] == "fmr":
                    if call_direct:
                        import fmr_registration

                        reg_result = fmr_registration.perform_fmr_registration(source, target)
                    else:
                        return_val = p2a_utils.execute_function(files[algo_list[algorithm]], interpreters["Anac_Python_fmr"],
                                                                "perform_fmr_registration", (source, target))
                        print(return_val)
                        reg_result = p2a_utils.load_temp_registration()

                elif algo_list[algorithm] == "pointnetlk":
                    # For pointnetlk_revisited the two pointclouds need to have some degree of overlay
                    # This funktion overlays the point clouds, so that the algorithm works
                    source_cpy = copy.deepcopy(source)
                    source_original_cpy = copy.deepcopy(src_original)
                    if random_rotation:
                        source_cpy.transform(np.linalg.inv(rnd_trans_matrix))
                        source_original_cpy.transform(np.linalg.inv(rnd_trans_matrix))
                    else:
                        return_val = p2a_utils.execute_function('global_registration.py', interpreters["Anac_Python_3_7"],
                                                                "perform_global_registration", (source, target))
                        print(return_val)
                        transf = p2a_utils.load_temp_registration()
                        source_cpy.transform(transf)
                        source_original_cpy.transform(transf)

                    #pcdl.visualize_point_clouds(source_cpy, target)
                    #source_cpy.points, source_original_cpy.points, translation_vector = p2a_utils.translate_to_overlay(patient, alignment,
                    #                                                                source_cpy.points, source_original_cpy.points)

                    #pcdl.visualize_point_clouds(source_cpy, target)
                    #pcdl.visualize_point_clouds(source_original_cpy, target)

                    if call_direct:
                        import pnlk_registration

                        reg_result, init_reg = pnlk_registration.perform_pnlk_registration(source_original_cpy, trgt_original)
                    else:
                        return_val = p2a_utils.execute_function(files[algo_list[algorithm]], interpreters["Anac_Python_pointnet_lk"],
                                                                "perform_pnlk_registration", (source_original_cpy, trgt_original))
                        print(return_val)

                        reg_result = p2a_utils.load_temp_registration()
                        init_reg = p2a_utils.load_temp_init_registration()

                    # Create the transformation matrix (A to B)
                    #a_to_b = np.eye(4)
                    #a_to_b[:3, 3] = translation_vector

                    # Calculate the transformation matrix from source origin to registration result
                    #if random_rotation:
                    #    a_to_d = reg_result @ np.array(init_reg) @ a_to_b @ np.linalg.inv(rnd_trans_matrix)
                    #else:
                    a_to_d = reg_result @ np.array(init_reg) @ np.linalg.inv(transf)

                    reg_result = a_to_d

                elif algo_list[algorithm] == "dgr":
                    if call_direct:
                        import dgr_registration

                        reg_result = dgr_registration.perform_dgr_registration(source, target)
                    else:
                        return_val = p2a_utils.execute_function(files[algo_list[algorithm]], interpreters["Anac_Python_3_8_dgr"],
                                                                "perform_dgr_registration", (source, target))
                        print(return_val)
                        reg_result = p2a_utils.load_temp_registration()

                # Visualize the down sampled and preprocessed patient and scene data with the registration result
                #pcdl.visualize_point_clouds(source, target, reg_result)
                # Visualize the original patient and scene data with the registration result
                pcdl.visualize_point_clouds(src_original, trgt_original, reg_result, True, Number=run, Algo=algo_list[algorithm], Pat=patient, Align=alignment, Experiment=True)
                # Save the registration result with the name, alignment and algorithm
                # p2a_utils.save_registration(algorithm, patient, alignment, reg_result)
                # Calculate the translation and registration error with the current registration result compared to the benchmark

                if random_rotation:
                    err_t, err_r = p2a_utils.calculate_registration_errors(bm_result, reg_result)
                else:
                    err_t, err_r = p2a_utils.calculate_registration_errors(p2a_utils.load_benchmark(patient, alignment), reg_result)

                if err_t < 0.4 and err_r < 15:
                    success = 1
                else:
                    success = 0

                result = {
                    'Patient': patient,
                    'Alignment': alignment,
                    'Algorithm': algo_list[algorithm],
                    'Run': run + 1,
                    'Translation Error': err_t,
                    'Rotation Error': err_r,
                    'Success': success
                }

                results.append(result)

df = pd.DataFrame(results)
df.set_index(['Patient', 'Alignment', 'Algorithm', 'Run'], inplace=True)
df.to_csv('registration_results.csv')
