import subprocess
import open3d as o3d
import numpy as np
import os
import pointcloud_data_loader as pcdl
import copy


# Execute a function with a certain interpreter and arguments in a subprocess
def execute_function(file_path, interpreter_path, function_name, *args):
    # Save point cloud arguments to temporary files
    temp_files = []
    args_tmp = []
    for i, arg in enumerate(args[0]):
        if isinstance(arg, o3d.geometry.PointCloud):
            o3d.io.write_point_cloud("tmp/" + str(i) + ".ply", arg)
            args_tmp.append("tmp/" + str(i) + ".ply")

    # Construct the command to execute the function
    command = f'{interpreter_path} {file_path} "{function_name}"'
    for arg in args_tmp:
        command += f' "{arg}"'

    # Execute the command
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Check for any errors
    if result.returncode != 0:
        print(f'Error executing function: {result.stderr}')
        return None

    # Return the output
    return result.stdout.strip()


# Function to save a registration to a file
def save_registration(algorithm, patient, alignment, registration):
    np.savetxt("registrations/" + str(algorithm) + "_" + str(patient) + "_" + str(alignment) + ".txt", registration)


# Function to save the current benchmark registration for experiments
def save_current_bm_registration(bm_result, patient, alignment):
    np.savetxt("registrations/exp_bm/exp_" + str(patient) + "_" + str(alignment) + ".txt", bm_result)



# Function to load a registration from a file
def load_registration(algorithm, patient, alignment):
    filepath = "registrations/bm_" + str(algorithm) + "_" + str(patient) + "_" + str(alignment) + ".txt"
    if os.path.exists(filepath):
        return np.loadtxt(filepath)
    else:
        return None


# Function to load the temporary created registration file
def load_temp_registration():
    filepath = "tmp/tmp_reg.txt"
    return np.loadtxt(filepath)


# Function to load the temporary created registration file
def load_temp_init_registration():
    filepath = "tmp/tmp_init_reg.txt"
    return np.loadtxt(filepath)


# Function to load a benchmark registration for a given patient and alignment
def load_benchmark(pat, align):
    # Folder, where the current benchmarks are saved
    folder_path = "registrations/current_bm"

    # Part of the filename which we want to search for
    part_in_filename = "_" + str(pat) + "_" + str(align)

    # Search the defined folder for a .txt file that contains the name
    for file_name in os.listdir(folder_path):
        if part_in_filename in file_name and file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)

            # Check, if the file exists, else returns None
            if os.path.exists(file_path):
                return np.loadtxt(file_path)
            else:
                return None
    return None


# Function to visualize a benchmark registration with the source and target point clouds
def visualize_bm_registration(pat, align, src, trgt):
    # Load the current benchmark registration for the patient and alignment
    benchmark = load_benchmark(pat, align)

    # If a benchmark exists, visualize it with the source and target point cloud
    if benchmark is not None:
        pcdl.visualize_point_clouds(src, trgt, benchmark)
        return benchmark
    else:
        print("Benchmark not found!")
        return None


# Function, that calculates the translation and rotation error between the benchmark registration and a new one
def calculate_registration_errors(transformation1, transformation2):
    # Extract translation vectors
    translation1 = transformation1[:3, 3]
    translation2 = transformation2[:3, 3]

    # The translation error is calculated as the Euclidean distance between the two translation vectors.
    translation_error = np.linalg.norm(translation1 - translation2, ord=2)

    # Extract rotation matrices
    rotation1 = transformation1[:3, :3]
    rotation2 = transformation2[:3, :3]

    # The rotation error is computed using the formula for the angle between two rotation matrices.
    rotation_error = np.arccos((np.trace(np.dot(rotation1.T, rotation2)) - 1.0) / 2.0)

    rotation_error = rotation_error * 180 / np.pi

    print("Translation Error (Meter): " + str(translation_error))
    print("Rotation Error (Degree): " + str(rotation_error))

    return translation_error, rotation_error


# Function, that overlays two pont clouds based on manually found translation
def translate_to_overlay(patient, alignment, source, src_original):
    translations = {
        (6, "front"): np.array([0.09, 0.14, 0.9]),
        (6, "left"): np.array([0.09, 0.14, 1.]),
        (6, "right"): np.array([-0.13, 0.14, 1.]),
        (10, "front"): np.array([0., 0.14, 0.56]),
        (10, "left"): np.array([0.17, 0.14, 0.65]),
        (10, "right"): np.array([-0.15, 0.14, 0.8])
    }

    key = (patient, alignment)
    if key not in translations:
        raise ValueError("Invalid patient or alignment")

    src_pnts = o3d.utility.Vector3dVector(np.asarray(source) + translations[key])
    src_original_pnts = o3d.utility.Vector3dVector(np.asarray(src_original) + translations[key])

    return src_pnts, src_original_pnts, translations[key]


# Function, that groups points into voxels and keeps only one point per voxel, rearranging points within the same area
def rearrange_points(point_cloud, voxel_size):
    # Convert point cloud to open3d.geometry.PointCloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # Apply voxel downsampling to rearrange points within the same area
    downsampled_pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    # Retrieve rearranged points as numpy array
    rearranged_points = np.asarray(downsampled_pcd.points)

    return rearranged_points


def process_point_cloud(point_cloud, num_points=None, filter_radius=None):
    # Convert point cloud to numpy array
    points = np.asarray(point_cloud.points)

    normals = np.asarray(point_cloud.normals)

    # Change number of points
    if num_points is not None:
        if num_points > len(points):
            # Upsample the point cloud using linear interpolation
            indices = np.linspace(0, len(points) - 1, num_points).astype(int)
            points = points[indices]
            normals = normals[indices]
        elif num_points < len(points):
            # Downsample the point cloud randomly
            indices = np.random.choice(len(points), num_points, replace=False)
            points = points[indices]
            normals = normals[indices]

    # Apply filter
    if filter_radius is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.normals = o3d.utility.Vector3dVector(normals)
        pcd = pcd.voxel_down_sample(voxel_size=filter_radius)
        points = np.asarray(pcd.points)
        normals = np.asarray(pcd.normals)

    # Convert numpy array back to point cloud
    processed_point_cloud = o3d.geometry.PointCloud()
    processed_point_cloud.points = o3d.utility.Vector3dVector(points)
    processed_point_cloud.normals = o3d.utility.Vector3dVector(normals)

    return processed_point_cloud


# Function, that creates a transformation matrix with
# rotations randomly drawn from [0◦, 45◦] and translations randomly sampled from [0, 0.8]
def random_transformation_matrix():
    # Generate random rotation angles
    rot_angle_x = np.random.uniform(0, 45)  # Rotation angle around X-axis
    rot_angle_y = np.random.uniform(0, 45)  # Rotation angle around Y-axis
    rot_angle_z = np.random.uniform(0, 45)  # Rotation angle around Z-axis

    # Generate random translation values
    translation = np.random.uniform(0, 0.8, size=3)

    # Create rotation matrix for each axis
    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(np.radians(rot_angle_x)), -np.sin(np.radians(rot_angle_x))],
                           [0, np.sin(np.radians(rot_angle_x)), np.cos(np.radians(rot_angle_x))]])

    rotation_y = np.array([[np.cos(np.radians(rot_angle_y)), 0, np.sin(np.radians(rot_angle_y))],
                           [0, 1, 0],
                           [-np.sin(np.radians(rot_angle_y)), 0, np.cos(np.radians(rot_angle_y))]])

    rotation_z = np.array([[np.cos(np.radians(rot_angle_z)), -np.sin(np.radians(rot_angle_z)), 0],
                           [np.sin(np.radians(rot_angle_z)), np.cos(np.radians(rot_angle_z)), 0],
                           [0, 0, 1]])

    # Create transformation matrix
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = rotation_x @ rotation_y @ rotation_z
    transform_matrix[:3, 3] = translation

    return transform_matrix
