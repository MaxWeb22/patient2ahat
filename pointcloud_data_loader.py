import open3d as o3d
import numpy as np
import copy
import os


# Function to load a point cloud from a file
def load_point_cloud(file_path):
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


# Function to scale a point cloud based on a reference point cloud and a scale factor
def scale_point_cloud(pcd, pcd2, scale_factor):
    normals = np.asarray(pcd.normals)
    pcd.scale(scale_factor, center=pcd2.get_center())
    pcd.estimate_normals()
    pcd.normals = o3d.utility.Vector3dVector(normals)

    return pcd


def scale_point_cloud_wo_center(pcd, scaling_factor):
    pcd_cpy = copy.deepcopy(pcd)
    normals = np.asarray(pcd_cpy.normals)
    scaled_points = np.asarray(pcd_cpy.points) * scaling_factor
    scaled_point_cloud = o3d.geometry.PointCloud()
    scaled_point_cloud.points = o3d.utility.Vector3dVector(scaled_points)
    scaled_point_cloud.estimate_normals()
    pcd_cpy.normals = o3d.utility.Vector3dVector(normals)

    return scaled_point_cloud


# Function to delete color information from a point cloud
def delete_colors(pcd):
    pcd.colors = o3d.utility.Vector3dVector([])
    return pcd


# Function to load an STL file as a point cloud by sampling points using Poisson disk sampling
def load_stl_as_point_cloud(file_path, number_of_points):
    stl_mesh = o3d.io.read_triangle_mesh(file_path)
    pcd = stl_mesh.sample_points_poisson_disk(number_of_points=number_of_points)
    return pcd


# Function to remove planes from a point cloud using RANSAC
def remove_planes(pcd, distance_threshold, ransac_n, num_iterations):
    # Segment planes from the point cloud
    plane_model, inliers = pcd.segment_plane(distance_threshold=distance_threshold,
                                             ransac_n=ransac_n,
                                             num_iterations=num_iterations)
    # Remove the planes from the point cloud
    pcd_without_planes = pcd.select_by_index(inliers, invert=True)

    return pcd_without_planes


# Function to remove single islands from a point cloud based on a minimum number of points per cluster
def remove_single_islands(pcd, island_size):
    # Compute connected components in the point cloud
    labels = np.array(pcd.cluster_dbscan(eps=0.01, min_points=island_size, print_progress=True))

    # Compute the number of points in each connected component
    unique_labels, counts = np.unique(labels, return_counts=True)

    # Create a list of connected components with more than one point
    valid_labels = unique_labels[counts > 1]

    # Create a new point cloud with only the valid connected components
    new_pcd = o3d.geometry.PointCloud()
    for label in valid_labels:
        new_pcd += pcd.select_by_index(np.where(labels == label)[0])

    return new_pcd


# Function to cut a CT scan point cloud based on z and y limits
def cut_ct_scan(pcd):
    # Save the normals
    normals = np.asarray(pcd.normals)

    # Convert the point cloud to a numpy array
    points = np.asarray(pcd.points)

    # Calculate the percentage to cut in z and y directions
    z_cut_percentage = 0.1
    y_cut_percentage = 0.35

    # Calculate the z and y limits to cut
    z_max_limit = np.percentile(points[:, 2], 100 - z_cut_percentage * 100)
    z_min_limit = np.percentile(points[:, 2], z_cut_percentage * 100)
    y_limit = np.percentile(points[:, 1], 100 - y_cut_percentage * 100)

    # Filter the points to remove the first and last 20% in the z direction and last 50% in the y direction
    flt_points = points[(points[:, 2] >= z_min_limit) & (points[:, 2] <= z_max_limit) & (points[:, 1] <= y_limit)]

    # Filter the normals, to keep them after deletion of points
    flt_normals = normals[(points[:, 2] >= z_min_limit) & (points[:, 2] <= z_max_limit) & (points[:, 1] <= y_limit)]

    # Create a new point cloud object with the filtered points
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(flt_points)
    filtered_point_cloud.normals = o3d.utility.Vector3dVector(flt_normals)

    return filtered_point_cloud


# Function to visualize two point clouds with a transformation matrix
def visualize_point_clouds(pcd1, pcd2, transformation=np.identity(4),
                           SaveImage=False, Number=0, Algo=None, Pat=None, Align=None, Experiment=False):
    # Create the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create copies of the point clouds and set their colors to blue and yellow, respectively
    pcd1_cpy = copy.deepcopy(pcd1).paint_uniform_color([1, 0.706, 0])  # Blue color
    pcd2_cpy = copy.deepcopy(pcd2).paint_uniform_color([0, 0.651, 0.929])  # Orange color

    # Add the point clouds to the visualization
    vis.add_geometry(pcd1_cpy.transform(transformation))
    vis.add_geometry(pcd2_cpy)

    # Run the visualization
    if not Experiment:
        vis.run()

    # Render the point clouds to an off-screen buffer
    vis.poll_events()
    vis.update_renderer()

    # Save the image as a file
    if SaveImage:
        # Save Screenshot
        vis.capture_screen_image("screenshots/screen_" + str(Algo) + "_" + str(Pat) + "_" + Align + "_" + str(Number) + ".png")

    vis.destroy_window()


# Function to compare two registrations by visualizing them with their respective point clouds
def compare_registrations(pcd1, pcd2, transformation1, transformation2):
    # Create the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create copies of the point clouds and set their colors to blue and yellow, respectively
    pcd1_cpy = copy.deepcopy(pcd1).paint_uniform_color([1, 0.706, 0])  # Blue color
    pcd2_cpy = copy.deepcopy(pcd1).paint_uniform_color([1.0, 0.0, 0.0])  # Red color
    pcd3_cpy = copy.deepcopy(pcd2).paint_uniform_color([0, 0.651, 0.929])  # Orange color

    # Add the point clouds to the visualization
    vis.add_geometry(pcd1_cpy.transform(transformation1))
    vis.add_geometry(pcd2_cpy.transform(transformation2))
    vis.add_geometry(pcd3_cpy)

    # Run the visualization
    vis.run()
    vis.destroy_window()


# Function to compare three benchmarks by visualizing them with their respective point clouds and transformations
def compare_benchmarks(pcd, transformation1, transformation2, transformation3):
    # Create the visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create copies of the point clouds and set their colors to red, green and blue, respectively
    pcd1_cpy = copy.deepcopy(pcd).paint_uniform_color([0.0, 1.0, 0])  # Green color
    pcd2_cpy = copy.deepcopy(pcd).paint_uniform_color([1.0, 0.0, 0.0])  # Red color
    pcd3_cpy = copy.deepcopy(pcd).paint_uniform_color([0, 0, 1.0])  # Blue color

    # Add the point clouds to the visualization
    vis.add_geometry(pcd1_cpy.transform(transformation1))
    vis.add_geometry(pcd2_cpy.transform(transformation2))
    vis.add_geometry(pcd3_cpy.transform(transformation3))

    # Run the visualization
    vis.run()
    vis.destroy_window()


# Function to load the scene point cloud for a given patient and alignment
def load_scene(pat, align):
    scene_file_path = "data/Pat" + str(pat) + "_" + align + ".ply"
    return load_point_cloud(scene_file_path)

def load():
    scene = "DeepGlobalRegistration/redkitchen_000.ply"
    return load_point_cloud(scene)


# Function to load the patient point cloud for a given patient
def load_patient(pat):
    stl_file_path = "data/Pat" + str(pat) + ".stl"
    return load_stl_as_point_cloud(stl_file_path, number_of_points=10000)


# Function to load and preprocess the necessary files for registration
def load_files(pat=6, align="front"):
    # Load the first point cloud
    scene_original = load_scene(pat, align)

    #kitchen = load()

    # Load the STL file and convert it to a point cloud
    ct_scan_original = load_patient(pat)

    # Scale the Patient data coordinates from meters to millimeters
    ct_scan = scale_point_cloud(ct_scan_original, scene_original, 0.001)

    # Delete the color from the point clouds
    scene = delete_colors(scene_original)
    ct_scan = delete_colors(ct_scan)

    # Cut the back and top and bottom of the Patient data
    ct_scan = cut_ct_scan(ct_scan)

    # Save the cut patient point cloud
    o3d.io.write_point_cloud("cut_" + str(pat) + ".ply", ct_scan)

    # Delete planes from the point clouds
    scene = remove_planes(scene, distance_threshold=0.016, ransac_n=3, num_iterations=1000)

    # Removes single islands from scene
    scene = remove_single_islands(scene, 1)

    # Save the scene
    o3d.io.write_point_cloud("scene_" + str(pat) + "_" + align + ".ply", scene)

    # Visualize the two point clouds in the same scene
    #visualize_point_clouds(scene, ct_scan)

    return ct_scan, scene, ct_scan_original, scene_original
