import time
import syglass as sy
import trimesh
import numpy as np
import pyvista as pv
import os
import shutil
import pathlib

#              ________________________________________________              #
#/=============|  Surface-to-Surface Colocalization Example   |=============\#
#|             ‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾‾             |#
#|  Compares the distance between meshes and pairs them based on a          |#
#|  distance threshold.                                                     |#
#|                                                                          |#
#|  Note that for now the project must NOT be opened in syGlass when this   |#
#|  script runs. This can later be built into a syGlass plugin, which will  |#
#|  not suffer from this issue.                                             |#
#|                                                                          |#
#\==========================================================================/#

# constants: modify as needed before running the script
EXPERIMENT_NAME = 'default'
PROJECT_PATH = 'C:/syGlassProjects/VCNMended/VCNMended.syg'
# in project units
DISTANCE_THRESHOLD = 200

if __name__ == '__main__':
    # get the project object, meshes
    project = sy.get_project(PROJECT_PATH)
    MESH_PATH = pathlib.Path(PROJECT_PATH).parent.resolve()
    if not os.path.exists(str(MESH_PATH) + '/project_meshes'):
        os.mkdir(str(MESH_PATH) + '/project_meshes')
    mesh_names = project.impl.GetMeshNamesAndSizes(EXPERIMENT_NAME)
    voxel_dimensions = project.get_voxel_dimensions()
    pv_read_meshes = []
    mean_distances_unsorted = []
    blacklist_meshes = []
    pairs = []
    orphaned = []
    sorted_dists = []
    centers_list = []
    new_surfaces_paths = []

    # iterate through list of meshes
    for mesh_name in mesh_names:
        print('\nProcessing mesh: ' + mesh_name)

        mesh_path = str(MESH_PATH) + '/project_meshes/' + mesh_name
        project.impl.ExportMeshOBJs(EXPERIMENT_NAME, mesh_name, mesh_path)
       
        # meshes take a second to export—here we wait for them
        while project.impl.GetMeshIOPercentage() != 100.0:
            time.sleep(0.1)

        trimesh_object = trimesh.load_mesh(mesh_path)
        trimesh_splits = trimesh.graph.split(trimesh_object)

        # pyvista reads each mesh
        pv_read_mesh_group = []
        component_id = 1
        for connected_component in trimesh_splits:
            name = mesh_name[:-4] + "_" + str(component_id) + ".obj"
            path = str(MESH_PATH) + '/project_meshes/' + name
            new_surfaces_paths.append(path)
            connected_component.export(path)
            pv_read_mesh_group.append((name, pv.read(path)))
            component_id = component_id + 1

        pv_read_meshes.append(pv_read_mesh_group)

    project.delete_all_surfaces()
    project.import_meshes(new_surfaces_paths)

    # compare all meshes
    print('Making comparisons...')
    for pv_mesh_group_a in pv_read_meshes:
        for pv_mesh_a in pv_mesh_group_a:
            for pv_mesh_group_b in pv_read_meshes:
                for pv_mesh_b in pv_mesh_group_b:
                    if pv_mesh_group_a == pv_mesh_group_b:
                        continue
                    closest_cells, closest_points = pv_mesh_b[1].find_closest_cell(pv_mesh_a[1].points, return_closest_point = True)
                    d_exact = np.linalg.norm(pv_mesh_a[1].points - closest_points, axis = 1)
                    mean_dist = np.mean(d_exact)
                    mean_distances_unsorted.append((pv_mesh_a[0], pv_mesh_b[0], mean_dist))
                    print('Comparison between ' + pv_mesh_a[0] + ' and ' + pv_mesh_b[0] + ': Complete')
    
    # sort and then pair meshes
    print('Sorting...')
    sorted_dists = sorted(mean_distances_unsorted, key=lambda i: i[-1])
    for sorted_dist in sorted_dists:
        if (blacklist_meshes.count(sorted_dist[0]) == 0 and blacklist_meshes.count(sorted_dist[1]) == 0):
            if sorted_dist[2] <= DISTANCE_THRESHOLD:
                pairs.append(sorted_dist)
                
                color_a = (0, 0, 0, 0)
                color_b = (0, 0, 0, 0)

                group_counter = 1
                for pv_mesh_group in pv_read_meshes:
                    for pv_mesh in pv_mesh_group:
                        if pv_mesh[0] == sorted_dist[0]:
                            color_a = (128 + (group_counter % 2) * 128, 128 + (group_counter % 4) * 128, 128 + (group_counter % 8) * 128)
                        if pv_mesh[1] == sorted_dist[1]:
                            color_b = (128 + (group_counter % 2) * 128, 128 + (group_counter % 4) * 128, 128 + (group_counter % 8) * 128)
                    group_counter = group_counter + 1

                project.set_surface_color(sorted_dist[0], (255, 0, 0, 1) , EXPERIMENT_NAME)
                project.set_surface_color(sorted_dist[1], (0, 255, 0, 1) , EXPERIMENT_NAME)
                blacklist_meshes.append(sorted_dist[0])
                blacklist_meshes.append(sorted_dist[1])
                mesh1 = trimesh.load_mesh(str(MESH_PATH) + '/project_meshes/' + sorted_dist[0])
                mesh2 = trimesh.load_mesh(str(MESH_PATH) + '/project_meshes/' + sorted_dist[1])
                xyz_center_points1 = mesh1.center_mass
                zyx_center_points1 = [xyz_center_points1[2] / voxel_dimensions[0], xyz_center_points1[1] / voxel_dimensions[1], xyz_center_points1[0] / voxel_dimensions[2]]
                xyz_center_points2 = mesh2.center_mass
                zyx_center_points2 = [xyz_center_points2[2] / voxel_dimensions[0], xyz_center_points2[1] / voxel_dimensions[1], xyz_center_points2[0] / voxel_dimensions[2]]
                center_mass_points = [zyx_center_points1, zyx_center_points2]
                centers_list.append(center_mass_points)
    
    centers_array = np.array(centers_list)
    project.set_distance_measurements(centers_array, EXPERIMENT_NAME)

    # check for orphaned meshes
    for pv_mesh_group in pv_read_meshes:
        for pv_mesh in pv_mesh_group:
            if blacklist_meshes.count(pv_mesh[0]) == 0:
                orphaned.append(pv_mesh[0])
                project.set_surface_color(pv_mesh[0], (70, 70, 70, 1) , EXPERIMENT_NAME)
    print("Pairs: " + str(pairs))
    print("Orphaned: " + str(orphaned))

    shutil.rmtree(str(MESH_PATH) + '/project_meshes')