import pymesh
import argparse
import json
from os import listdir, mkdir
from os.path import isfile, join, exists
import numpy as np
from numpy.linalg import norm
import multiprocessing
from functools import partial
import subprocess


def fix_mesh(mesh, detail="normal"):
    bbox_min, bbox_max = mesh.bbox;
    diag_len = norm(bbox_max - bbox_min);
    if detail == "normal":
        target_len = diag_len * 5e-3;
    elif detail == "high":
        target_len = diag_len * 2.5e-3;
    elif detail == "low":
        target_len = diag_len * 1e-2;
    elif detail == "extra_low":
        target_len = diag_len * 5e-2;
    print("Target resolution: {} mm".format(target_len));

    count = 0;
    mesh, __ = pymesh.remove_degenerated_triangles(mesh, 100);
    mesh, __ = pymesh.split_long_edges(mesh, target_len);
    num_vertices = mesh.num_vertices;
    while True:
        mesh, __ = pymesh.collapse_short_edges(mesh, 1e-6);
        mesh, __ = pymesh.collapse_short_edges(mesh, target_len,
                preserve_feature=True);
        mesh, __ = pymesh.remove_obtuse_triangles(mesh, 150.0, 100);
        if mesh.num_vertices == num_vertices:
            break;

        num_vertices = mesh.num_vertices;
        print("#v: {}".format(num_vertices));
        count += 1;
        if count > 10: break;

    mesh = pymesh.resolve_self_intersection(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh = pymesh.compute_outer_hull(mesh);
    mesh, __ = pymesh.remove_duplicated_faces(mesh);
    mesh, __ = pymesh.remove_obtuse_triangles(mesh, 179.0, 5);
    mesh, __ = pymesh.remove_isolated_vertices(mesh);

    return mesh;


def process_obj(obj_file, dir_in, dir_out, grid_size, max_faces, make_convex=False, fit_cylinder=False, quality='low'):
    mesh_path = join(dir_in, obj_file)
    print('---------------------------------------------------------------------------')
    print('Processing: ' + obj_file)
    print('---------------------------------------------------------------------------')
    # load the mesh
    print('Loading mesh...')
    mesh = pymesh.load_mesh(mesh_path)
    print('Mesh verts: ' + str(mesh.vertices.shape))
    print('Mesh faces: ' + str(mesh.faces.shape))
    # filter out those that have too many faces
    if (mesh.faces.shape[0] > max_faces):
        print('Over max faces...continuing to next shape!')
        return
    # cleanup
    surf_mesh = mesh
    if make_convex:
        # first tet to get just outer surface
        print('Tetrahedralizing mesh...')
        tetgen = pymesh.tetgen()
        tetgen.points = mesh.vertices; # Input points
        tetgen.triangles = np.empty(0)
        tetgen.verbosity = 0
        tetgen.run(); # Execute tetgen
        surf_mesh = tetgen.mesh
    elif fit_cylinder:
        # same top and bottom radius
        # want to center around origin
        # first find radius (max of all x and z)
        # print (mesh.bbox)
        # print(mesh.bbox[0][0])
        cyl_rad = max([abs(mesh.bbox[0][0]), abs(mesh.bbox[1][0]), abs(mesh.bbox[0][2]), abs(mesh.bbox[1][2])])
        # find height max y - min y
        cyl_half_height = (mesh.bbox[1][1] - mesh.bbox[0][1]) / 2.0
        surf_mesh = pymesh.generate_cylinder(np.array([0, -cyl_half_height, 0]), np.array([0, cyl_half_height, 0]), cyl_rad, cyl_rad)
    # print('Tet verts: ' + str(tet_mesh.vertices.shape))
    # print('Tet faces: ' + str(tet_mesh.faces.shape))
    # remesh to improve vertex distribution for moment calculation
    # print('Remeshing...')
    # surf_mesh = tet_mesh #fix_mesh(tet_mesh, quality)
    print('Surface verts: ' + str(surf_mesh.vertices.shape))
    print('Surface faces: ' + str(surf_mesh.faces.shape))
    # voxelize to find volume
    print('Voxelizing...')
    grid = pymesh.VoxelGrid(grid_size, surf_mesh.dim)
    grid.insert_mesh(surf_mesh)
    grid.create_grid()
    vox_mesh = grid.mesh

    # save sim information
    # the number of voxels will be used for the volume (mass)
    num_voxels = vox_mesh.voxels.shape[0]
    vol = num_voxels * grid_size*grid_size*grid_size

    # move mesh to true COM based on voxelization (centroid of the center of all voxels)
    centroid = np.array([0., 0., 0.])
    for i in range(0, num_voxels):
        # find average position of all vertices defining voxel
        vox_pos = np.mean(vox_mesh.vertices[vox_mesh.voxels[i], :], axis=0)
        centroid += vox_pos
    centroid /= num_voxels

    print('Centroid: (%f, %f, %f)' % (centroid[0], centroid[1], centroid[2]))
    centroid_vox_mesh = pymesh.form_mesh(vox_mesh.vertices - np.reshape(centroid, (1, -1)), vox_mesh.faces, vox_mesh.voxels)
    centroid_surf_mesh = pymesh.form_mesh(surf_mesh.vertices - np.reshape(centroid, (1, -1)), surf_mesh.faces)

    # also calculate moment of inertia around principal axes for a DENSITY of 1 (mass = volume)
    inertia = np.array([0., 0., 0.])
    point_mass = vol / float(num_voxels)
    print('Point mass: %f' % (point_mass))
    for i in range(0, num_voxels):
        # find average position of all vertices defining voxel
        vox_pos = np.mean(centroid_vox_mesh.vertices[centroid_vox_mesh.voxels[i], :], axis=0)
        x2 = vox_pos[0]*vox_pos[0]
        y2 = vox_pos[1]*vox_pos[1]
        z2 = vox_pos[2]*vox_pos[2]
        # inertia += np.array([point_mass*(y2+z2),point_mass*(x2+z2),point_mass*(x2+y2)])
        inertia += np.array([x2*point_mass,y2*point_mass,z2*point_mass])

    print('Num voxels: ' + str(num_voxels))
    print('Volume: ' + str(vol))
    print('Moment of inertia: (%f, %f, %f)' % (inertia[0], inertia[1], inertia[2]))
    json_dict = {'num_vox' : num_voxels, 'vol' : vol, 'inertia' : inertia.tolist()}
    json_out_path = join(dir_out, obj_file.replace('.obj', '.json'))
    with open(json_out_path, 'w') as outfile:
        json_str = json.dump(json_dict, outfile)

    
    # save surface mesh
    mesh_out_path = join(dir_out, obj_file)
    pymesh.save_mesh(mesh_out_path, centroid_surf_mesh)

    # sample point cloud on surface mesh
    print('Sampling point cloud...')
    points_out_path = join(dir_out, obj_file.replace('.obj', '.pts'))
    subprocess.check_output(['./MeshSample', '-n1024', '-s3', mesh_out_path, points_out_path])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_in', default='../shape_net/cleaned_bottles', help='Input directory contained OBJ files to process [default: ../shape_net/cleaned_bottles]')
    parser.add_argument('--dir_out', default='./data_out', help='Ouput directory [default: ./data_out')
    parser.add_argument('--quality', default='low', help='Remeshing quality to use [default: low]')
    parser.add_argument('--grid_size', default=0.025, help='voxel side length to use [default : 0.025]')
    parser.add_argument('--max_faces', default=8000, help='maximum number of faces the object is allowed to have [default : 8000]')
    parser.add_argument('--convex', dest='convex', action='store_true')
    parser.add_argument('--non-convex', dest='convex', action='store_false')
    parser.set_defaults(convex=False)
    parser.add_argument('--fit-cylinder', dest='cylinder', action='store_true')
    parser.add_argument('--no-fit-cylinder', dest='cylinder', action='store_false')
    parser.set_defaults(cylinder=False)
    FLAGS = parser.parse_args()

    dir_in = FLAGS.dir_in 
    dir_out = FLAGS.dir_out 
    quality = FLAGS.quality
    max_faces = FLAGS.max_faces
    grid_size = float(FLAGS.grid_size)
    make_convex = FLAGS.convex
    fit_cylinder = FLAGS.cylinder
    print("Data in from: " + dir_in)
    print("Writing results in: " + dir_out)

    if not exists(dir_out): 
        mkdir(dir_out)

    all_files = [f for f in listdir(dir_in) if isfile(join(dir_in, f))]
    obj_files = [f for f in all_files if f.split('.')[-1] == 'obj']

    # for obj_file in obj_files:
    #     process_obj(obj_file, dir_in, dir_out, grid_size, max_faces, make_convex, fit_cylinder, quality)

    pool_size = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=pool_size, maxtasksperchild=2)
    pool.map(partial(process_obj, dir_in=dir_in, dir_out=dir_out, grid_size=grid_size, max_faces=max_faces, make_convex=make_convex, fit_cylinder=fit_cylinder, quality=quality), obj_files)
    pool.close()
    pool.join()


# mesh = pymesh.load_mesh("model_1ef68777bfdb7d6ba7a07ee616e34cd7.obj")
# print "mesh"
# print mesh.vertices.shape
# print mesh.faces.shape

# print mesh.attribute_names

# surf_mesh = pymesh.compute_outer_hull(mesh)
# print "surf"
# print surf_mesh.vertices.shape
# print surf_mesh.faces.shape

# pymesh.save_mesh("test_surface.obj", surf_mesh)

# # tet_mesh = pymesh.tetrahedralize(surf_mesh, 0.01)
# # print "tet"
# # print tet_mesh.vertices.shape
# # print tet_mesh.faces.shape
# # print tet_mesh.voxels.shape
# # print tet_mesh.attribute_names


# # pymesh.save_mesh("test_tet.obj", tet_mesh)

# grid = pymesh.VoxelGrid(GRID_CELL_SIZE, mesh.dim)
# grid.insert_mesh(mesh)
# grid.create_grid()

# vox_mesh = grid.mesh
# print "vox"
# print vox_mesh.vertices.shape
# print vox_mesh.faces.shape
# print vox_mesh.voxels.shape
# # print vox_mesh.voxels

# # vox_vol = vox_mesh.element_volumes
# # print vox_vol

# pymesh.save_mesh("test_vox.obj", vox_mesh)

# vol = vox_mesh.voxels.shape[0] * GRID_CELL_SIZE*GRID_CELL_SIZE*GRID_CELL_SIZE
# print vol

