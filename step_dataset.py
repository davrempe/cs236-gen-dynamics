import os.path
from os import listdir, mkdir
from os.path import isfile, join, exists
import json
import csv
import numpy as np
import math
import random

POINTS_DIR = 'points'
PCL_PERTURB = 0.0 #0.02
COM_FILTER_MAX = 50
COM_FILTER_MIN = 0.1
ROT_FILTER = 3000

SIM_STEP = 1.0 / 60.0

VIZ = False

if VIZ:
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

class BtDataNormalizationInfo():
    def __init__(self, norm_values):
        '''
        Structure to hold all the normalization values for a dataset.
        - norm_values : list of the norm values in the order in this constructor
        '''
        self.com_max = norm_values[0]
        self.total_rot_max = norm_values[1]
        self.force_vec_max = norm_values[2]
        self.scale_max = norm_values[3]
        self.pc_max = norm_values[4]
        self.force_pos_max = norm_values[5]
        # normalization values for shape-related stuff
        self.density_offset = norm_values[6]
        self.density_max = norm_values[7]
        self.mass_offset = norm_values[8]
        self.mass_max = norm_values[9]
        self.inertia_offset = norm_values[10]
        self.inertia_max = norm_values[11]
        self.init_vel_max = norm_values[12]
        self.init_angvel_max = norm_values[13]
    

class BtStepDataset():
    '''
    Dataset of pushes applied to objects taken from ShapeNet. Data is split based
    on the indices of test set shapes provided. Training data is shuffled, test
    data is not.
    '''
    
    def __init__(self, root, batch_size=32, num_pts=1024, num_steps=6, shuffle_pts=False, filter=False, shuffle_train=True, validation_split=0.2, norm_info=None):
        '''
        Load the dataset.
        - root : root directory of the data. Assumed to have a 'train' and 'test' directory, each with their own
                points/ directory with canonical point clouds for each shape used along with sim_* simulation info. 
        - batch_size : number of data points to return on next_batch
        - num_pts : the number of points in each shape pointcloud
        - num_steps: the number of timesteps to use for each batch
        - shuffle_pts: randomly shuffles ordering of point cloud points every time data is requested
        - train_split : percentage of shapes to use for triaining set
        - use_force_coords: if true, data will be transformed and normalized in
                            coordinate system defined by the applied force vector 
                            in each simulation.
        - rotation_aug: if true, data will be randomly rotated around the up axis. Cannot use with force coords.
        '''
                
        self.root = root
        self.batch_size = batch_size
        self.num_pts = num_pts
        self.shuffle_train = shuffle_train
        self.shuffle_pts = shuffle_pts
        self.num_steps = num_steps
        
        # load all point clouds from points/ dir into single array
        print('Loading point clouds...')
        train_points_path = join(self.root, join('train', POINTS_DIR))
        test_points_path = join(self.root, join('test', POINTS_DIR))

        all_files = [f for f in sorted(listdir(train_points_path)) if isfile(join(train_points_path, f))]
        train_pts_files = [join(train_points_path, f) for f in all_files if f.split('.')[-1] == 'pts']
        all_files = [f for f in sorted(listdir(test_points_path)) if isfile(join(test_points_path, f))]
        test_pts_files = [join(test_points_path, f) for f in all_files if f.split('.')[-1] == 'pts']

        pts_files = train_pts_files + test_pts_files

        # print(pts_files)
        self.num_shapes = len(pts_files)
        test_inds = range(len(train_pts_files), self.num_shapes)
        num_eval_shapes = int(validation_split*len(train_pts_files))
        eval_inds = np.random.choice(np.arange(0, len(train_pts_files)), size=num_eval_shapes, replace=False)

        print('%d training shapes' % (len(train_pts_files) - len(eval_inds)))
        print('%d validation shapes' % (len(eval_inds)))
        print('%d testing shapes' % (len(test_inds)))

        self.point_cloud = np.zeros((self.num_shapes, num_pts, 3))
        for i, pc_file in enumerate(pts_files):
            print(pc_file)
            with open(pc_file, 'r') as f:
                reader = csv.reader(f, delimiter=' ')
                for j, row in enumerate(reader):
                    pt = [float(x) for x in row[0:3]]
                    self.point_cloud[i, j, :] = pt
        print('Point clouds loaded!')

        # set data size by taking first pass through data
        self.data_size = 0
        obj_sim_dirs = [f.replace('.pts', '').replace('/points', '') for f in pts_files]
        for i, obj_dir in enumerate(obj_sim_dirs):
            sim_all_files = [f for f in listdir(obj_dir) if isfile(join(obj_dir, f))]
            sim_json_files = [join(obj_dir, f) for f in sim_all_files if f.split('.')[-1] == 'json']
            self.data_size += len(sim_json_files)

        # print(obj_sim_dirs)
        # sim0_all_files = [f for f in listdir(obj_sim_dirs[0]) if isfile(join(obj_sim_dirs[0], f))]
        # sim0_json_files = [f for f in sim0_all_files if f.split('.')[-1] == 'json']
        # # print(sim0_json_files)
        # sims_per_obj = len(sim0_json_files)
        # self.data_size = sims_per_obj * self.num_shapes
        # print('Found ' + str(sims_per_obj) + ' sims per obbject.')
        print('Found ' + str(self.data_size) + ' total sims.')
                
        # next load all simulation data
        # TODO there's no reason to assume every obj has same number of sims, can just precount...
        print('Loading simulation data...')
        self.shape_name = [] 
        self.shape_idx = np.zeros((self.data_size), dtype=np.int32)
        self.scale = np.zeros((self.data_size, 3))
        self.final_com = np.zeros((self.data_size, 3))
        self.total_rot = np.zeros((self.data_size, 3))
        self.force_vec = np.zeros((self.data_size, 3))
        self.force_pos = np.zeros((self.data_size, 3))
        self.init_vel = np.zeros((self.data_size, 3))
        self.init_angvel = np.zeros((self.data_size, 3))
        self.init_rot = np.zeros((self.data_size, 3))
        self.density = np.zeros((self.data_size), dtype=float)
        self.mass = np.zeros((self.data_size), dtype=float)
        self.inertia = np.zeros((self.data_size, 3))

        # time-series data
        self.step_vel = []
        self.step_angvel = []
        self.step_pos = []
        self.step_rot = []
        self.step_eulerrot = []
        self.step_pos = []

        step_sum = 0
        cur_idx = 0
        for i, obj_dir in enumerate(obj_sim_dirs):
            sim_all_files = [f for f in listdir(obj_dir) if isfile(join(obj_dir, f))]
            sim_json_files = [join(obj_dir, f) for f in sim_all_files if f.split('.')[-1] == 'json']
            for j, sim_file in enumerate(sim_json_files):
                with open(sim_file, 'r') as f:
                    sim_dict = json.loads(f.readline())
                    self.shape_name.append(sim_dict['shape'])
                    self.shape_idx[cur_idx] = i
                    self.scale[cur_idx] = self.load_json_vec(sim_dict['scale'])
                    self.final_com[cur_idx] = self.load_json_vec(sim_dict['comf'])
                    self.total_rot[cur_idx] = self.load_json_vec(sim_dict['totalRot'])
                    self.force_vec[cur_idx] = self.load_json_vec(sim_dict['forceVec'])
                    self.force_pos[cur_idx] = self.load_json_vec(sim_dict['forcePoint'])
                    self.init_vel[cur_idx] = self.load_json_vec(sim_dict['vel0'])
                    self.init_angvel[cur_idx] = self.load_json_vec(sim_dict['angvel0'])
                    self.init_rot[cur_idx] = self.load_json_vec(sim_dict['eulerrot0'])
                    self.density[cur_idx] = float(sim_dict['density'])
                    self.mass[cur_idx] = float(sim_dict['mass'])
                    self.inertia[cur_idx] = self.load_json_vec(sim_dict['inertia'])

                    self.step_vel.append(self.load_json_vec_list(sim_dict['stepVel']))     
                    self.step_angvel.append(self.load_json_vec_list(sim_dict['stepAngVel']))
                    self.step_pos.append(self.load_json_vec_list(sim_dict['stepPos']))
                    self.step_rot.append(self.load_json_vec_list(sim_dict['stepRot']))
                    self.step_eulerrot.append(self.load_json_vec_list(sim_dict['stepEulerRot']))
                    step_sum += len(self.step_vel[-1])

                    cur_idx += 1
                
        self.shape_name = np.array(self.shape_name)
        step_sum /= float(cur_idx)
        print('Simulation data loaded!')
        print('Avg num timesteps: ' + str(step_sum))
        print('Size before filtering: ' + str(self.shape_idx.shape[0]))

        if filter:
            # clean up data 
            # first outlier COM runs      
            cur_inds_to_keep = (abs(np.sqrt(np.sum(self.final_com[:,[0,2]]**2, axis=1))) < COM_FILTER_MAX)
            
            self.shape_name = self.shape_name[cur_inds_to_keep]
            self.shape_idx = self.shape_idx[cur_inds_to_keep]
            self.scale = self.scale[cur_inds_to_keep]
            self.final_com = self.final_com[cur_inds_to_keep]
            self.total_rot = self.total_rot[cur_inds_to_keep]
            self.force_vec = self.force_vec[cur_inds_to_keep]
            self.force_pos = self.force_pos[cur_inds_to_keep]
            self.init_vel = self.init_vel[cur_inds_to_keep]
            self.init_angvel = self.init_angvel[cur_inds_to_keep]
            self.init_rot = self.init_rot[cur_inds_to_keep]
            self.density = self.density[cur_inds_to_keep]
            self.mass = self.mass[cur_inds_to_keep]
            self.inertia = self.inertia[cur_inds_to_keep]

            self.step_vel = [self.step_vel[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_angvel = [self.step_angvel[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_pos = [self.step_pos[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_rot = [self.step_rot[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_eulerrot = [self.step_eulerrot[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]

            self.data_size = self.shape_idx.shape[0]      
            print('Size after COM MAX filtering: ' + str(self.data_size))

            cur_inds_to_keep = (abs(np.sqrt(np.sum(self.final_com[:,[0,2]]**2, axis=1))) > COM_FILTER_MIN)
            
            self.shape_name = self.shape_name[cur_inds_to_keep]
            self.shape_idx = self.shape_idx[cur_inds_to_keep]
            self.scale = self.scale[cur_inds_to_keep]
            self.final_com = self.final_com[cur_inds_to_keep]
            self.total_rot = self.total_rot[cur_inds_to_keep]
            self.force_vec = self.force_vec[cur_inds_to_keep]
            self.force_pos = self.force_pos[cur_inds_to_keep]
            self.init_vel = self.init_vel[cur_inds_to_keep]
            self.init_angvel = self.init_angvel[cur_inds_to_keep]
            self.init_rot = self.init_rot[cur_inds_to_keep]
            self.density = self.density[cur_inds_to_keep]
            self.mass = self.mass[cur_inds_to_keep]
            self.inertia = self.inertia[cur_inds_to_keep]

            self.step_vel = [self.step_vel[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_angvel = [self.step_angvel[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_pos = [self.step_pos[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_rot = [self.step_rot[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_eulerrot = [self.step_eulerrot[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]

            self.data_size = self.shape_idx.shape[0]      
            print('Size after COM MIN filtering: ' + str(self.data_size))

            cur_inds_to_keep = (abs(self.total_rot[:,1]) < ROT_FILTER)
            
            self.shape_name = self.shape_name[cur_inds_to_keep]
            self.shape_idx = self.shape_idx[cur_inds_to_keep]
            self.scale = self.scale[cur_inds_to_keep]
            self.final_com = self.final_com[cur_inds_to_keep]
            self.total_rot = self.total_rot[cur_inds_to_keep]
            self.force_vec = self.force_vec[cur_inds_to_keep]
            self.force_pos = self.force_pos[cur_inds_to_keep]
            self.init_vel = self.init_vel[cur_inds_to_keep]
            self.init_angvel = self.init_angvel[cur_inds_to_keep]
            self.init_rot = self.init_rot[cur_inds_to_keep]
            self.density = self.density[cur_inds_to_keep]
            self.mass = self.mass[cur_inds_to_keep]
            self.inertia = self.inertia[cur_inds_to_keep]

            self.step_vel = [self.step_vel[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_angvel = [self.step_angvel[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_pos = [self.step_pos[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_rot = [self.step_rot[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            self.step_eulerrot = [self.step_eulerrot[i] for i, keep_ind in enumerate(cur_inds_to_keep) if keep_ind]
            
            self.data_size = self.shape_idx.shape[0]      
            print('Size after COM AND ROT filtering: ' + str(self.data_size))
        
        # list of data indices in each split
        in_test_split = np.array([(x in test_inds) for x in self.shape_idx])
        self.test_inds = np.nonzero(in_test_split)[0]
        self.test_data_size = self.test_inds.shape[0]

        in_eval_split = np.array([(x in eval_inds) for x in self.shape_idx])
        self.eval_inds = np.nonzero(in_eval_split)[0]
        self.eval_data_size = self.eval_inds.shape[0]

        not_in_train_split = np.array([(x in test_inds or x in eval_inds) for x in self.shape_idx])
        self.train_inds = np.where(not_in_train_split == False)[0]
        self.train_data_size = self.train_inds.shape[0]

        print('Training size: ' + str(self.train_data_size))
        print('Validation size: ' + str(self.eval_data_size))
        print('Testing size: ' + str(self.test_data_size))
        
        # normalize the data and print out set statistics
        self.norm_info = norm_info
        self.normalize_data(print_stats=True, norm_info=self.norm_info)
        
        # prepare to iterate through data
        self.reset_train()
        self.reset_eval()
        self.reset_test()
            
    def normalize_data(self, print_stats=False, norm_info=None):  
        print('Normalizing data...')
        if norm_info is None:
            norm_values = []
            # only care about x-z for COM, use max norm
            com_norms = np.sqrt(np.sum(self.final_com[:,[0,2]]**2, axis=1))
            self.com_max = np.max(com_norms)
            norm_values.append(self.com_max)
            # calculate some stats that we want on the pre-normalized data
            if print_stats:
                avg_com_norms = np.mean(com_norms)
                med_com_norms = np.median(com_norms)
                std_com_norms = np.std(com_norms)
                avg_total_rot = np.mean(self.total_rot[:, 1])
                std_total_rot = np.std(self.total_rot[:, 1])
            
            # only need around y axis
            self.total_rot_max = np.max(np.abs(self.total_rot[:, 1]))
            norm_values.append(self.total_rot_max)
            # again only care about x-z, use max norm
            self.force_vec_max = np.max(np.sqrt(np.sum(self.force_vec[:,[0,2]]**2, axis=1)))
            print('avg force vec: ' + str(np.mean(np.linalg.norm(self.force_vec[:, [0, 2]], axis=1))))
            print('std force vec: ' + str(np.std(np.linalg.norm(self.force_vec[:, [0, 2]], axis=1))))
            if self.force_vec_max == 0:
                self.force_vec_max = 1
            norm_values.append(self.force_vec_max)
            
            # maximum scale in any dimension, used to normalize point clouds
            self.scale_max = np.max(self.scale)
            norm_values.append(self.scale_max)
            # for point cloud, must find maximum possible norm of any point
            # first centroid each pointcloud
            # centroids = np.reshape(np.mean(self.point_cloud, axis=1), (self.num_shapes, 1, 3))
            # self.point_cloud -= centroids
            # centroids = np.reshape(np.mean(self.point_cloud, axis=1), (self.num_shapes, 1, 3))
            # then we need to find the maximally scaled point clouds to determine this
            pc = np.copy(self.point_cloud)
            self.pc_max = np.max(np.sqrt(np.sum((pc*self.scale_max)**2, axis=2)))
            norm_values.append(self.pc_max)
            # finally scale each point cloud so we have a normalized canonical version for
            # each shape. We will scale accordingly on each get_item() call
            self.point_cloud /= self.pc_max

            # use same as for point cloud 
            self.force_pos_max = self.pc_max #np.max(np.sqrt(np.sum(self.force_pos**2, axis=1)))
            norm_values.append(self.force_pos_max)

            # normalization values for shape-related stuff
            self.density_offset = np.min(self.density)
            self.density_max = np.max(self.density) - self.density_offset
            if self.density_max == 0:
                self.density_offset = 0
                self.density_max = np.max(self.density)
            norm_values.append(self.density_offset)
            norm_values.append(self.density_max)
            self.mass_offset = np.min(self.mass)
            self.mass_max = np.max(self.mass) - self.mass_offset
            if self.mass_max == 0:
                self.mass_max = self.mass_offset
                self.mass_offset = 0
            norm_values.append(self.mass_offset)
            norm_values.append(self.mass_max)
            self.inertia_offset = np.min(self.inertia[:, 1])
            self.inertia_max = np.max(self.inertia[:, 1]) - self.inertia_offset 
            if self.inertia_max == 0:
                self.inertia_max = self.inertia_offset
                self.inertia_offset = 0
            norm_values.append(self.inertia_offset)
            norm_values.append(self.inertia_max)     
            
            # calculate these just to print out, but we don't actually use them
            self.init_vel_max = np.max(np.sqrt(np.sum(self.init_vel[:,[0,2]]**2, axis=1)))
            if self.init_vel_max == 0:
                self.init_vel_max = 1
            self.init_angvel_max = np.max(self.init_angvel[:, 1]) # only care about y
            if self.init_angvel_max == 0:
                self.init_angvel_max = 1
            norm_values.append(self.init_vel_max)
            norm_values.append(self.init_angvel_max)

            self.norm_info = BtDataNormalizationInfo(norm_values)
        else:
            self.com_max = norm_info.com_max
            self.total_rot_max = norm_info.total_rot_max
            self.force_vec_max = norm_info.force_vec_max
            self.scale_max = norm_info.scale_max
            self.pc_max = norm_info.pc_max
            self.point_cloud /= self.pc_max
            self.force_pos_max = norm_info.force_pos_max
            self.density_offset = norm_info.density_offset
            self.density_max = norm_info.density_max
            self.mass_offset = norm_info.mass_offset
            self.mass_max = norm_info.mass_max
            self.inertia_offset = norm_info.inertia_offset
            self.inertia_max = norm_info.inertia_max
            self.init_vel_max = norm_info.init_vel_max
            self.init_angvel_max = norm_info.init_angvel_max

            com_norms = np.sqrt(np.sum(self.final_com[:,[0,2]]**2, axis=1))
            # calculate some stats that we want on the pre-normalized data
            if print_stats:
                avg_com_norms = np.mean(com_norms)
                med_com_norms = np.median(com_norms)
                std_com_norms = np.std(com_norms)
                avg_total_rot = np.mean(self.total_rot[:, 1])
                std_total_rot = np.std(self.total_rot[:, 1])

            self.norm_info = norm_info

        # some extra analysis to determine how consistent the dataset is
        if VIZ:
            vel_norm = np.linalg.norm(self.init_vel, axis=1)
            vel_sqr = vel_norm*vel_norm
            dist = np.linalg.norm(self.final_com[:, [0,2]], axis=1)

            plt.figure()
            plt.scatter(vel_sqr, dist, s=0.1)
            plt.xlabel('Vel^2')
            plt.ylabel('Dist')
            plt.title(self.root)

            line_fit = np.polyfit(vel_sqr, dist, 1, full=True)
            line_fit = line_fit[0]
            line_func = np.poly1d(line_fit)
            line_fit_y = line_func(vel_sqr)

            data_slope = line_func.deriv()
            print('Slope: ' + str(data_slope))
            dist_diff = dist - line_fit_y
            dist_rmse = np.sqrt(np.mean(dist_diff*dist_diff))
            print('dist RMSE: '  + str(dist_rmse))

            plt.plot(vel_sqr, line_fit_y, 'r')
            plt.xlim((0.5, 5))
            plt.ylim((0.5, 5))
            plt.savefig('./data/dataset_statistics/' + self.root.split('/')[-1] + '_trans.png')
            plt.show()

            angvel_sqr = self.init_angvel[:, 1]*self.init_angvel[:, 1]
            theta = np.abs(self.total_rot[:, 1])
            quad_fit = np.polyfit(angvel_sqr, theta, 2, full=True)
            quad_fit = quad_fit[0]
            quad_func = np.poly1d(quad_fit)
            quad_fit_y = quad_func(angvel_sqr)

            rot_diff = theta - quad_fit_y
            rot_rmse = np.sqrt(np.mean(rot_diff*rot_diff))
            print('rot RMSE: '  + str(rot_rmse))

            plt.figure()
            plt.scatter(angvel_sqr, theta, s=0.1)
            plt.scatter(angvel_sqr, quad_fit_y, s=0.5)

            plt.xlim((-20, 800))
            plt.ylim((-20, 3000))
            plt.xlabel('AngVel^2')
            plt.ylabel('Theta')
            plt.title(self.root)
            plt.savefig('./data/dataset_statistics/' + self.root.split('/')[-1] + '_rot.png')
            plt.show()

            # total rotation distribution
            plt.figure()
            plt.hist(dist, 100)
            plt.xlabel('Total distance (m)')
            plt.ylabel('Count')
            plt.title(self.root)
            plt.savefig('./data/dataset_statistics/' + self.root.split('/')[-1] + '_pos_distrib.png')
            plt.show()

            # total rotation distribution
            plt.figure()
            plt.hist(self.total_rot[:, 1], 100)
            plt.xlabel('Total rot (deg)')
            plt.ylabel('Count')
            plt.title(self.root)
            plt.savefig('./data/dataset_statistics/' + self.root.split('/')[-1] + '_rot_distrib.png')
            plt.show()

            plt.figure()
            plt.hist(np.abs(self.total_rot[:, 1]) / self.total_rot_max, 100, density=True)
            plt.xlabel('Total rot (normalized)')
            plt.ylabel('Count (normalized)')
            plt.title(self.root)
            plt.savefig('./data/dataset_statistics/' + self.root.split('/')[-1] + '_norm_rot_distrib.png')
            plt.show()
        
        #
        # Now do the actual normalization and transformation to force coords if needed.
        #
        # if self.use_force_coords or self.rotation_aug:
        #     self.pinv = self.init_com = np.zeros((self.data_size, 9))

        # for i in range(0, self.data_size):
        #     if self.use_force_coords or self.rotation_aug:
        #         Pinv, rot_y, com, force_pos, force_vec, init_vel, init_angvel, density, mass, inertia_y = \
        #                         self.precomp_item(i)
        #         self.pinv[i] = np.reshape(Pinv, (9))
        #     else:

        for i in range(0, self.data_size):
            rot_y, com, force_pos, force_vec, init_vel, init_angvel, density, mass, inertia_y, step_vel, step_angvel, step_pos, step_rot, step_eulerrot = \
                                self.precomp_item(i)
                    
            self.total_rot[i, 1] = rot_y
            self.final_com[i] = com
            self.force_vec[i] = force_vec
            self.force_pos[i] = force_pos
            self.init_vel[i] = init_vel
            self.init_angvel[i] = init_angvel
            self.density[i] = density 
            self.mass[i] = mass
            self.inertia[i, 1] = inertia_y
            
            self.step_vel[i] = step_vel
            self.step_angvel[i] = step_angvel
            self.step_pos[i] = step_pos
            self.step_rot[i] = step_rot
            self.step_eulerrot[i] = step_eulerrot
        
        print('Data normalized!')
        
        if print_stats:
            print("Max pc norm: " + str(self.pc_max))
            print("Max scale: " + str(self.scale_max))
            print("Max init vel: " + str(self.init_vel_max))
            print("Max init ang vel: " + str(self.init_angvel_max))
            print("Max total rot: " + str(self.total_rot_max))
            print("Avg total rot: " + str(avg_total_rot))
            print("Std total rot: " + str(std_total_rot))
            print("Max COM: " + str(self.com_max))
            print("Avg COM: " + str(avg_com_norms))
            print("Med COM: " + str(med_com_norms))
            print("Std COM: " + str(std_com_norms))
            print("Max force vec: " + str(self.force_vec_max))
            print("Max force pos: " + str(self.force_pos_max))
            print("Max Mass: " + str(self.mass_max + self.mass_offset))
            print("Min Mass: " + str(self.mass_offset))
            print("Max Density: " + str(self.density_max + self.density_offset))
            print("Max Inertia: " + str(self.inertia_max + self.inertia_offset))

    def precomp_item(self, idx):
        ''' 
        Transforms a single data point to force coordinates if enabled, and
        normalizes. 
        '''
        com = self.final_com[idx]
        rot_y = self.total_rot[idx, 1]
        force_pos = self.force_pos[idx]
        force_vec = self.force_vec[idx]
        init_vel = self.init_vel[idx]
        init_angvel = self.init_angvel[idx]
        density = self.density[idx]
        mass = self.mass[idx]
        inertia = self.inertia[idx, 1]

        step_vel = self.step_vel[idx]
        step_angvel = self.step_angvel[idx]
        step_pos = self.step_pos[idx]
        step_rot = self.step_rot[idx]
        step_eulerrot = self.step_eulerrot[idx]
        
                
        # # transform to correct coordinates
        # if self.use_force_coords or self.rotation_aug:
        #     if self.use_force_coords:
        #         unit_force = force_vec[[0,2]] / np.linalg.norm(force_vec[[0,2]])
        #     elif self.rotation_aug:
        #         # no point using force coordinates if augmenting rotation
        #         # choose random x axis and normalize
        #         unit_force = np.random.uniform(low=-1.0, high=1.0, size=(2))
        #         unit_force /= np.linalg.norm(unit_force)
        #     # construct transition matrix (force vector is new x axis)
        #     force_perp = np.array([unit_force[1], -unit_force[0]])
        #     P = np.array([[unit_force[0], 0., force_perp[0]],
        #                   [0., 1., 0.],
        #                   [unit_force[1], 0., force_perp[1]]])
        #     Pinv = np.linalg.inv(P)
        #     # transform force_pos and vec
        #     force_pos_out = np.dot(Pinv, force_pos)
        #     force_vec_out = np.dot(Pinv, force_vec)
        #     if self.use_force_coords:
        #         force_vec_out[2] = 0 # zero out z for good measure
        #     # transform final position
        #     com_out = np.dot(Pinv, com)
        #     # total rot won't change with different coords so doesn't matter
        #     init_vel_out = np.dot(Pinv, init_vel)
        #     # angular velocity is not transformed since y is still the same
        # else:

        com_out = com
        force_pos_out = force_pos
        force_vec_out = force_vec
        init_vel_out = init_vel
          

        # normalization
        # COM -> [-1, 1]
        com_out = com_out / self.com_max
        # rotation -> [0, 1]
        rot_out = rot_y / self.total_rot_max
        # force vec -> [-1, 1]
        force_vec_out = force_vec_out / self.force_vec_max
        # force pos -> [-1, 1]
        force_pos_out = force_pos_out / self.force_pos_max

        # the following values are in a limited range so we substract out the min
        # density -> [0, 1]
        density_out = (density - self.density_offset) / self.density_max
        # mass -> [0, 1]
        mass_out = (mass - self.mass_offset) / self.mass_max 
        # inertia y value -> [0, 1]
        inertia_out = (inertia - self.inertia_offset) / self.inertia_max

        # linear/angular vel -> [-1, 1]
        init_vel_out = init_vel_out / self.init_vel_max
        init_angvel_out = init_angvel / self.init_angvel_max

        step_vel_out = [(x / self.init_vel_max) for x in step_vel]
        step_angvel_out = [(x / self.init_angvel_max) for x in step_angvel]
        step_pos_out = [(x / self.com_max) for x in step_pos]
        step_eulerrot_out = [(x / self.total_rot_max) for x in step_eulerrot]
        step_rot_out = step_rot # don't need to normalize quaternions will only use this to calc change in rot
        
        # if self.use_force_coords or self.rotation_aug:
        #     return Pinv, rot_out, com_out, force_pos_out, force_vec_out, \
        #                 init_vel_out, init_angvel_out, density_out, mass_out, inertia_out
        # else:

        return rot_out, com_out, force_pos_out, force_vec_out, \
                        init_vel_out, init_angvel_out, density_out, mass_out, inertia_out, \
                        step_vel_out, step_angvel_out, step_pos_out, step_rot_out, step_eulerrot_out
    
    #### NORMALIZATION GETTERS #############
    
    def get_normalization_info(self):
        return self.norm_info

    def get_com_normalization(self):
        return self.com_max
    
    def get_rot_normalization(self):
        return self.total_rot_max
    
    def get_force_vec_normalization(self):
        return self.force_vec_max
    
    def get_force_pos_normalization(self):
        return self.force_pos_max
    
    def get_init_vel_normalization(self):
        return self.init_vel_max
    
    def get_init_angvel_normalization(self):
        return self.init_angvel_max

    def get_mass_normalization(self):
        return self.mass_max

    def get_density_normalization(self):
        return self.density_max

    def get_inertia_normalization(self):
        return self.inertia_max
    
    def get_num_points(self):
        return self.num_pts
    
    def get_pc_normalization(self):
        return self.pc_max

    def unnormalize_density(self, density):
        return density*self.density_max + self.density_offset

    def unnormalize_mass(self, mass):
        return mass*self.mass_max + self.mass_offset

    def unnormalize_inertia_y(self, inertia_y):
        return inertia_y*self.inertia_max + self.inertia_offset
        
    #### RELATED TO GETTING DATA ##########
    def get_test_diameter(self, idx):
        '''
        Approximates the greatest diameter for the shape
        at the given index in the test split.
        '''
        data_idx = self.test_inds[idx]
        # get the point cloud
        pc = np.copy(self.point_cloud[self.shape_idx[data_idx]])
        scale = self.scale[data_idx]
        # scale accordingly
        pc *= np.reshape(scale, (1, -1))
        # find max norm point in (x,z) plane
        max_planar_norm = np.max(np.sqrt(np.sum(pc[:,[0,2]]**2, axis=1)))
        # diameter assumed to be 2x this
        # (only holds for symmetric shapes)
        max_diam = max_planar_norm*2
        # unnormalize
        return max_diam*self.pc_max
        
    
    def get_test_scale_idx(self, idx):
        return self.scale[self.test_inds[idx]]

    def get_train_scale_idx(self, idx):
        return self.scale[self.train_inds[idx]]

    def get_eval_scale_idx(self, idx):
        return self.scale[self.eval_inds[idx]]

    def get_scale_idx(self, idx, split='test'):
        if split=='train':
            return self.get_train_scale_idx(idx)
        elif split=='test':
            return self.get_test_scale_idx(idx)
        elif split=='eval':
            return self.get_eval_scale_idx(idx)
        return

    def get_test_shape_name(self, idx):
        return self.shape_name[self.test_inds[idx]]

    def get_train_shape_name(self, idx):
        return self.shape_name[self.train_inds[idx]]

    def get_eval_shape_name(self, idx):
        return self.shape_name[self.eval_inds[idx]]

    def get_shape_name(self, idx, split='test'):
        if split=='train':
            return self.get_train_shape_name(idx)
        elif split=='test':
            return self.get_test_shape_name(idx)
        elif split=='eval':
            return self.get_eval_shape_name(idx)
        return

    def get_test_full_com(self, idx):
        return self.final_com[self.test_inds[idx]]

    def get_train_full_com(self, idx):
        return self.final_com[self.train_inds[idx]]

    def get_eval_full_com(self, idx):
        return self.final_com[self.eval_inds[idx]]

    def get_full_com(self, idx, split='test'):
        if split=='train':
            return self.get_train_full_com(idx)
        elif split=='test':
            return self.get_test_full_com(idx)
        elif split=='eval':
            return self.get_eval_full_com(idx)
        return

    def get_test_full_step_pos(self, idx):
        return self.step_pos[self.test_inds[idx]]

    def get_train_full_step_pos(self, idx):
        return self.step_pos[self.train_inds[idx]]

    def get_eval_full_step_pos(self, idx):
        return self.step_pos[self.eval_inds[idx]]

    def get_full_step_pos(self, idx, split='test'):
        if split=='train':
            return self.get_train_full_step_pos(idx)
        elif split=='test':
            return self.get_test_full_step_pos(idx)
        elif split=='eval':
            return self.get_eval_full_step_pos(idx)
        return

    
    def get_test_item(self, idx):
        return self.get_item(self.test_inds[idx])

    def get_train_item(self, idx):
        return self.get_item(self.train_inds[idx])

    def get_eval_item(self, idx):
        return self.get_item(self.eval_inds[idx])

    def get_data_item(self, idx, split='test'):
        if split=='train':
            return self.get_train_item(idx)
        elif split=='test':
            return self.get_test_item(idx)
        elif split=='eval':
            return self.get_eval_item(idx)
        return
    
    def reset_train(self):
        ''' Prepares data loader to iterate through the training split. '''
        # shuffle training data
        if self.shuffle_train:
            np.random.shuffle(self.train_inds)
        self.num_train_batches = (self.train_data_size + self.batch_size - 1) // self.batch_size
        self.train_batch_idx = 0

    def reset_eval(self):
        ''' Prepares data loader to iterate through the validation split. '''
        # don't shuffle validation
        self.num_eval_batches = (self.eval_data_size + self.batch_size - 1) // self.batch_size
        self.eval_batch_idx = 0
        
    def reset_test(self):
        ''' Prepares data loader to iterate through the test split. '''
        # don't shuffle test split
        self.num_test_batches = (self.test_data_size + self.batch_size - 1) // self.batch_size
        self.test_batch_idx = 0

    def reset_split(self, split='test'):
        if split=='train':
            return self.reset_train()
        elif split=='test':
            return self.reset_test()
        elif split=='eval':
            return self.reset_eval()
        return
        
    def get_item(self, idx):
        # get the normalized canonical point cloud for this simulation
        pc = np.copy(self.point_cloud[self.shape_idx[idx]])
        scale = self.scale[idx]
        # scale accordingly
        pc *= np.reshape(scale, (1, -1))

        # randomly perturb point cloud
        pc += np.random.normal(0.0, PCL_PERTURB, pc.shape)
        
        # if self.use_force_coords or self.rotation_aug:
        #     # initial rotation matrix
        #     yrot = self.init_rot[idx, 1]
        #     R = np.array([[math.cos(math.radians(yrot)), 0., math.sin(math.radians(yrot))],
        #                   [0., 1., 0.],
        #                   [-math.sin(math.radians(yrot)), 0., math.cos(math.radians(yrot))]])
        #     Pinv = np.reshape(self.pinv[idx], (3, 3)).dot(R)
        #     # transform point cloud
        #     pc = np.dot(pc, Pinv.T)

        if self.shuffle_pts:
            # randomly shuffle point cloud pts
            np.random.shuffle(pc)

        # visualize
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(pc[:, 0], pc[:, 2], pc[:, 1], c='r', marker='o')
        # ax.set_xlim(-self.pc_max, self.pc_max)
        # ax.set_ylim(-self.pc_max, self.pc_max)
        # ax.set_zlim(-self.pc_max, self.pc_max)
        # ax.set_xlabel('X')
        # ax.set_ylabel('Z')
        # ax.set_zlabel('Y')
        # plt.show()
    
        # get other stuff
        # com_out = self.final_com[idx]
        # rot_out = self.total_rot[idx, 1]
        # force_pos_out = self.force_pos[idx]
        # force_vec_out = self.force_vec[idx]
        # init_vel_out = self.init_vel[idx]
        # init_angvel_out = self.init_angvel[idx]

        # inertia_out = self.inertia[idx, 1]
        # mass_out = self.mass[idx]

        # randomly choose a size num_steps sequence from the simulation to return time-series data
        total_steps = len(self.step_vel[idx])
        max_start_step = total_steps - self.num_steps
        if max_start_step < 0:
            # print('Found simulation shorter than num_steps parameter!')
            # quit()
            step_vel_out = np.zeros((self.num_steps, 3), dtype=float)
            step_angvel_out = np.zeros((self.num_steps, 3), dtype=float)
            step_pos_out = np.zeros((self.num_steps, 3), dtype=float)
            step_eulerrot_out = np.zeros((self.num_steps, 3), dtype=float)
            # pad ending with last value
            step_vel_out[0:total_steps] = np.array(self.step_vel[idx][0:total_steps])
            step_vel_out[total_steps:self.num_steps] = step_vel_out[total_steps-1]
            step_angvel_out[0:total_steps] = np.array(self.step_angvel[idx][0:total_steps])
            step_angvel_out[total_steps:self.num_steps] = step_angvel_out[total_steps-1]
            step_pos_out[0:total_steps] = np.array(self.step_pos[idx][0:total_steps])
            step_pos_out[total_steps:self.num_steps] = step_pos_out[total_steps-1]
            step_eulerrot_out[0:total_steps] = np.array(self.step_eulerrot[idx][0:total_steps])
            step_eulerrot_out[total_steps:self.num_steps] = step_eulerrot_out[total_steps-1]
        else:
            start_step = random.randint(0, max_start_step)
            end_step = start_step + self.num_steps
            # print('Range: %d, %d' % (start_step, end_step))
            step_vel_out = np.array(self.step_vel[idx][start_step:end_step])
            step_angvel_out = np.array(self.step_angvel[idx][start_step:end_step])
            step_pos_out = np.array(self.step_pos[idx][start_step:end_step])
            step_eulerrot_out = np.array(self.step_eulerrot[idx][start_step:end_step])
        
        return pc, step_vel_out, step_angvel_out, step_pos_out, step_eulerrot_out
    
    def has_next_train_batch(self):
        return self.train_batch_idx < self.num_train_batches

    def has_next_eval_batch(self):
        return self.eval_batch_idx < self.num_eval_batches
    
    def has_next_test_batch(self):
        return self.test_batch_idx < self.num_test_batches

    def has_next_batch(self, split='test'):
        if split=='train':
            return self.has_next_train_batch()
        elif split=='test':
            return self.has_next_test_batch()
        elif split=='eval':
            return self.has_next_eval_batch()
        return
    
    def next_train_batch(self):
        batch = self.next_batch(self.train_batch_idx, self.train_data_size, self.train_inds)
        self.train_batch_idx += 1
        return batch

    def next_eval_batch(self):
        batch = self.next_batch(self.eval_batch_idx, self.eval_data_size, self.eval_inds)
        self.eval_batch_idx += 1
        return batch
    
    def next_test_batch(self):
        batch = self.next_batch(self.test_batch_idx, self.test_data_size, self.test_inds)
        self.test_batch_idx += 1
        return batch

    def next_data_batch(self, split='test'):
        if split=='train':
            return self.next_train_batch()
        elif split=='test':
            return self.next_test_batch()
        elif split=='eval':
            return self.next_eval_batch()
        return

    def next_batch(self, batch_idx, data_size, iter_inds):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = batch_idx * self.batch_size
        end_idx = min((batch_idx+1) * self.batch_size, data_size)
        bsize = end_idx - start_idx
        batch_pc = np.zeros((bsize, self.num_pts, 3))
        # batch_final_rot = np.zeros((bsize))
        # batch_final_com = np.zeros((bsize, 3))
        # batch_force_vec = np.zeros((bsize, 3))
        # batch_force_pos = np.zeros((bsize, 3))
        # batch_init_vel = np.zeros((bsize, 3))
        # batch_init_angvel = np.zeros((bsize, 3))
        # batch_mass = np.zeros((bsize))
        # batch_inertia = np.zeros((bsize))

        batch_step_vel = np.zeros((bsize,self.num_steps,3))
        batch_step_angvel = np.zeros((bsize,self.num_steps,3))
        batch_step_pos = np.zeros((bsize,self.num_steps,3))
        batch_step_rot = np.zeros((bsize,self.num_steps,3))

        for i in range(bsize):
            pc, step_vel, step_angvel, step_pos, step_rot = \
                            self.get_item(iter_inds[i+start_idx])
            batch_pc[i] = pc
            # batch_final_rot[i] = rot_y
            # batch_final_com[i] = com
            # batch_force_vec[i] = force_vec
            # batch_force_pos[i] = force_pos
            # batch_init_vel[i] = init_vel
            # batch_init_angvel[i] = init_angvel
            # batch_mass[i] = mass
            # batch_inertia[i] = inertia

            batch_step_vel[i] = step_vel
            batch_step_angvel[i] = step_angvel
            batch_step_pos[i] = step_pos
            batch_step_rot[i] = step_rot

            # print(step_pos)

        return batch_pc, batch_step_vel[:,:,[0,2]], batch_step_angvel[:,:,1], batch_step_pos[:,:,[0,2]], batch_step_rot[:,:,1]
        
    #### HELPERS ##########################
        
    def load_json_vec(self, vec_dict):
        ''' Loads a json 3 (x, y, z) or 4 (x, y, z, w) vector into a numpy array '''
        np_vec = np.zeros((len(vec_dict)))
        if len(vec_dict) == 3:
            np_vec = np.array([vec_dict['x'], vec_dict['y'], vec_dict['z']])
        elif len(vec_dict) == 4:
            np_vec = np.array([vec_dict['x'], vec_dict['y'], vec_dict['z'], vec_dict['w']])
        return np_vec

    def load_json_vec_list(self, vec_list):
        return [self.load_json_vec(vec_dict) for vec_dict in vec_list]
    

##################### TESTING #########################

if __name__ == '__main__':
#     all_bott = [0, 1, 2, 4, 5, 8, 9, 10, 11, 12]
#     d = ShapeNetDataset(root='./data/bottle_test_5k/', prefix='bottle', batch_size=1, 
#                               num_shapes=13, num_pts=1024, num_sims=5000, \
#                                 use_force_coords=True, test_inds=all_bott)
#     for i in all_bott:
#         print('BOTTLE ' + str(i) + '-------------------------------------')
#         d = ShapeNetDataset(root='./data/bottle_test_5k/', prefix='bottle', batch_size=1, 
#                               num_shapes=13, num_pts=1024, num_sims=5000, \
#                                 use_force_coords=True, test_inds=[i])
#         print('--------------------------------------------------')
        
    d = BtStepDataset(root='./data/step_trashcans_long_split/', batch_size=1, num_pts=1024, num_steps=25, shuffle_train=True, shuffle_pts=False, filter=True, validation_split=0.2)
    
       
    count = 0
    while d.has_next_train_batch():
        d.next_train_batch()
        # print(d.get_scale_idx(count, 'train'))
        count += 1
        
    print count

    count = 0
    while d.has_next_eval_batch():
        d.next_eval_batch()
        count += 1
        
    print count
    
    count = 0
    while d.has_next_test_batch():
        d.next_test_batch()
        # print(d.get_scale_idx(count, 'test'))
        count += 1
        
    print count
        
    count = 0
    d.reset_train()
    while d.has_next_train_batch():
        d.next_train_batch()
        count += 1
        
    print count

    count = 0
    d.reset_eval()
    while d.has_next_eval_batch():
        d.next_eval_batch()
        count += 1
        
    print count
    
    count = 0
    d.reset_test()
    while d.has_next_test_batch():
        d.next_test_batch()
        count += 1
        
    print count