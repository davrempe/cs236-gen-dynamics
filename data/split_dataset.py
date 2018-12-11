import argparse

from os import listdir, mkdir
from os.path import isfile, join, exists
from shutil import copy, copytree


parser = argparse.ArgumentParser()
parser.add_argument('--train_fraction', default=0.8)
parser.add_argument('--data_in', default='./data_in')
parser.add_argument('--data_out', default='./data_in')
FLAGS = parser.parse_args()

train_fraction = float(FLAGS.train_fraction)
data_in_dir = FLAGS.data_in
data_out_dir = FLAGS.data_out

POINTS_DIR = 'points'
TRAIN_DIR = 'train'
TEST_DIR = 'test'

# check if our destinations exists
if not exists(data_out_dir): 
    mkdir(data_out_dir)
train_path = join(data_out_dir, TRAIN_DIR)
if not exists(train_path):
    mkdir(train_path)
test_path = join(data_out_dir, TEST_DIR)
if not exists(test_path):
    mkdir(test_path)
train_pts_path = join(train_path, POINTS_DIR)
if not exists(train_pts_path):
    mkdir(train_pts_path)
test_pts_path = join(test_path, POINTS_DIR)
if not exists(test_pts_path):
    mkdir(test_pts_path)

# look in the points directory to find all the simulated shapes
points_path = join(data_in_dir, POINTS_DIR)
all_files = [f for f in sorted(listdir(points_path)) if isfile(join(points_path, f))]
pts_files = [f for f in all_files if f.split('.')[-1] == 'pts']
# print(pts_files)
num_shapes = len(pts_files)

# create splits
last_train_idx = int(num_shapes * train_fraction)
test_inds = range(last_train_idx, num_shapes)
train_inds = range(0, last_train_idx)

# copy data over
for idx in train_inds:
    # first pts file
    pts_file = pts_files[idx]
    copy(join(points_path, pts_file), train_pts_path)
    # now the sim data
    prefix = pts_file.split('.')[0]
    sim_dir = join(data_in_dir, prefix)
    copytree(sim_dir, join(train_path, prefix))
for idx in test_inds:
    # first pts file
    pts_file = pts_files[idx]
    copy(join(points_path, pts_file), test_pts_path)
    # now the sim data
    prefix = pts_file.split('.')[0]
    sim_dir = join(data_in_dir, prefix)
    copytree(sim_dir, join(test_path, prefix))
