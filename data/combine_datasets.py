import argparse

from os import listdir, mkdir, remove
from os.path import isfile, join, exists
from shutil import copy, copytree


parser = argparse.ArgumentParser()
parser.add_argument('--datasets_in', default='./datasets_in')
parser.add_argument('--new_data_out', default='./combine_data_out')
FLAGS = parser.parse_args()

dataset_in_path = FLAGS.datasets_in
datset_out_path = FLAGS.new_data_out

TRAIN_DIR = 'train'
TEST_DIR = 'test'
POINTS_DIR = 'points'

component_datasets = [join(dataset_in_path, f) for f in listdir(dataset_in_path)]
# check if our destinations exists and set up inside
if not exists(datset_out_path): 
    mkdir(datset_out_path)
train_path = join(datset_out_path, TRAIN_DIR)
if not exists(train_path):
    mkdir(train_path)
test_path = join(datset_out_path, TEST_DIR)
if not exists(test_path):
    mkdir(test_path)
train_pts_path = join(train_path, POINTS_DIR)
if not exists(train_pts_path):
    mkdir(train_pts_path)
test_pts_path = join(test_path, POINTS_DIR)
if not exists(test_pts_path):
    mkdir(test_pts_path)

# copy all contents from each component dataset
for dataset in component_datasets:
    print('Copying ' + str(dataset) + '...')
    cur_train_path = join(dataset, TRAIN_DIR)
    cur_test_path = join(dataset, TEST_DIR)
    cur_train_pts = join(cur_train_path, POINTS_DIR)
    cur_test_pts = join(cur_test_path, POINTS_DIR)

    # train sims and pts
    all_train_shapes = [f for f in listdir(cur_train_path) if f != POINTS_DIR]
    for train_dir in all_train_shapes:
        copytree(join(cur_train_path, train_dir), join(train_path, train_dir))

    all_train_pts = [join(cur_train_pts, f) for f in listdir(cur_train_pts)]
    for pts_file in all_train_pts:
        copy(pts_file, train_pts_path)

    # test sims and pts
    all_test_shapes = [f for f in listdir(cur_test_path) if f != POINTS_DIR]
    for test_dir in all_test_shapes:
        copytree(join(cur_test_path, test_dir), join(test_path, test_dir))

    all_test_pts = [join(cur_test_pts, f) for f in listdir(cur_test_pts)]
    for pts_file in all_test_pts:
        copy(pts_file, test_pts_path)
    
