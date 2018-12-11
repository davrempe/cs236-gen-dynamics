'''
    Single-GPU training.
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='step_rnn', help='Model name [default: step_rnn]')
parser.add_argument('--data_root', default='./data/step_cube_scaled_totrot_5k/', help='Dataset root [default: ./data/step_cube_scaled_totrot_5k/]')
parser.add_argument('--alt_eval_data_root', default=None, help='Dataset to evaluate root, if not given will evaluate on trained dataset [default: None]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--log_sub_dir', default=None)
parser.add_argument('--num_point', type=int, default=1024, help='Number of Points in Point Cloud [default: 1024]')
parser.add_argument('--num_steps', type=int, default=25, help='Number of steps to roll out on [default: 25]')
parser.add_argument('--batch_size', type=int, default=1, help='Batch Size during training [default: 32]')
parser.add_argument('--model_path_test', default='log/log_1539724433/planar_force_pos_best_model.ckpt', help='model checkpoint file path')
parser.add_argument('--output_pred', dest='output_pred', action='store_true')
parser.set_defaults(output_pred=False)
parser.add_argument('--sample_mean', dest='sample_mean', action='store_true')
parser.set_defaults(sample_mean=False)
parser.add_argument('--single_step_eval', dest='single_step_eval', action='store_true')
parser.set_defaults(single_step_eval=False)
parser.add_argument('--no_viz', dest='viz', action='store_false')
parser.set_defaults(viz=True)

parser.add_argument('--num_units', type=int, default=128, help='Number of units to use in each RNN unit [default: 128]')
parser.add_argument('--cell_type', default='lstm', help='Cell type to use (rnn, gru, or lstm) [default: lstm]')
parser.add_argument('--num_cells', type=int, default=3, help='Number of cells to stack to form the RNN module [default: 3]')

FLAGS = parser.parse_args()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

import math
from datetime import datetime
import time
import random
import json
import numpy as np
import tensorflow as tf
import socket
import importlib
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import tf_util
import step_dataset

EPOCH_CNT = 0

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
NUM_STEPS = FLAGS.num_steps+1 # to step 25 times, need 26 time points
GPU_INDEX = FLAGS.gpu
OUTPUT_PRED = FLAGS.output_pred
SINGLE_STEP_EVAL = FLAGS.single_step_eval
SAMPLE_MEAN = FLAGS.sample_mean

NUM_UNITS = FLAGS.num_units
CELL_TYPE = FLAGS.cell_type
NUM_CELLS = FLAGS.num_cells

VIZ_CURVES = FLAGS.viz

DATA_ROOT = FLAGS.data_root
EVAL_DATA_ROOT = FLAGS.alt_eval_data_root

MODEL_PATH_TEST = FLAGS.model_path_test

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
POINTNET_WEIGHTS = 'pretrained/pointnet_cls_basic_model.ckpt'
POINTNET_PATH = 'models/pointnet_cls_basic.py'

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_SUB_DIR = FLAGS.log_sub_dir
if LOG_SUB_DIR is not None:
    LOG_DIR = os.path.join(LOG_DIR, LOG_SUB_DIR)
    if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
else:
    LOG_DIR = LOG_DIR + '/log_' + str(int(time.time()))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (POINTNET_PATH, LOG_DIR)) # bkp of model def
os.system('cp step_train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

PRED_OUT_PATH = os.path.join(LOG_DIR, 'pred_out')

# dataset of shape simulations
DATASET = step_dataset.BtStepDataset(root=DATA_ROOT, batch_size=BATCH_SIZE, num_pts=1024, num_steps=NUM_STEPS, filter=True, validation_split=0.0)
if EVAL_DATA_ROOT != None:
    DATASET = step_dataset.BtStepDataset(root=EVAL_DATA_ROOT, batch_size=BATCH_SIZE, num_pts=1024, num_steps=NUM_STEPS, filter=True, norm_info=DATASET.get_normalization_info(), validation_split=0.0)

def create_json_vec(vec):
        ''' creates a json dict vec from a np array of size 3 or 4'''
        json_dict = {}
        if len(vec) == 3:
            json_dict = {'x' : float(vec[0]), 'y' : float(vec[1]), 'z' : float(vec[2])}
        elif len(vec) == 4:
            json_dict = {'x' : float(vec[0]), 'y' : float(vec[1]), 'z' : float(vec[2]), 'w' : float(vec[3])}
        return json_dict

def create_json_vec_list(vec_list):
    return [create_json_vec(vec) for vec in vec_list]

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):            
            pointcloud_pl, vel_pl, ang_pl, pos_pl, roty_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_STEPS)            
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Get model and loss 
            if not SINGLE_STEP_EVAL:
                gaussian_params, samples = MODEL.get_test_model(pointcloud_pl, vel_pl[:,:(NUM_STEPS-1),:], ang_pl[:,:(NUM_STEPS-1),:], pos_pl[:,:(NUM_STEPS-1),:], roty_pl[:,:(NUM_STEPS-1),:], CELL_TYPE, NUM_CELLS, NUM_UNITS, NUM_STEPS-1, SAMPLE_MEAN)
                vel_err, ang_err, pos_err, rot_err, pos_step_err, rot_step_err = MODEL.get_test_loss(vel_pl, ang_pl, pos_pl, roty_pl, samples)
            
                vel_err *= DATASET.get_init_vel_normalization()
                ang_err *= DATASET.get_init_angvel_normalization()
            
                pos_err *= DATASET.get_com_normalization()
                rot_err *= DATASET.get_rot_normalization()

                pos_step_err = [x*DATASET.get_com_normalization() for x in pos_step_err]
                rot_step_err = [x*DATASET.get_rot_normalization() for x in rot_step_err]

                mean_devs = MODEL.get_test_uncertainty(gaussian_params)
                norm_coeffs = [DATASET.get_init_vel_normalization(), DATASET.get_init_angvel_normalization(), DATASET.get_com_normalization(), DATASET.get_rot_normalization()]
                mean_devs = [x[0]*x[1] for x in zip(mean_devs, norm_coeffs)]

                loss = tf.constant(0)
            elif SINGLE_STEP_EVAL:
                # Get model and loss 
                gaussian_params, _ = MODEL.get_model(pointcloud_pl, vel_pl[:,:(NUM_STEPS-1),:], ang_pl[:,:(NUM_STEPS-1),:], CELL_TYPE, NUM_CELLS, NUM_UNITS, 1.0, is_training_pl, bn_decay=None)
                loss, nll, errors = MODEL.get_loss(gaussian_params, vel_pl, ang_pl, pos_pl, roty_pl)

                vel_err = errors[0] * DATASET.get_init_vel_normalization()
                ang_err = errors[1] * DATASET.get_init_angvel_normalization()
            
                pos_err = errors[2] * DATASET.get_com_normalization()
                rot_err = errors[3] * DATASET.get_rot_normalization()

                pos_step_err = tf.zeros([NUM_STEPS])
                rot_step_err = tf.zeros([NUM_STEPS])

                samples = tf.constant(0)
                mean_devs = tf.zeros([4])
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

            # vars = tf.contrib.framework.get_variables_to_restore()
            # print(vars)
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)        

        ops = {'pointcloud_pl': pointcloud_pl,
               'vel_pl' : vel_pl,
               'ang_pl' : ang_pl,
               'pos_pl' : pos_pl,
               'roty_pl' : roty_pl,
               'gaussian_pred' : gaussian_params,
               'is_training_pl': is_training_pl,
               'vel_err' : vel_err,
               'ang_err' : ang_err,
               'pos_err' : pos_err,
               'rot_err' : rot_err,
               'pos_step_err' : pos_step_err,
               'rot_step_err' : rot_step_err,
               'mean_devs' : mean_devs,
               'loss' : loss,
               'samples' : samples
               } #,
               #'grad_vars': grads_and_vars}

        saver.restore(sess, MODEL_PATH_TEST)
        log_string("Test model restored.")
        eval_one_epoch(sess, ops, split='test')
        
def eval_one_epoch(sess, ops, split='test'):
    """ ops: dict mapping from string to tf ops """
    log_string("======================================== EVALUATING ON " + split + " DATA ============================================")
    is_training = False

    if OUTPUT_PRED:
        if not os.path.exists(PRED_OUT_PATH): os.mkdir(PRED_OUT_PATH)
        SPLIT_PRED_OUT_PATH = os.path.join(PRED_OUT_PATH, split)
        if not os.path.exists(SPLIT_PRED_OUT_PATH): os.mkdir(SPLIT_PRED_OUT_PATH)

    # Make sure batch data is of same size
    cur_pointcloud_batch = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_com_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 2))
    cur_roty_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 1))
    
    cur_vel_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 2))
    cur_ang_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 1))

    total_loss_sum = 0
    vel_err_sum = 0
    ang_err_sum = 0
    com_err_sum = 0
    roty_err_sum = 0
    pos_step_err_sum = np.zeros((NUM_STEPS))
    rot_step_err_sum = np.zeros((NUM_STEPS))
    mean_devs_sum = np.zeros((4))
    batch_idx = 0.
    
    while DATASET.has_next_batch(split=split):
        # get next data batch
        batch_pc, batch_vel, batch_ang, batch_com, batch_roty = DATASET.next_data_batch(split=split)
        bsize = batch_pc.shape[0]
        cur_pointcloud_batch[0:bsize] = batch_pc
        cur_com_batch[0:bsize,:,:] = batch_com
        cur_roty_batch[0:bsize,:,0] = batch_roty
        cur_vel_batch[0:bsize,:,:] = batch_vel
        cur_ang_batch[0:bsize,:,0] = batch_ang
        
        feed_dict = {ops['pointcloud_pl']: cur_pointcloud_batch,
                     ops['pos_pl'] : cur_com_batch,
                     ops['roty_pl'] : cur_roty_batch,
                     ops['vel_pl'] : cur_vel_batch,
                     ops['ang_pl'] : cur_ang_batch,
                     ops['is_training_pl']: is_training}
        
        loss, samples, vel_err_val, ang_err_val, com_err_val, roty_err_val, pos_step_err_val, rot_step_err_val, mean_devs_val = \
                                 sess.run([ops['loss'], ops['samples'], ops['vel_err'], ops['ang_err'], ops['pos_err'], ops['rot_err'], ops['pos_step_err'], ops['rot_step_err'], ops['mean_devs']], feed_dict=feed_dict)
        
        # print(samples)
        # print(com_err_val)
#         test_writer.add_summary(summary, step)
        com_err_sum += com_err_val
        roty_err_sum += roty_err_val
        vel_err_sum += vel_err_val
        ang_err_sum += ang_err_val
        total_loss_sum += loss

        pos_step_err_sum += np.array(pos_step_err_val)
        rot_step_err_sum += np.array(rot_step_err_val)

        mean_devs_sum += np.array(mean_devs_val)

        # print(batch_idx)

        if OUTPUT_PRED:
            file_out = os.path.join(SPLIT_PRED_OUT_PATH, 'eval_sim_' + str(int(batch_idx)) + '.json')
            with open(file_out, 'w') as f:
                batch_int = int(batch_idx)
                shape_out = DATASET.get_shape_name(batch_int, split)
                cur_scale = DATASET.get_scale_idx(batch_int, split)
                scale_out = {'x' : cur_scale[0], 'y' : cur_scale[1], 'z' : cur_scale[2]}
                # gt pos
                cur_com_y = DATASET.get_full_step_pos(batch_int, split)[0][1]
                cur_com = batch_com[0] * DATASET.get_com_normalization()
                cur_com = np.array([[x[0], cur_com_y * DATASET.get_com_normalization(), x[1]] for x in cur_com])
                # print(cur_com)
                gt_pos_out = create_json_vec_list(cur_com)
                # sampled pos
                sampled_pos = np.array([[x[0, 3]*DATASET.get_com_normalization(), cur_com[0][1], x[0, 4]*DATASET.get_com_normalization()] for x in samples])
                # print(sampled_pos)
                samp_pos_out = create_json_vec_list(sampled_pos)
                # gt rot
                cur_rot = batch_roty[0] * DATASET.get_rot_normalization()
                cur_rot = np.array([[0.0, x, 0.0] for x in cur_rot])
                gt_rot_out = create_json_vec_list(cur_rot)
                # sampled rot
                sampled_rot = np.array([[0.0, x[0, 5]*DATASET.get_rot_normalization(), 0.0] for x in samples])
                samp_rot_out = create_json_vec_list(sampled_rot)
                # error
                pos_err_out = float(com_err_val)
                rot_err_out = float(roty_err_val)
                # create dict and output
                json_dict = {'shape' : shape_out, \
                             'scale' : scale_out, \
                             'gt_pos' : gt_pos_out, \
                             'gt_rot' : gt_rot_out, \
                             'samp_pos' : samp_pos_out, \
                             'samp_rot' : samp_rot_out, \
                             'pos_err' : pos_err_out, \
                             'rot_err' : rot_err_out}
                # print(json_dict)
                json_string = json.dumps(json_dict, sort_keys=True, separators=(',', ':'))
                f.write(json_string)

        batch_idx += 1.

    total_loss_sum /= batch_idx
    vel_err_sum /= batch_idx
    ang_err_sum /= batch_idx
    com_err_sum /= batch_idx
    roty_err_sum /= batch_idx
    pos_step_err_sum /= batch_idx
    rot_step_err_sum /= batch_idx
    mean_devs_sum /= batch_idx

    log_string(' ---- after batch: %03d ----' % (batch_idx+1))
    log_string('mean loss: %f' % (total_loss_sum))
    log_string('mean vel err: %f' % (vel_err_sum))
    log_string('mean angvel err: %f' % (ang_err_sum))
    log_string('mean com err: %f' % (com_err_sum))
    log_string('mean roty err: %f' % (roty_err_sum))
    log_string('mean deviations [vel, angvel, pos, rot]: ' + str(mean_devs_sum))
    # print(pos_step_err_sum)
    # print(rot_step_err_sum)
    log_string('mean step pos err: ' + str(pos_step_err_sum))
    log_string('mean step rot err: ' + str(rot_step_err_sum))

    DATASET.reset_split(split=split)

    return (total_loss_sum, vel_err_sum, ang_err_sum, com_err_sum, roty_err_sum)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
