'''
    Single-GPU training.
'''
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=1, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='step_rnn', help='Model name [default: step_rnn]')
parser.add_argument('--data_root', default='./data/step_cube_no_scale_totrot_5k/', help='Dataset root [default: ./data/step_cube_no_scale_totrot_5k/]')
parser.add_argument('--eval_split', type=float, default=0.2, help='Percentage of training set used for early stopping validation [default: 0.2]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_point', type=int, default=1024, help='Number of Points in Point Cloud [default: 1024]')
parser.add_argument('--num_steps', type=int, default=15, help='Number of steps to train on [default: 5]')
parser.add_argument('--max_epoch', type=int, default=301, help='Epoch to run [default: 201]')
parser.add_argument('--validate_every', type=int, default=5, help='Number of epochs between each early stopping validation epoch [default: 5]')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 128]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.005]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 500000]')
parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate for lr decay [default: 0.5]')

# network properties
parser.add_argument('--num_units', type=int, default=128, help='Number of units to use in each RNN unit [default: 128]')
parser.add_argument('--cell_type', default='lstm', help='Cell type to use (rnn, gru, or lstm) [default: lstm]')
parser.add_argument('--num_cells', type=int, default=3, help='Number of cells to stack to form the RNN module [default: 3]')
parser.add_argument('--dropout_keep_prob', type=float, default=1.0, help='Probability of keeping nodes between RNN cells [default:1.0 (no dropout)]')

parser.add_argument('--no_viz', dest='viz', action='store_false')
parser.set_defaults(viz=True)

FLAGS = parser.parse_args()

import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

import math
from datetime import datetime
import time
import random
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
EVAL_SPLIT = FLAGS.eval_split
NUM_POINT = FLAGS.num_point
NUM_STEPS = FLAGS.num_steps+1 # want to train on num_steps, but we need num_steps+1 data to do this
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
EVAL_RATE = FLAGS.validate_every

VIZ_CURVES = FLAGS.viz

DATA_ROOT = FLAGS.data_root

# net properties
NUM_UNITS = FLAGS.num_units
CELL_TYPE = FLAGS.cell_type
NUM_CELLS = FLAGS.num_cells
DROP_PROB = FLAGS.dropout_keep_prob

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(ROOT_DIR, 'models', FLAGS.model+'.py')
POINTNET_WEIGHTS = 'pretrained/pointnet_cls_basic_model.ckpt'
POINTNET_PATH = 'models/pointnet_cls_basic.py'

LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_DIR = LOG_DIR + '/log_' + str(int(time.time()))
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (POINTNET_PATH, LOG_DIR)) # bkp of model def
os.system('cp step_train.py %s' % (LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_train.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99

# dataset of shape simulations
DATASET = step_dataset.BtStepDataset(root=DATA_ROOT, batch_size=BATCH_SIZE, num_pts=1024, num_steps=NUM_STEPS, filter=True, validation_split=EVAL_SPLIT)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learning_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate        
#     return BASE_LEARNING_RATE

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def train():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):            
            pointcloud_pl, vel_pl, ang_pl, pos_pl, roty_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT, NUM_STEPS)            
            is_training_pl = tf.placeholder(tf.bool, shape=())
            
            batch = tf.get_variable('batch', [],
                initializer=tf.constant_initializer(0), trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss 
            gaussian_params, _ = MODEL.get_model(pointcloud_pl, vel_pl[:,:(NUM_STEPS-1),:], ang_pl[:,:(NUM_STEPS-1),:], CELL_TYPE, NUM_CELLS, NUM_UNITS, DROP_PROB, is_training_pl, bn_decay=bn_decay)
            loss, nll, errors = MODEL.get_loss(gaussian_params, vel_pl, ang_pl, pos_pl, roty_pl)

            vel_err = errors[0] * DATASET.get_init_vel_normalization()
            ang_err = errors[1] * DATASET.get_init_angvel_normalization()
            
            pos_err = errors[2] * DATASET.get_com_normalization()
            rot_err = errors[3] * DATASET.get_rot_normalization()

            tf.summary.scalar('loss', loss)       
            tf.summary.scalar('vel error', vel_err)
            tf.summary.scalar('angvel error', ang_err)
            tf.summary.scalar('pos error', pos_err)
            tf.summary.scalar('rot error', rot_err)
     
            print "--- Get training operator ---"
            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)
            
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            # pointnet saver to load pretrained weights
            ptnet_variables = tf.contrib.framework.get_variables_to_restore()
            ptnet_variables = [v for v in ptnet_variables if v.name.split('/')[0] in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'maxpool', 'fc1', 'fc2', 'fc3', 'dp1']]
#             print(ptnet_variables)
            ptnet_saver = tf.train.Saver(ptnet_variables)

            # count number of params
            # print(tf.trainable_variables())
            total_parameters = 0
            for variable in tf.trainable_variables():
                # shape is an array of tf.Dimension
                shape = variable.get_shape()
                # print(shape)
                # print(len(shape))
                variable_parameters = 1
                for dim in shape:
                    # print(dim)
                    variable_parameters *= dim.value
                # print(variable_parameters)
                total_parameters += variable_parameters
            print('TOTAL PARAMS: ' + str(total_parameters))
        
        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'), sess.graph)
        eval_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'eval'), sess.graph)
        test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'), sess.graph)
             
        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init)
        
        ptnet_saver.restore(sess, POINTNET_WEIGHTS)
        log_string("PointNet model restored.")
        

        ops = {'pointcloud_pl': pointcloud_pl,
               'vel_pl' : vel_pl,
               'ang_pl' : ang_pl,
               'pos_pl' : pos_pl,
               'roty_pl' : roty_pl,
               'gaussian_pred' : gaussian_params,
               'nll' : nll,
               'is_training_pl': is_training_pl,
               'vel_err' : vel_err,
               'ang_err' : ang_err,
               'pos_err' : pos_err,
               'rot_err' : rot_err,
               'total_loss': loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch} #,
               #'grad_vars': grads_and_vars}

        # keep stats for plotting training curves
        train_total_loss = []
        train_vel_err = []
        train_ang_err_sum = [] 
        train_com_err_sum = [] 
        train_roty_err_sum = [] 
        # validation curves
        eval_total_loss = []
        eval_vel_err = []
        eval_ang_err_sum = [] 
        eval_com_err_sum = [] 
        eval_roty_err_sum = [] 
        # test curves
        test_total_loss = []
        test_vel_err = []
        test_ang_err_sum = [] 
        test_com_err_sum = [] 
        test_roty_err_sum = [] 

        min_eval_loss = float('inf')
        min_test_loss = float('inf')
        global EPOCH_CNT
        for epoch in range(MAX_EPOCH):
            log_string('**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
             
            train_batch_stats = train_one_epoch(sess, ops, train_writer)
            train_total_loss.append(train_batch_stats[0])
            train_vel_err.append(train_batch_stats[1])
            train_ang_err_sum.append(train_batch_stats[2])
            train_com_err_sum.append(train_batch_stats[3])
            train_roty_err_sum.append(train_batch_stats[4])

            if epoch % EVAL_RATE == 0:
                eval_batch_stats = eval_one_epoch(sess, ops, eval_writer, split='eval')
                cur_eval_loss = eval_batch_stats[0]
                eval_total_loss.append(eval_batch_stats[0])
                eval_vel_err.append(eval_batch_stats[1])
                eval_ang_err_sum.append(eval_batch_stats[2])
                eval_com_err_sum.append(eval_batch_stats[3])
                eval_roty_err_sum.append(eval_batch_stats[4])

                test_batch_stats = eval_one_epoch(sess, ops, test_writer, split='test')
                cur_test_loss = test_batch_stats[0]
                test_total_loss.append(test_batch_stats[0])
                test_vel_err.append(test_batch_stats[1])
                test_ang_err_sum.append(test_batch_stats[2])
                test_com_err_sum.append(test_batch_stats[3])
                test_roty_err_sum.append(test_batch_stats[4])

                if cur_eval_loss < min_eval_loss or cur_eval_loss < 1e-8:
                    min_eval_loss = cur_eval_loss
                    save_path = saver.save(sess, os.path.join(LOG_DIR, "step_rnn_best_model.ckpt"))
                    log_string("BEST EVAL Model saved in file: %s" % save_path)
                if cur_test_loss < min_test_loss:
                    min_test_loss = cur_test_loss
                    log_string("BEST MODEL SO FAR FOR TEST DATA")

            if epoch % 10 == 0:
                save_path = saver.save(sess, os.path.join(LOG_DIR, "step_rnn_model.ckpt"))
                log_string("Model saved in file: %s" % save_path)

            EPOCH_CNT += 1
        
        # plot curves
        if VIZ_CURVES:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            plot_path = os.path.join(LOG_DIR, 'plots')
            if not os.path.exists(plot_path): os.mkdir(plot_path)

            # total loss
            plt.figure()
            plt.plot(np.arange(0, MAX_EPOCH), np.array(train_total_loss), 'r-', label='Train')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), np.array(test_total_loss), 'g-', label='Test')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), np.array(eval_total_loss), 'b-', label='Eval')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title(DATASET.root + ' Total Loss')
            plt.savefig(os.path.join(plot_path, 'total_loss.png'))

            # linear velocity
            train_vel_err_arr = np.array(train_vel_err)
            test_vel_err_arr = np.array(test_vel_err)
            eval_vel_err_arr = np.array(eval_vel_err)

            plt.figure()
            plt.plot(np.arange(0, MAX_EPOCH), train_vel_err_arr, 'r-', label='Train')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), test_vel_err_arr, 'g-', label='Test')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), eval_vel_err_arr, 'b-', label='Eval')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('Vel Err (m/s)')
            plt.title(DATASET.root + ' Vel Err')
            plt.savefig(os.path.join(plot_path, 'vel_err.png'))

            # angular velocity
            plt.figure()
            plt.plot(np.arange(0, MAX_EPOCH), np.array(train_ang_err_sum), 'r-', label='Train')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), np.array(test_ang_err_sum), 'g-', label='Test')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), np.array(eval_ang_err_sum), 'b-', label='Eval')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('AngVel Err (rad/s)')
            plt.title(DATASET.root + ' AngVel Err')
            plt.savefig(os.path.join(plot_path, 'angvel_err.png'))

            # COM
            train_com_err_arr = np.array(train_com_err_sum)
            test_com_err_arr = np.array(test_com_err_sum)
            eval_com_err_arr = np.array(eval_com_err_sum)

            plt.figure()
            plt.plot(np.arange(0, MAX_EPOCH), train_com_err_arr, 'r-', label='Train')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), test_com_err_arr, 'g-', label='Test')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), eval_com_err_arr, 'b-', label='Eval')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('COM Err (m)')
            plt.title(DATASET.root + ' COM Err')
            plt.savefig(os.path.join(plot_path, 'com_err.png'))

            # Rotation
            plt.figure()
            plt.plot(np.arange(0, MAX_EPOCH), np.array(train_roty_err_sum), 'r-', label='Train')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), np.array(test_roty_err_sum), 'g-', label='Test')
            plt.plot(np.arange(0, MAX_EPOCH, EVAL_RATE), np.array(eval_roty_err_sum), 'b-', label='Eval')
            plt.legend()
            plt.xlabel('Epoch')
            plt.ylabel('ROT Err (degrees)')
            plt.title(DATASET.root + ' ROT Err')
            plt.savefig(os.path.join(plot_path, 'rot_err.png'))

def train_one_epoch(sess, ops, train_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = True
    
    log_string(str(datetime.now()))

    # Make sure batch data is of same size
    cur_pointcloud_batch = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    cur_com_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 2))
    cur_roty_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 1))
    
    cur_vel_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 2))
    cur_ang_batch = np.zeros((BATCH_SIZE, NUM_STEPS, 1))

    total_loss_sum = 0
    com_err_sum = 0
    roty_err_sum = 0
    vel_err_sum = 0
    ang_err_sum = 0
    batch_idx = 0
    while DATASET.has_next_train_batch():
        # get next data batch
        batch_pc, batch_vel, batch_ang, batch_com, batch_roty = DATASET.next_train_batch()
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
        
        summary, step, _, total_loss_val, nll, vel_err_val, ang_err_val, com_err_val, roty_err_val, gaussian_pred = \
                                 sess.run([ops['merged'], ops['step'], ops['train_op'], ops['total_loss'], ops['nll'], ops['vel_err'], ops['ang_err'], ops['pos_err'], ops['rot_err'], ops['gaussian_pred']], feed_dict=feed_dict)

        # print(gaussian_pred)
        # print(nll)
        # print('loss: %f' % (total_loss_val))
        # print('pos: %f' % (com_err_val))
        # print('rot: %f' % (roty_err_val))
#         print(com_pred_val)
#         print(rot_pred_val)
        
#         train_writer.add_summary(summary, step)
        total_loss_sum += total_loss_val
        com_err_sum += com_err_val
        roty_err_sum += roty_err_val
        vel_err_sum += vel_err_val
        ang_err_sum += ang_err_val

        batch_idx += 1


    total_loss_sum /= batch_idx
    vel_err_sum /= batch_idx
    ang_err_sum /= batch_idx
    com_err_sum /= batch_idx
    roty_err_sum /= batch_idx

    log_string(' ---- after batch: %03d ----' % (batch_idx+1))
    log_string('mean loss: %f' % (total_loss_sum))
    log_string('mean vel err: %f' % (vel_err_sum))
    log_string('mean angvel err: %f' % (ang_err_sum))
    log_string('mean com err: %f' % (com_err_sum))
    log_string('mean roty err: %f' % (roty_err_sum))

    train_writer.add_summary(summary, step)
    
    DATASET.reset_train()

    return (total_loss_sum, vel_err_sum, ang_err_sum, com_err_sum, roty_err_sum)
        
def eval_one_epoch(sess, ops, test_writer, split='test'):
    """ ops: dict mapping from string to tf ops """
    global EPOCH_CNT
    is_training = False

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
    batch_idx = 0.

    summary_list = []
    step_list = []
    
    log_string(str(datetime.now()))
    log_string('---- EPOCH %03d VALIDATION ----'%(EPOCH_CNT))

    if not DATASET.has_next_batch(split=split):
        # validation split is empty, just return 0
        return (total_loss_sum, vel_err_sum, ang_err_sum, com_err_sum, roty_err_sum)
    
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
        
        summary, step, total_loss_val, vel_err_val, ang_err_val, com_err_val, roty_err_val = \
                                 sess.run([ops['merged'], ops['step'], ops['total_loss'], ops['vel_err'], ops['ang_err'], ops['pos_err'], ops['rot_err']], feed_dict=feed_dict)
        
#         test_writer.add_summary(summary, step)
        total_loss_sum += total_loss_val
        com_err_sum += com_err_val
        roty_err_sum += roty_err_val
        vel_err_sum += vel_err_val
        ang_err_sum += ang_err_val

        batch_idx += 1.

        summary_list.append(summary)
        step_list.append(step)
    
    if len(summary_list) > 0:
        out_idx = random.randint(0, len(summary_list) - 1)
        test_writer.add_summary(summary_list[out_idx], step_list[out_idx])

    total_loss_sum /= batch_idx
    vel_err_sum /= batch_idx
    ang_err_sum /= batch_idx
    com_err_sum /= batch_idx
    roty_err_sum /= batch_idx

    log_string(' ---- after batch: %03d ----' % (batch_idx+1))
    log_string('mean loss: %f' % (total_loss_sum))
    log_string('mean vel err: %f' % (vel_err_sum))
    log_string('mean angvel err: %f' % (ang_err_sum))
    log_string('mean com err: %f' % (com_err_sum))
    log_string('mean roty err: %f' % (roty_err_sum))

    DATASET.reset_split(split=split)

    return (total_loss_sum, vel_err_sum, ang_err_sum, com_err_sum, roty_err_sum)

if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    train()
    LOG_FOUT.close()
