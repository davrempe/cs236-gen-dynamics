import tensorflow as tf
import numpy as np
import math
import sys
import os
import pointnet_cls_basic as pointnet
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util

tfd = tf.distributions

# USE_RNN = False
# USE_GRU = False
# USE_LSTM = False
# USE_MULTI_LSTM = True

def placeholder_inputs(batch_size, num_points, num_steps):
    pointcloud_pl = tf.placeholder(tf.float32, shape=(batch_size, num_points, 3))
    vel_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 2))
    angvel_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 1))
    pos_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 2))
    rot_pl = tf.placeholder(tf.float32, shape=(batch_size, num_steps, 1))

    return pointcloud_pl, vel_pl, angvel_pl, pos_pl, rot_pl 

def get_model(point_cloud, vel, angvel, cell_type, num_cells, hidden_size, dropout_keep_prob, is_training, bn_decay=None):
    # first get shape feature
    pointnet_feat = get_pointnet_model(point_cloud, is_training, bn_decay=bn_decay)  
    # process pointnet 
    pt_vec = tf_util.fully_connected(pointnet_feat, 1024, weight_decay=0.005, bn=True, \
                           is_training=is_training, scope='fwd_fc1', bn_decay=bn_decay)
    pt_vec = tf_util.fully_connected(pt_vec, 512,  weight_decay=0.005, bn=True, \
                           is_training=is_training, scope='fwd_fc2', bn_decay=bn_decay)
    pt_vec = tf_util.fully_connected(pt_vec, 128,  weight_decay=0.005, bn=True, \
                           is_training=is_training, scope='fwd_fc3', bn_decay=bn_decay)
    pt_vec = tf_util.fully_connected(pt_vec, 32,  weight_decay=0.005, bn=True, \
                           is_training=is_training, scope='fwd_fc4', bn_decay=bn_decay)
    shape_feat = tf_util.fully_connected(pt_vec, 9,  weight_decay=0.005, bn=True, \
                           is_training=is_training, scope='fwd_fc5', bn_decay=bn_decay)

    batch_size = shape_feat.get_shape()[0].value
    time_steps = vel.get_shape()[1].value

    # inputs are a 12-vec [vel, angvel, shape_feat]
    step_shape = tf.tile(tf.expand_dims(shape_feat, 1), tf.constant([1, time_steps, 1]))
    inputs = tf.concat([vel, angvel, step_shape], axis=2)

    # get gaussian parameters, need 12 of them for dv, dw, dp, d\theta
    num_params = 12
    W_hy = tf.get_variable('W_hy', shape=(hidden_size, num_params), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    weight_decay = tf.multiply(tf.nn.l2_loss(W_hy), 0.005, name='weight_loss')
    tf.add_to_collection('losses', weight_decay)
    b_hy = tf.get_variable('b_hy', shape=(1, num_params), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)

    if cell_type=='fc':
        # need to do it completely differently
        # num_cells used as number of FC layers, each with hidden_size nodes
        # inputs is B, num_steps, 12
        inputs = tf.reshape(inputs, [batch_size*time_steps, num_params])
        cur_input = inputs
        for j in range(num_cells):
            cell_name = 'cell_fc' + str(j)
            cur_input = tf_util.fully_connected(cur_input, hidden_size, weight_decay=0.005, bn=True, \
                        is_training=is_training, scope=cell_name, activation_fn=tf.nn.tanh, bn_decay=bn_decay)
        # final output
        y = tf.matmul(cur_input, W_hy) + b_hy
        y = tf.reshape(y, [batch_size, time_steps, num_params])

        return y, y

    # then feed to RNN with velocites
    # basic RNN cell
    if cell_type=='rnn':
        rnn_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.BasicRNNCell(hidden_size), output_keep_prob=dropout_keep_prob) for i in range(0, num_cells)]
    if cell_type=='gru':
        rnn_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer()), output_keep_prob=dropout_keep_prob) for i in range(0, num_cells)]
    if cell_type=='lstm':
        rnn_cell = [tf.nn.rnn_cell.DropoutWrapper(tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=tf.contrib.layers.xavier_initializer()), output_keep_prob=dropout_keep_prob) for i in range(0, num_cells)]
    
    if num_cells > 1:
        rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell)
    else:
        rnn_cell = rnn_cell[0]

    
    init_state = rnn_cell.zero_state(batch_size, dtype=tf.float32)

    # feed through RNN
    # outputs are [batch, time_steps, hidden_size]
    outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs, initial_state=init_state, dtype=tf.float32)
    
    y = tf.matmul(tf.reshape(outputs, [batch_size*time_steps, hidden_size]), W_hy) + b_hy
    y = tf.reshape(y, [batch_size, time_steps, num_params])

    return y, state

def get_test_model(point_cloud, vel, angvel, pos, rot, cell_type, num_cells, hidden_size, num_steps, sample_mean):
    ''' Test version of the model to be sampled from. Given the object point cloud
        and initial vel/angvel, will roll out samples from model for num_steps. '''
    # first get shape feature
    pointnet_feat = get_pointnet_model(point_cloud, tf.constant(False), bn_decay=None)  
    # process pointnet 
    pt_vec = tf_util.fully_connected(pointnet_feat, 1024, weight_decay=0.005, bn=True, \
                           is_training=tf.constant(False), scope='fwd_fc1', bn_decay=None)
    pt_vec = tf_util.fully_connected(pt_vec, 512,  weight_decay=0.005, bn=True, \
                           is_training=tf.constant(False), scope='fwd_fc2', bn_decay=None)
    pt_vec = tf_util.fully_connected(pt_vec, 128,  weight_decay=0.005, bn=True, \
                           is_training=tf.constant(False), scope='fwd_fc3', bn_decay=None)
    pt_vec = tf_util.fully_connected(pt_vec, 32,  weight_decay=0.005, bn=True, \
                           is_training=tf.constant(False), scope='fwd_fc4', bn_decay=None)
    shape_feat = tf_util.fully_connected(pt_vec, 9,  weight_decay=0.005, bn=True, \
                           is_training=tf.constant(False), scope='fwd_fc5', bn_decay=None)

    if cell_type=='rnn':
        rnn_cell = [tf.nn.rnn_cell.BasicRNNCell(hidden_size) for i in range(0, num_cells)]
    if cell_type=='gru':
        rnn_cell = [tf.nn.rnn_cell.GRUCell(hidden_size, kernel_initializer=tf.contrib.layers.xavier_initializer()) for i in range(0, num_cells)]
    if cell_type=='lstm':
        rnn_cell = [tf.nn.rnn_cell.LSTMCell(hidden_size, initializer=tf.contrib.layers.xavier_initializer()) for i in range(0, num_cells)]
    
    # TODO make sure all the fc stuff here works
    if cell_type!='fc':
        if num_cells > 1:
            rnn_cell = tf.nn.rnn_cell.MultiRNNCell(rnn_cell)
        else:
            rnn_cell = rnn_cell[0]

    # rnn_cell.scope_name = 'rnn'
    W_hy = tf.get_variable('W_hy', shape=(hidden_size, 12), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    b_hy = tf.get_variable('b_hy', shape=(1, 12), initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
    # rnn_cell = tf.contrib.cudnn_rnn.CudnnRNNTanh(1, hidden_size)
    batch_size = shape_feat.get_shape()[0].value
    time_steps = vel.get_shape()[1].value
    if cell_type!='fc':
        state = rnn_cell.zero_state(batch_size, dtype=tf.float32)
    # inputs are a 12-vec [vel, angvel, shape_feat]
    step_shape = tf.tile(tf.expand_dims(shape_feat, 1), tf.constant([1, time_steps, 1]))
    all_inputs = tf.concat([vel, angvel, step_shape], axis=2)
    all_states = tf.concat([vel, angvel, pos, rot], axis=2)

    # for sampling only feed in first step
    init_input = all_inputs[:,0,:]
    cur_input = init_input #tf.reshape(init_input, [batch_size, 1, -1])
    cur_state = all_states[:,0,:]
    sampled_states = [cur_state]
    # print(cur_input)
    gaussian_params = []
    for i in range(num_steps):
        if cell_type=='fc':
            mid_feat = cur_input
            for j in range(num_cells):
                cell_name = 'cell_fc' + str(j)
                mid_feat = tf_util.fully_connected(mid_feat, hidden_size, weight_decay=0.005, bn=True, \
                            is_training=tf.constant(False), scope=cell_name, activation_fn=tf.nn.tanh, bn_decay=None)
            output = mid_feat
        else:
            with tf.variable_scope("rnn"):
                output, state = rnn_cell(cur_input, state)
        # print(output)
        # get gaussian params from output
        cur_params = tf.matmul(output, W_hy) + b_hy
        gaussian_params.append(cur_params)
        # print(y)
        # sample
        N = tfd.Normal(loc=cur_params[:,0:6], scale=tf.exp(cur_params[:,6:]))
        if sample_mean:
            sample = N.mean()
        else:
            sample = N.sample()
        # calculate new state, shape does not change, just velocity
        cur_input += tf.concat([sample[:,:3], tf.zeros([batch_size, 9])], axis=1)
        cur_state += sample
        sampled_states.append(cur_state)

    # print(sampled_states)
    return gaussian_params, sampled_states

def get_pointnet_model(point_cloud, is_training, bn_decay=None):
    '''
    PointNet classifier model. Returns only global feature.
    '''
    _, _, global_feat = pointnet.get_model(point_cloud, is_training, bn_decay=bn_decay)
    return global_feat

def get_loss(gaussian_params, vel, angvel, pos, rot):
    num_steps = vel.get_shape()[1].value
    # calculate change in linear vel for gt
    vel_t = vel[:,0:(num_steps-1)]
    vel_tp1 = vel[:, 1:]
    vel_diff = vel_tp1 - vel_t
    # calclate change in ang vel for gt
    angvel_t = angvel[:, 0:(num_steps-1)]
    angvel_tp1 = angvel[:, 1:]
    angvel_diff = angvel_tp1 - angvel_t
    # calculate change in pos for gt
    pos_t = pos[:, 0:(num_steps-1)]
    pos_tp1 = pos[:, 1:]
    pos_diff = pos_tp1 - pos_t
    # calculate change in rot for gt
    rot_t = rot[:, 0:(num_steps-1)]
    rot_tp1 = rot[:, 1:]
    rot_diff = rot_tp1 - rot_t
    # stack together
    gt_data = tf.concat([vel_diff, angvel_diff, pos_diff, rot_diff], axis=2)

    # build gaussians
    num_distr = gaussian_params.get_shape()[2].value / 2
    N = tfd.Normal(loc=gaussian_params[:,:,0:num_distr], scale=tf.exp(gaussian_params[:,:,num_distr:]))
    # evaluate likelihood of gt data
    nll = -N.log_prob(gt_data)
    loss = tf.reduce_mean(tf.reduce_sum(tf.reduce_sum(nll, axis=2), axis=1))
    # nll = N.mean()

    # get absolute error if we choose the mean from each
    errors = tf.abs(gt_data - N.mean())
    vel_err = tf.reduce_mean(errors[:,:,0:2])
    angvel_err = tf.reduce_mean(errors[:,:,2])
    pos_err = tf.reduce_mean(errors[:,:,3:5])
    rot_err = tf.reduce_mean(errors[:,:,5])

    return loss, nll, (vel_err, angvel_err, pos_err, rot_err)

def get_test_loss(gt_vel, gt_angvel, gt_pos, gt_rot, sampled_states):
    vel_err = tf.constant(0.0)
    angvel_err = tf.constant(0.0)
    pos_err = tf.constant(0.0)
    rot_err = tf.constant(0.0)
    
    pos_step_err = []
    rot_step_err = []
    for i, state in enumerate(sampled_states):
        vel_err += tf.reduce_mean(tf.norm(state[:,0:2] - gt_vel[:,i,:], axis=1))
        angvel_err += tf.reduce_mean(tf.abs(state[:,2] - gt_angvel[:,i,:]))
        cur_pos_err = tf.reduce_mean(tf.norm(state[:,3:5] - gt_pos[:,i,:], axis=1))
        pos_err += cur_pos_err
        pos_step_err.append(cur_pos_err)
        cur_rot_err = tf.reduce_mean(tf.abs(state[:,5] - gt_rot[:,i,:]))
        rot_err += cur_rot_err
        rot_step_err.append(cur_rot_err)

    vel_err /= len(sampled_states)
    angvel_err /= len(sampled_states)
    pos_err /= len(sampled_states)
    rot_err /= len(sampled_states)

    return vel_err, angvel_err, pos_err, rot_err, pos_step_err, rot_step_err

def get_test_uncertainty(gaussian_params):
    ''' calculates the mean variance of the predicted gaussian parameters '''
    vel_dev = tf.constant(0.0)
    angvel_dev = tf.constant(0.0)
    pos_dev = tf.constant(0.0)
    rot_dev = tf.constant(0.0)

    for i, params in enumerate(gaussian_params):
        cur_dev = tf.exp(params[:,6:])
        vel_dev += tf.reduce_mean(cur_dev[:,0:2])
        angvel_dev += tf.reduce_mean(cur_dev[:,2])
        pos_dev += tf.reduce_mean(cur_dev[:,3:5])
        rot_dev += tf.reduce_mean(cur_dev[:,5])
    
    vel_dev /= len(gaussian_params)
    angvel_dev /= len(gaussian_params)
    pos_dev /= len(gaussian_params)
    rot_dev /= len(gaussian_params)

    return vel_dev, angvel_dev, pos_dev, rot_dev

if __name__=='__main__':
    with tf.Graph().as_default():        
        point_cloud = tf.zeros((32, 1024, 3))
        vel = tf.zeros((32, 6, 2))
        angvel = tf.zeros((32, 6, 1))
        pos = tf.zeros((32, 6, 2))
        rot = tf.zeros((32, 6, 1))

        # outputs, state = get_model(point_cloud, vel[:,:6,:], angvel[:,:6,:], 128, tf.constant(True))

        # gaussian_params = tf.ones((32, 5, 12))
        # loss = get_loss(gaussian_params, vel, angvel, pos, rot)

        gaussian_params, sampled_states = get_test_model(point_cloud, vel[:,:6,:], angvel[:,:6,:], pos[:,:6,:], rot[:,:6,:], 128, 5)
        err = get_test_loss(vel, angvel, pos, rot, sampled_states)

        # print(outputs)
        # print(loss)
