import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import tushare as ts

from SQL_Dao import inset_db

# Iinitial Variable
TIME_STEPS = 10
INPUT_SIZE = 12
OUTPUT_SIZE = 1
BATCH_SIZE = 10
CELL_SIZE = 10
LR = 0.0006


def load_data(stock_id, col_begin=0, col_end=14):
    date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    df = ts.get_hist_data(stock_id, start='2015-05-05', end=date)
    df = df.iloc[2:]
    df = df[::-1]
    data = df.iloc[:, col_begin: col_end].values
    return data


# get TrainTest
def get_train_data(stock_id, batch_size, time_step, time_span):
    data = load_data(stock_id)
    batch_index = []
    data_train = data[:-time_span]
    mean = np.mean(data_train, axis=0)
    std = np.std(data_train, axis=0)
    normalized_train_data = (data_train - mean) / std  # normalize train data
    train_x, train_y = [], []  # TrainSet
    for i in range(len(normalized_train_data) - time_step):
        if i % batch_size == 0:
            batch_index.append(i)
        x = normalized_train_data[i: i + time_step, : INPUT_SIZE]
        y = normalized_train_data[i: i + time_step, INPUT_SIZE, np.newaxis]
        train_x.append(x.tolist())
        train_y.append(y.tolist())
    batch_index.append((len(normalized_train_data) - time_step))
    # print('train_x:\n', train_x)
    return batch_index, train_x, train_y


# get TestSet
def get_test_data(stock_id, time_step, time_span):
    data = load_data(stock_id)
    data_test = data[-time_span:]
    mean = np.mean(data_test, axis=0)
    std = np.std(data_test, axis=0)
    normalized_test_data = (data_test - mean) / std  # normalize test data
    size = (len(normalized_test_data) + time_step - 1) // time_step  # the size of sample
    test_x, test_y = [], []  # TestSet
    for i in range(size - 1):
        x = normalized_test_data[i * time_step:(i + 1) * time_step, :INPUT_SIZE]
        y = normalized_test_data[i * time_step:(i + 1) * time_step, INPUT_SIZE]
        test_x.append(x.tolist())
        test_y.extend(y)
    test_x.append((normalized_test_data[(i + 1) * time_step:, :INPUT_SIZE]).tolist())
    test_y.extend((normalized_test_data[(i + 1) * time_step:, INPUT_SIZE]).tolist())
    return mean, std, test_x, test_y


# define Variable

# 权重
with tf.name_scope('Weights'):
    weights = {
        'in': tf.Variable(tf.random_normal([INPUT_SIZE, CELL_SIZE])),
        'out': tf.Variable(tf.random_normal([CELL_SIZE, 1]))
    }
# 偏置
with tf.name_scope('biases'):
    biases = {
        'in': tf.Variable(tf.constant(0.1, shape=[CELL_SIZE, ])),
        'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
    }


# define LSTM
def lstm(object):
    batch_size = tf.shape(object)[0]
    time_step = tf.shape(object)[1]

    # ----------------- input_layer ------------------------
    with tf.variable_scope('in_hidden'):
        # Ws (in_size, cell_size)
        w_in = weights['in']

        # bs (cell_size, )
        b_in = biases['in']

        # shape = (batch * n_steps, cell_size) : to2D as INPUT_LAYER input
        input = tf.reshape(object, [-1, INPUT_SIZE], name='to_2D')

        with tf.name_scope('Wx_plus_b'):
            # l_in_y = (batch * time_steps, cell_size)
            input_rnn = tf.matmul(input, w_in) + b_in

    # reshape input_rnn ==> (batch, time_steps, cell_size) : to3D as LSTM_CELL input
    input_rnn = tf.reshape(input_rnn, [-1, time_step, CELL_SIZE], name='to_3D')

    # ----------------- cell_layer ------------------------

    with tf.variable_scope('LSTM_cell'):
        cell = tf.contrib.rnn.BasicLSTMCell(CELL_SIZE)

        with tf.name_scope('initial_state'):
            init_state = cell.zero_state(batch_size, dtype=tf.float32)

    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    output_rnn, final_states = tf.nn.dynamic_rnn(cell, input_rnn, initial_state=init_state, dtype=tf.float32)

    # ----------------- output_layer ------------------------

    with tf.variable_scope('out_hidden'):
        # Ws (cell_size, output_size)
        w_out = weights['out']

        # bs (output_size)
        b_out = biases['out']

        output = tf.reshape(output_rnn, [-1, CELL_SIZE])

        with tf.name_scope('Wx_plus_b'):
            pred = tf.matmul(output, w_out) + b_out

    return pred, final_states


# Train Model
def train_lstm(stock_id, batch_size=20, time_step=10, time_span=20):
    with tf.name_scope('inputs'):
        X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE], name='x_input')
        Y = tf.placeholder(tf.float32, shape=[None, time_step, OUTPUT_SIZE], name='y_input')

    batch_index, train_x, train_y = get_train_data(stock_id, batch_size, time_step, time_span)
    pred, _ = lstm(X)

    # loss function
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(tf.square(tf.reshape(pred, [-1]) - tf.reshape(Y, [-1])))

    with tf.name_scope('train'):
        train_op = tf.train.AdamOptimizer(LR).minimize(loss)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter("tf_log", sess.graph)
        sess.run(tf.global_variables_initializer())

        for i in range(100):
            for step in range(len(batch_index) - 1):
                feed_dict = {X: train_x[batch_index[step]:batch_index[step + 1]],
                             Y: train_y[batch_index[step]:batch_index[step + 1]]}

                _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)

            print(i, loss_)
            # result = sess.run(merged, feed_dict)
            # writer.add_summary(result, i)

        # save Model Variables
        date = time.strftime('%Y%m%d', time.localtime(time.time()))
        save_file_path = "tf_net/" + stock_id + "-" + date + ".ckpt"
        save_path = saver.save(sess, save_file_path)
        print("Save to path: ", save_path)
        return "success", save_file_path


# Test Model
def prediction(stock_id, save_file_path, time_step=1, time_span=20):
    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])

    mean, std, test_x, test_y = get_test_data(stock_id, time_step, time_span)
    pred, _ = lstm(X)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        # input Model Variables
        saver.restore(sess, save_file_path)

        test_predict = []
        for step in range(20):
            prob = sess.run(pred, feed_dict={X: [test_x[step]]})
            predict = prob.reshape((-1))
            test_predict.extend(predict)

        # prediction of next trading day
        prob = sess.run(pred, feed_dict={X: [test_x[-2]]})
        predict = prob.reshape((-1))
        test_predict.extend(predict)

        test_y = np.array(test_y) * std[INPUT_SIZE] + mean[INPUT_SIZE]
        test_predict = np.array(test_predict) * std[INPUT_SIZE] + mean[INPUT_SIZE]

        # Accumulated Error
        acc = np.average(np.abs(test_predict[:20] - test_y[:len(test_predict) - 1]) / test_y[:len(test_predict) - 1])
        print(acc)

        # # draw Graph
        # plt.figure()
        # plt.plot(list(range(len(test_predict))), test_predict, color='r')
        # plt.plot(list(range(len(test_y))), test_y, color='b')
        # plt.legend(['predict', 'true'])
        # plt.show()
        return test_y, test_predict


# train_lstm(stock_id='600000')
stock_id='600000'
date = time.strftime('%Y%m%d', time.localtime(time.time()))
save_file_path = "tf_net/" + stock_id + "-" + date + ".ckpt"
true_data, prediction_data = prediction(stock_id='600000', save_file_path=save_file_path)
date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
inset_db(stock_id='600000', date=date, true_data=true_data, prediction_data=prediction_data)
