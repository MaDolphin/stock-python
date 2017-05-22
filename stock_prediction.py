import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import time
import tushare as ts

rnn_def = ""

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
weights = {
    'in': tf.Variable(tf.random_normal([INPUT_SIZE, CELL_SIZE])),
    'out': tf.Variable(tf.random_normal([CELL_SIZE, 1]))
}
# 偏置
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[CELL_SIZE, ])),
    'out': tf.Variable(tf.constant(0.1, shape=[1, ]))
}


# define LSTM
def lstm(object):
    global rnn_def
    batch_size = tf.shape(object)[0]
    time_step = tf.shape(object)[1]

    # ----------------- input_layer ------------------------

    # Ws (in_size, cell_size)
    w_in = weights['in']

    # bs (cell_size, )
    b_in = biases['in']

    # shape = (batch * n_steps, cell_size) : to2D as INPUT_LAYER input
    input = tf.reshape(object, [-1, INPUT_SIZE])

    # l_in_y = (batch * time_steps, cell_size)
    input_rnn = tf.matmul(input, w_in) + b_in

    # reshape input_rnn ==> (batch, time_steps, cell_size) : to3D as LSTM_CELL input
    input_rnn = tf.reshape(input_rnn, [-1, time_step, CELL_SIZE])

    # ----------------- cell_layer ------------------------
    with tf.variable_scope('prediction_LSTM'):
        prediction_cell = tf.contrib.rnn.BasicLSTMCell(CELL_SIZE)

    init_state = prediction_cell.zero_state(batch_size, dtype=tf.float32)

    # output_rnn是记录lstm每个输出节点的结果，final_states是最后一个cell的结果
    with tf.variable_scope(rnn_def, reuse=True):
        output_rnn, final_states = tf.nn.dynamic_rnn(prediction_cell, input_rnn, initial_state=init_state,
                                                     dtype=tf.float32)

    # ----------------- output_layer ------------------------

    # Ws (cell_size, output_size)
    w_out = weights['out']

    # bs (output_size)
    b_out = biases['out']

    output = tf.reshape(output_rnn, [-1, CELL_SIZE])

    pred = tf.matmul(output, w_out) + b_out

    return pred, final_states


# Test Model
def prediction(stock_id, save_file_path, time_step=1, time_span=20, rnn_name=""):
    global rnn_def
    rnn_def = rnn_name

    X = tf.placeholder(tf.float32, shape=[None, time_step, INPUT_SIZE])

    mean, std, test_x, test_y = get_test_data(stock_id, time_step, time_span)
    pred, _ = lstm(X)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        tf.global_variables_initializer()
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
        sess.close()
        return test_y, test_predict

# train_lstm(stock_id='601766')
# stock_id='600000'
# date = time.strftime('%Y%m%d', time.localtime(time.time()))
# save_file_path = "tf_net/" + stock_id + "-" + date + ".ckpt"
# true_data, prediction_data = prediction(stock_id='600000', save_file_path=save_file_path)
# date = time.strftime('%Y-%m-%d', time.localtime(time.time()))
# inset_db(stock_id='600000', date=date, true_data=true_data, prediction_data=prediction_data)
# execute_lstm(stock_id='601766')
