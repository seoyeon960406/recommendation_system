import numpy as np
import tensorflow as tf

batch_size = 256
vector_size = 300
max_seq_length = 140
learning_rate = 0.001
lstm_units = 128
num_class = 2 # 분류할 클래스 수(긍정, 부정)
training_epochs = 100

X = tf.placeholder(tf.float32,[None, max_seq_length, vector_size])
Y = tf.placeholder(tf.float32,[None, num_class])

keep_prob = tf.placeholder(tf.float32)
seq_len = tf.placeholder(tf.int32, shape=[None])

# lstm cell 2개를 생성합니다. 각 셀은 Dropout을 시켜줘 Overfitting을 방지합니다.
lstm_fw_cell = tf.nn.rnn_cell.LSTMCell(num_units = lstm_units, state_is_tuple = True)
lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_fw_cell, output_keep_prob=keep_prob)
lstm_bw_cell = tf.nn.rnn_cell.LSTMCell(num_units = lstm_units, state_is_tuple = True)
lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(lstm_bw_cell, output_keep_prob=keep_prob)

W = tf.get_variable(name="W", shape=[2 * lstm_units, num_class],
                                dtype=tf.float32, initializer = tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name="b", shape=[num_class], dtype=tf.float32,
                                initializer=tf.zeros_initializer())

output, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                 dtype=tf.float32, inputs = X, sequence_length=seq_len)

## concat fw, bw final states
outputs = tf.concat([output[0], output[1]], 2)
final_hidden_state = tf.concat([states[1][0], states[1][1]], 1)
final_hidden_state = tf.expand_dims(final_hidden_state, 2)

# attn_weights : [batch_size, n_step]
attn_weights = tf.squeeze(tf.matmul(outputs, final_hidden_state), 2)
soft_attn_weights = tf.nn.softmax(attn_weights, 1)
tf.transpose(outputs, [0, 2, 1])

context = tf.matmul(tf.transpose(outputs, [0, 2, 1]), tf.expand_dims(soft_attn_weights, 2)) # context : [batch_size, n_hidden * num_directions(=2), 1]
context = tf.squeeze(context, 2) # [batch_size, n_hidden * num_directions(=2)]

logit = tf.matmul(context, W) + b

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = logit, labels = Y))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

prediction = tf.nn.softmax(logit)