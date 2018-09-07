# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import time

def get_batches(encoded,batch_size,sequence_length):

    size = batch_size * sequence_length

    batch_num = int(len(encoded)/size)

    encoded = encoded[:size*batch_num]

    encoded = encoded.reshape((batch_size,-1))

    for n in range(0, encoded.shape[1],sequence_length):
        x = encoded[:, n:n+sequence_length]


        target_data = np.zeros_like(x)

        # train_data = x[:,:-1]
        # target_data = x[:,1:] ### failed because you idiot define target_data as the shape of x (100,100), fuck:D

        target_data[:, :-1], target_data[:, -1] = x[:, 1:], x[:, 0]

        yield x, target_data


def input_layer(batch_size,sequence_length):

    input = tf.placeholder(tf.int32,shape=(batch_size,sequence_length),name = 'input')

    ground_truth = tf.placeholder(tf.int32,shape=(batch_size,sequence_length),name = 'ground_truth')

    keep_prob = tf.placeholder(tf.float32, name = 'keep_prob')

    return input, ground_truth, keep_prob

def lstm_layer(lstm_size,layer_num,batch_size,keep_prob):

    # lstm = tf.contrib.rnn.LSTMCell(lstm_size)  # basic lstm cell

    # first of all define a single LSTM cell, then wrap with Dropout, finally use numLayers to dicide the RNN depth

    stack_rnn = []
    for i in range(layer_num):
        stack_rnn.append(tf.contrib.rnn.BasicLSTMCell(lstm_size))

    cell = tf.contrib.rnn.MultiRNNCell(stack_rnn)  # MuiltRNNCell take a list as input

    dropout = tf.contrib.rnn.DropoutWrapper(cell,output_keep_prob = keep_prob)  # define dropout for every cell

    # lstm_cell_list = [dropout for _ in xrange(layer_num)]  # make cells to a list



    initial_state = cell.zero_state(batch_size,tf.float32)  # state is 0 at the beginning

    return cell,initial_state

def output_layer(lstm_output,in_size,out_size):

    seq_output = tf.concat(lstm_output,1)

    x = tf.reshape(seq_output, [-1,in_size])  # reshape lstm layer output to in_size, in_size is equal to lstm_size

    with tf.variable_scope('softmax'):

        softmax_w = tf.Variable(tf.truncated_normal([in_size,out_size], stddev = 0.1))  # tf.truncated_normal create a Normal distribution tensor [in_size,out_size]

        softmax_b = tf.Variable(tf.zeros(out_size))

    logits = tf.matmul(x,softmax_w) + softmax_b  # wx+b

    out = tf.nn.softmax(logits, name = 'prediction')

    return out, logits

def build_loss(logits,ground_truth,lstm_size,classes_num):

    gt_one_hot = tf.one_hot(ground_truth,classes_num)

    gt_reshape = tf.reshape(gt_one_hot, logits.get_shape())

    loss = tf.nn.softmax_cross_entropy_with_logits(logits = logits, labels = gt_reshape)  # loss

    loss = tf.reduce_mean(loss)

    return loss

def build_optimizer(loss,learning_rate,grad_clip):

    tvars = tf.trainable_variables();

    grads, _ = tf.clip_by_global_norm(tf.gradients(loss,tvars), grad_clip)

    train_op = tf.train.AdamOptimizer(learning_rate)

    optimizer = train_op.apply_gradients(zip(grads,tvars))

    return optimizer

def pick_top_n(pred,char_num,n):

    n = 5;

    p = np.squeeze(pred)

    p[np.argsort(p)[:-n]] = 0  # np.argsort(p)[:-n] sort p from samll to large, and return index, here zeros every elemt to 0 except the n_biggest

    p = p/np.sum(p)  # regularization

    c = np.random.choice(char_num,1,p=p)[0]

    return c

def sample(checkpoint, n_samples, lstm_size, vocab, prime=""):

    samples = [c for c in prime]

    model = charRNN(char_num, batch_size = 1,sequence_length = 1,
                lstm_size = lstm_size,
                layer_num = layer_num,
                learning_rate = learning_rate )

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess,checkpoint)
        new_state = sess.run(model.initial_state)
        for c in prime:
            x = np.zeros((1,1))  # one row and one col
            x[0,0] = vocab_to_int[c]
            feed = {model.inputs: x,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            prediction, new_state = sess.run([model.prediction,model.final_state],feed_dict=feed)

        c = pick_top_n(prediction,char_num,5)

        samples.append(int_to_vocab[c])

        for i in range(n_samples):
            x[0,0] = c  # feed one when doing sample
            feed = {
                model.inputs: x,
                model.keep_prob: 1.,
                model.initial_state: new_state
            }
            prediction, new_state = sess.run([model.prediction,model.final_state],feed_dict=feed)

            c = pick_top_n(prediction,char_num,5)

            samples.append(int_to_vocab[c])

    return ''.join(samples)


class charRNN:

    def __init__(self,
                 classes_num,
                 batch_size ,
                 sequence_length ,
                 lstm_size ,
                 layer_num ,
                 learning_rate ,
                 grad_clip = 5):

        batch_size, sequence_length = batch_size, sequence_length

        tf.reset_default_graph()

        self.inputs, self.ground_truth, self.keep_prob = input_layer(batch_size, sequence_length)

        cell, self.initial_state = lstm_layer(lstm_size,layer_num,batch_size,self.keep_prob)

        in_one_hot = tf.one_hot(self.inputs,classes_num)

        outputs, state = tf.nn.dynamic_rnn(cell, in_one_hot, initial_state = self.initial_state)

        self.final_state = state

        self.prediction, self.logits = output_layer(outputs,lstm_size,classes_num)

        self.loss = build_loss(self.logits,self.ground_truth,lstm_size,classes_num)

        self.optimizer = build_optimizer(self.loss,learning_rate,grad_clip)






with open('sample2.txt', 'rb') as f:
    # use 'rb' instead of 'r' to read chinese charactor
    text = f.read()

lines_of_text = text.split('\n')
print lines_of_text[0].decode("utf-8")

print 'Number of characters in document {0}'.format(len(text))  # TODO: cannot read file when text was chinese charactor
vocab = set(text)

char_num = len(vocab)

print char_num

vocab_to_int = {c: i for i, c in enumerate(vocab)}  # c:i pair

print vocab_to_int

int_to_vocab = dict(enumerate(vocab))

print int_to_vocab

encoded = np.array([vocab_to_int[c] for c in text],dtype=np.int32)  # translate char value to int key


text[:100]

encoded[:100]


print("opened")

# train
train_model = False

epochs = 20

save_every_n = 100

batch_size = 100        # Sequences per batch
sequence_length = 100          # Number of sequence steps per batch
lstm_size = 512         # Size of hidden layers in LSTMs
layer_num = 2          # Number of LSTM layers
learning_rate = 0.001    # Learning rate
keep_prob = 0.5 # Dropout keep probability

model = charRNN(classes_num = char_num,
                batch_size = batch_size,
                sequence_length = sequence_length,
                lstm_size = lstm_size,
                layer_num = layer_num,
                learning_rate = learning_rate)

saver = tf.train.Saver(max_to_keep=100)

if train_model:

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())  # init

        counter = 0

        for e in xrange(epochs):

            new_state = sess.run(model.initial_state)  # the beginning of a epoch, not batches

            counter = 0

            for x, y in get_batches(encoded, batch_size, sequence_length):

                counter += 1  # step counter

                start = time.time()

                feed = {
                    model.inputs: x,  # feed 100 when doing train
                    model.ground_truth: y,
                    model.keep_prob: keep_prob,
                    model.initial_state: new_state
                }

                batch_loss, new_state, _ = sess.run([model.loss,
                                                     model.final_state,
                                                     model.optimizer],
                                                    feed_dict=feed)

                end = time.time()

                # print
                if counter % 100 == 0:
                    print('epoch:{}/{}...'.format(e + 1, epochs),
                          'batch:{}...'.format(counter),
                          'loss:{}...'.format(batch_loss),
                          '{:.4f}sec/batch'.format((end - start)))

                if (counter % save_every_n == 0):
                    saver.save(sess, "checkpoints/i{}_l{}.ckpt".format(counter, lstm_size))

        saver.save(sess, "checkpoints_all_done.ckpt")

        print("trained")


tf.train.get_checkpoint_state('checkpoints')  # checkpoint_dir

print("get_checkpoint")

checkpoint = tf.train.latest_checkpoint('checkpoints')

print("loaded")

samp = sample(checkpoint,5000,lstm_size,char_num,prime=" ")

print samp
