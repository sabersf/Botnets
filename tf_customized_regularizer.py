# Define a dnn using Tensorflow
# add customized regularizer that penalize the model that has high variance in prediction
with tf.Graph().as_default() as session:

    # Model variables
    X = tf.placeholder("float", [None, len(X_train[0])])
    Y = tf.placeholder("float", [None, len(Y_train[0])])

    # Multilayer perceptron
    def dnn(x):
        with tf.variable_scope('Layer1'):
            # Creating variable using TFLearn
            W1 = va.variable(name='W', shape=[len(X_train[0]), 256],
                             initializer='uniform_scaling',
                             regularizer='L2')
            b1 = va.variable(name='b', shape=[256])
            x = tf.nn.tanh(tf.add(tf.matmul(x, W1), b1))

        with tf.variable_scope('Layer2'):
            W2 = va.variable(name='W', shape=[256, 128],
                             initializer='uniform_scaling',
                             regularizer='L2')
            b2 = va.variable(name='b', shape=[128])
            x = tf.nn.tanh(tf.add(tf.matmul(x, W2), b2))

        with tf.variable_scope('Layer3'):
            W3 = va.variable(name='W', shape=[128, 64],
                             initializer='uniform_scaling')
            b3 = va.variable(name='b', shape=[64])
            x = tf.add(tf.matmul(x, W3), b3)

        with tf.variable_scope('Layer4'):
            W4 = va.variable(name='W', shape=[64, len(Y_train[0])],
                             initializer='uniform_scaling')
            b4 = va.variable(name='b', shape=[len(Y_train[0])])
            x = tf.add(tf.matmul(x, W4), b4)

        return x,W4,b4

    net, W4,b4 = dnn(X)
    const = .01
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(net, Y) + const*tf.nn.l2_loss(W4) +const*tf.nn.l2_loss(b4) )
    #optimizer = tf.train.AdadeltaOptimizer(learning_rate=0.1)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    
    accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(net, 1), tf.argmax(Y, 1)), tf.float32), name='acc')


    trainop = tflearn.TrainOp(loss=loss, optimizer=optimizer, metric=accuracy, batch_size=256)

    trainer = tflearn.Trainer(train_ops=trainop, tensorboard_verbose=3, tensorboard_dir='/tmp/tflearn_logs/')

    trainer.fit({X: X_train, Y: Y_train}, val_feed_dicts={X: X_test, Y: Y_test},
                n_epoch=100, show_metric=True, run_id='Variables_example')
    print("Accuracy on the test test: %.2f"% (100. * trainer.session.run(accuracy, feed_dict={X:X_test, Y:Y_test})))
    pred = trainer.session.run(tf.argmax(Y, 1),feed_dict={X:X_test, Y:Y_test})
