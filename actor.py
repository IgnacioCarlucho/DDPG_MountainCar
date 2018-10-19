import tensorflow as tf
import numpy as np



class ActorNetwork(object):
      

    def __init__(self, sess, state_dim, action_dim, action_bound, learning_rate, tau, device='/cpu:0'):
        self.sess = sess
        self.s_dim = state_dim
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.learning_rate = learning_rate
        self.tau = tau
        self.device = device
        # Actor Network
        self.inputs, self.out, self.scaled_out = self.create_actor_network()

        self.network_params = tf.trainable_variables()

        # Target Network
        self.target_inputs, self.target_out, self.target_scaled_out = self.create_actor_network()

        self.target_network_params = tf.trainable_variables()[len(self.network_params):]

        # Op for periodically updating target network with online network
        # weights
        self.update_target_network_params = \
            [self.target_network_params[i].assign(tf.multiply(self.network_params[i], self.tau) +
                                                  tf.multiply(self.target_network_params[i], 1. - self.tau))
                for i in range(len(self.target_network_params))]
        with tf.device(self.device):     
            # This gradient will be provided by the critic network
            self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])

            # Combine the gradients here
            self.actor_gradients = tf.gradients(self.scaled_out, self.network_params, -self.action_gradient)

            # Optimization Op
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.network_params))

        self.num_trainable_vars = len(self.network_params) + len(self.target_network_params)

    def create_actor_network(self):
        with tf.device(self.device):
            # weights initialization
            w1_initial = np.random.normal(size=(self.s_dim,400)).astype(np.float32)
            w2_initial = np.random.normal(size=(400,300)).astype(np.float32)
            w3_initial = np.random.uniform(size=(300,self.a_dim),low= -0.0003, high=0.0003 ).astype(np.float32)
            # Placeholders
            inputs = tf.placeholder(tf.float32, shape=[None, self.s_dim])
            # Layer 1 without BN
            w1 = tf.Variable(w1_initial)
            b1 = tf.Variable(tf.zeros([400]))
            z1 = tf.matmul(inputs,w1)+b1
            l1 = tf.nn.relu(z1)
            # Layer 2 without BN
            w2 = tf.Variable(w2_initial)
            b2 = tf.Variable(tf.zeros([300]))
            z2 = tf.matmul(l1,w2)+b2
            l2 = tf.nn.relu(z2)
            #output layer
            w3 = tf.Variable(w3_initial)
            b3 = tf.Variable(tf.zeros([self.a_dim]))
            out  = tf.nn.tanh(tf.matmul(l2,w3)+b3)
            scaled_out = tf.multiply(out, self.action_bound)
        self.saver = tf.train.Saver()
        return inputs, out, scaled_out


    def train(self, inputs, a_gradient):
        self.sess.run(self.optimize, feed_dict={
            self.inputs: inputs,
            self.action_gradient: a_gradient
        })

    def predict(self, inputs):
        return self.sess.run(self.scaled_out, feed_dict={
            self.inputs: inputs
        })

    def predict_target(self, inputs):
        return self.sess.run(self.target_scaled_out, feed_dict={
            self.target_inputs: inputs
        })

    def update_target_network(self):
        self.sess.run(self.update_target_network_params)

    def get_num_trainable_vars(self):
        return self.num_trainable_vars
    
    def save_actor(self):
        # I am only saving the actor network
        # this is not good if you need to periodically save and restore the model
        self.saver.save(self.sess,'./actor_model.ckpt')
        print("Model saved in file: actor_model")

    
    def recover_actor(self):
        self.saver.restore(self.sess,'./actor_model.ckpt')
        #saver.restore(self.sess,'critic_model.ckpt')
    