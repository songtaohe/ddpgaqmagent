import tensorflow as tf 
import tflearn

class ddpg():
	def __init__(self, sess, state_dim = 3, action_dim = 1, batch_size = 16, tau = 0.001):

		self.sess = sess
		self.tau = tau
		self.state_dim = state_dim 
		self.action_dim = action_dim
		self.batch_size = batch_size
		self.learning_rate = 0.001

		self.inputstate = tf.placefolder(tf.float32, shape=(None, state_dim))
		self.inputaction = tf.placefolder(tf.float32, shape(None, 1))


		self.actor_output = self.create_actor_network(self.inputstate)
		actor_network_params = tf.trainable_variables()

		self.qvalue = self.create_critic_network(self.inputstate, self.inputaction)
		network_params = tf.trainable_variables()


		self.actor_target_output = self.create_actor_network(self.inputstate)
		self.qvalue_target = self.create_critic_network(self.inputstate, self.inputaction)

		target_network_params = tf.trainable_variables()[len(network_params):]




		self.update_target_network_params = \
		[target_network_params[i].assign(tf.mul(network_params[i], self.tau) + \
			tf.mul(target_network_params[i], 1. - self.tau))
			for i in range(len(target_network_params))]





		self.action_gradient = tf.placeholder(tf.float32, [None, self.action_dim])

		self.unnormalized_actor_gradients = tf.gradients(
			self.actor_output, actor_network_params, -self.action_gradient)

		self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))

		# Optimization Op
		self.optimize_actor = tf.train.AdamOptimizer(self.learning_rate).\
			apply_gradients(zip(self.actor_gradients, actor_network_params))



		self.predicted_qvalue = tf.placeholder(tf.float32, [None,1])

		self.critic_loss = tflearn.mean_square(self.predicted_qvalue, self.qvalue)
		self.optimize_critic = tf.train.AdamOptimizer(self.learning_rate).minimize(self.critic_loss)

		self.action_gradient_output = tf.gradients(self.qvalue, self.inputaction)



		self.sess.run(tf.global_variables_initializer())
		self.saver = tf.train.Saver(max_to_keep=20)

		pass 


	def create_actor_network(self, state):

		net = state
		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

		nns = [100,100,100,1]
		for i in range(len(nns)):
			nn = nns[i]

			net = tflearn.fully_connected(net, nn, weights_init=w_init)
			
			if i < len(nns):
				net = tflearn.layers.normalization.batch_normalization(net)
				net = tflearn.activations.relu(net)
			else:
				net = tflearn.activations.sigmod(net)

		return net 
	


	def create_critic_network(self, state, action):
		net = tf.concat([state,action], axis=1)

		w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)

		nns = [100,100,100,1]
		for i in range(len(nns)):
			nn = nns[i]

			net = tflearn.fully_connected(net, nn, weights_init=w_init)
			
			if i < len(nns):
				net = tflearn.layers.normalization.batch_normalization(net)
				net = tflearn.activations.relu(net)
			else:
				pass #linear output

		return net 


	def train_actor(self, state, a_gradient):
		self.sess.run(self.optimize_actor, feed_dict={
			self.inputstate : state,
			self.action_gradient : a_gradient
			})

	def predict_actor(self, state):
		return self.sess.run(self.actor_output, feed_dict = {self.inputstate : state})

	def predict_actor_target(self, state):
		return self.sess.run(self.actor_target_output, feed_dict = {self.inputstate : state})


	def train_critic(self, state, action, qvalue):
		return  self.sess.run([self.optimize_critic, self.critic_loss], feed_dict = {
			self.inputstate : state,
			self.inputaction : action, 
			self.predicted_qvalue : qvalue,
			})

	def predict_critic(self, state, action):
		return self.sess.run(self.qvalue, feed_dict = {
			self.inputstate : state,
			self.inputaction : action
			})


	def predict_critic_target(self, state, action):
		return self.sess.run(self.qvalue_target, feed_dict = {
			self.inputstate : state,
			self.inputaction : action
			})


	def critic_gradient(self, state, action):
		return self.sess.run(self.action_gradient_output, feed_dict = {
			self.inputstate : state,
			self.inputaction : action
			})


	def update_target_network(self):
		self.sess.run(self.update_target_network_params)


	def saveModel(self, path):
		self.saver.save(self.sess, path)

	def restoreModel(self, path):
		self.saver.restore(self.sess, path)