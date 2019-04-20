import ddpg 
import tensorflow as tf 
import tflearn 
import numpy as np 

GAMMA = 0.9

class aqmagent():
	def __init__(self, batch_size = 16):
		self.sess = tf.Session()
		self.batch_size = 16
		self.ddpgModel = ddpg.ddpgModel(self.sess)
		self.replayBuffer = ddpg.ReplayBuffer(1024)


		self.last_obs = None 
		self.last_action = None

		self.step = 0 
		self.tdloss = 0 
		self.reward = 0

	def get_action(self, obs, reward):

		obs = np.reshape(obs,(-1,3))

		action = self.ddpgModel.predict_actor(obs) # todo actor noise

		print(obs, action)

		self.reward += reward

		if self.last_obs is None:
			self.last_obs = obs 
			self.last_action = action 

		self.replayBuffer.add(self.last_obs, self.last_action, reward, False, obs)
		
		self.last_action = action 
		self.last_obs = obs

		self.step += 1

		if self.replayBuffer.size() > self.batch_size:
			s_batch, a_batch, r_batch, t_batch, s2_batch = \
				self.replayBuffer.sample_batch(self.batch_size)

			s2_batch = np.reshape(s2_batch, (-1,3))
			s_batch = np.reshape(s_batch, (-1,3))
			a_batch = np.reshape(a_batch, (-1,1))

			target_q = self.ddpgModel.predict_critic_target(s2_batch, self.ddpgModel.predict_actor_target(s2_batch))

			y_i = []
			for k in range(self.batch_size):
				if t_batch[k]:
					y_i.append(r_batch[k])
				else:
					y_i.append(r_batch[k] + GAMMA * target_q[k])


			_, tdloss = self.ddpgModel.train_critic(s_batch, a_batch, np.reshape(y_i, (self.batch_size, 1)))

			self.tdloss += tdloss

			a_outs = self.ddpgModel.predict_actor(s_batch)
			grads = self.ddpgModel.critic_gradient(s_batch, a_outs)
			self.ddpgModel.train_actor(s_batch, grads[0])


			self.ddpgModel.update_target_network()



		if self.step % 100 == 0:
			self.tdloss /= 100.0 
			self.reward /= 100.0
			print("tdloss", self.tdloss, "reward", self.reward)

			self.tdloss = 0 
			self.reward = 0 




		return action 