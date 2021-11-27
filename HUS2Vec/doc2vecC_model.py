
import tensorflow as tf
import numpy as np
import random
from sklearn.svm import LinearSVC

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

config = tf.compat.v1.ConfigProto(allow_soft_placement = True, log_device_placement = False)
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = 0.98




class Doc2vecC_model(object):



	def __init__(self, weight_stddev, bias_stddev, gpu_idx, float_type):
		
		self.weight_stddev = weight_stddev
		self.bias_stddev = bias_stddev
		self.gpu_idx = gpu_idx
		self.float_type = float_type

		print ('weight_stddev:	', weight_stddev)
		print ('bias_stddev:	', bias_stddev)
		print ('gpu_idx:	', gpu_idx)
		print ('float_type:	', float_type)




	def build_model(self, context_len, doc_samp_len, vocabs, embed_dims, neg_samps):
		'''
		Difference from the paper:
			Boolean for BOW instead of count.
			Fixed proportion of words from document instead of fix number. The author used 0.1.
		'''

		print ('context_len:	', context_len)
		print ('doc_samp_len:	', doc_samp_len)
		print ('vocabs:	', vocabs)
		print ('embed_dims:	', embed_dims)
		print ('neg_samps:	', neg_samps)



		with tf.device('/GPU:' + str(self.gpu_idx)):

			with tf.variable_scope('input'):

				context_ph = tf.placeholder('int32', shape = [None, context_len], name = 'context_ph')
				doc_samp_ph = tf.placeholder('int32', shape = [None, doc_samp_len], name = 'doc_samp_ph')
				masked_ph = tf.placeholder('int32', shape = [None, 1], name = 'masked_ph')


				dropout_keep_prob_ph = tf.placeholder_with_default(1.0, shape = [], name = 'dropout_keep_prob_ph')
				keep_prob = tf.cast(dropout_keep_prob_ph, self.float_type)

				self.context_ph = context_ph
				self.doc_samp_ph = doc_samp_ph
				self.masked_ph = masked_ph
				self.dropout_keep_prob_ph = dropout_keep_prob_ph




			with tf.variable_scope('operate_idx2embedding'):

				embedding_table_ph = tf.get_variable(name = 'trainable_embedding_table', initializer = \
					tf.random_normal(shape = [vocabs, embed_dims], stddev=self.weight_stddev, dtype = self.float_type))

				vocab_table = tf.get_variable(name = 'vocab_table', initializer = \
					tf.random_normal(shape = [embed_dims, vocabs], stddev=self.weight_stddev, dtype = self.float_type))


				context_embed_ph = tf.reduce_sum(tf.nn.embedding_lookup(embedding_table_ph, context_ph), axis = 1)
				doc_samp_embed_ph = tf.reduce_mean(tf.nn.embedding_lookup(embedding_table_ph, doc_samp_ph), axis = 1)
				doc_samp_embed_ph = tf.nn.dropout(doc_samp_embed_ph, keep_prob = keep_prob)

				combined_embed_ph = context_embed_ph + doc_samp_embed_ph


				logit_ph = tf.matmul(combined_embed_ph, vocab_table)


				self.embedding_table_ph = embedding_table_ph



			with tf.variable_scope('loss'):

				neg_samp_idx_ph = tf.random_uniform([tf.shape(logit_ph)[0], neg_samps, 1], maxval = vocabs, dtype = tf.int32, name = 'neg_samp_idx_ph')
				batch_idx_expanded = tf.tile(tf.expand_dims(tf.expand_dims(tf.range(tf.shape(logit_ph)[0]), axis = 1), axis = 2), [1, neg_samps, 1])
				neg_samp_gather_indices_ph = tf.concat([batch_idx_expanded, neg_samp_idx_ph], axis = 2)

				masked_gather_indices_ph = tf.concat([tf.expand_dims(tf.range(tf.shape(logit_ph)[0]), axis = 1), masked_ph], axis = 1)

				print ('shape of neg_samp_gather_indices_ph:	', neg_samp_gather_indices_ph.shape)
				print ('shape of masked_gather_indices_ph:	', masked_gather_indices_ph.shape)

				neg_samp_logit_ph = tf.gather_nd(logit_ph, neg_samp_gather_indices_ph, name = 'neg_samp_logit_ph')
				masked_logit_ph = tf.expand_dims(tf.gather_nd(logit_ph, masked_gather_indices_ph, name = 'masked_logit_ph'), axis = 1)

				print ('shape of neg_samp_logit_ph:	', neg_samp_logit_ph.shape)
				print ('shape of masked_logit_ph:	', masked_logit_ph.shape)

				combined_loss_logit_ph = tf.concat([-neg_samp_logit_ph, masked_logit_ph], axis = 1)

				loss_ph = -tf.reduce_mean(tf.log(self.sigma(combined_loss_logit_ph)))



				self.combined_loss_logit_ph = combined_loss_logit_ph
				self.loss_ph = loss_ph





	def train_model(self, learning_rate, epochs, batch_size, train_context_arr, train_masked_arr, train_doc_samp_arr, val_context_arr, val_masked_arr, val_doc_samp_arr, train_word_arr, train_label_arr, val_word_arr, val_label_arr, dropout_keep_prob, epoch_per_eval, batch_per_print, model_save_path, print_samps):

		self.batch_size = batch_size

		print ('learning_rate:	', learning_rate)
		print ('epochs:		', epochs)
		print ('batch_size:	', batch_size)
		print ('total_params:	', self.get_total_params())
		print ('dropout_keep_prob:	', dropout_keep_prob)



		with tf.device('/GPU:' + str(self.gpu_idx)):
			optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate, name = 'optimizer').minimize(self.loss_ph)


		feed_dict = {self.context_ph:train_context_arr,self.masked_ph:train_masked_arr,self.doc_samp_ph:train_doc_samp_arr}
		val_feed_dict = {self.context_ph:val_context_arr, self.masked_ph:val_masked_arr, self.doc_samp_ph:val_doc_samp_arr}

		batches = (len(train_context_arr) - 1) // batch_size + 1


		with tf.Session(config = config) as self.sess:
			self.sess.run(tf.global_variables_initializer())

			max_train_acc_so_far = -float('inf')
			max_val_acc_so_far = -float('inf')
			max_train_loss_so_far = -float('inf')
			max_val_loss_so_far = -float('inf')

			for epoch_idx in range(epochs):
				print ('\ntraing epoch %d...' % epoch_idx)

				feed_dict = self.shuffle_dictionary(feed_dict)
				val_feed_dict = self.shuffle_dictionary(val_feed_dict)



				if epoch_idx % epoch_per_eval == 0:

					train_loss = self.sess_run_by_batch_size(self.loss_ph, feed_dict, average_all = True, hyper_para_feed_dict = {})
					val_loss = self.sess_run_by_batch_size(self.loss_ph, val_feed_dict, average_all = True, hyper_para_feed_dict = {})

					train_acc, val_acc = self.evaluate(train_word_arr, train_label_arr, val_word_arr, val_label_arr)

					print ('training loss:	', train_loss)
					print ('validate loss:	', val_loss)
					print ('training acc:	', train_acc)
					print ('validate acc:	', val_acc)





					if train_acc > max_train_acc_so_far:
						max_train_acc_so_far = train_acc
					
					if val_acc > max_val_acc_so_far:
						max_val_acc_so_far = val_acc

						saver = tf.train.Saver()
						real_save_path = saver.save(sess = self.sess, save_path = model_save_path)
						print ('NEW HIGH! model saved as ', real_save_path)

					print ('max training acc:	', max_train_acc_so_far)
					print ('max validate acc:	', max_val_acc_so_far)





					if print_samps:

						logit_arr = self.sess_run_by_batch_size(self.combined_loss_logit_ph, \
								feed_dict = self.get_batch(val_feed_dict, 0, print_samps), average_all = 0, \
								hyper_para_feed_dict = {})

						print ('logit:\n', logit_arr)
						# print ('pred: \n', np.greater(logit_arr, np.expand_dims(thres_arr, axis = 0)))
						# print ('label: \n', val_feed_dict[self.combined_loss_logit_ph][:print_samps])





				for batch_idx in range(batches):

					if batch_idx % batch_per_print == 0:
						print ('training batch %d...' % batch_idx)
					
					batch_feed_dict = self.get_batch(feed_dict, batch_idx * batch_size, batch_size)

					batch_feed_dict[self.dropout_keep_prob_ph] = dropout_keep_prob

					self.sess.run(optimizer, feed_dict = batch_feed_dict)







	def evaluate(self, train_word_arr, train_label_arr, val_word_arr, val_label_arr):

		embedding_table = self.sess.run(self.embedding_table_ph)

		overall_doc_embed_arr = None
		overall_label_arr = None

		train_doc_embed_arr = self.get_doc_embed_arr(train_word_arr, embedding_table)
		train_label_arr = np.squeeze(train_label_arr, axis = 1)
		val_doc_embed_arr = self.get_doc_embed_arr(val_word_arr, embedding_table)
		val_label_arr = np.squeeze(val_label_arr, axis = 1)

		print ('shape of train_doc_embed_arr:	', train_doc_embed_arr.shape)

		classifier = LinearSVC()
		classifier.fit(train_doc_embed_arr, train_label_arr)

		train_pred_arr = classifier.predict(train_doc_embed_arr)
		val_pred_arr = classifier.predict(val_doc_embed_arr)

		# print ('train_pred_arr:	', self.print_distrib_dict(train_pred_arr))
		# print ('train_label_arr:	', self.print_distrib_dict(train_label_arr))
		# print ('val_pred_arr:	', self.print_distrib_dict(val_pred_arr))
		# print ('val_label_arr:	', self.print_distrib_dict(val_label_arr))


		return np.sum(train_pred_arr == train_label_arr) / len(train_pred_arr), \
				np.sum(val_pred_arr == val_label_arr) / len(val_pred_arr)




	def print_distrib_dict(self, arr):

		distrib_dict = {}

		for x in arr:
			distrib_dict[x] = distrib_dict.get(x, 0) + 1

		print (distrib_dict)






	def get_doc_embed_arr(self, word_arr, embedding_table):

		doc_embed_arr = np.zeros((word_arr.shape[0], embedding_table.shape[1]))

		for row_idx in range(len(word_arr)):
			row_word_arr = word_arr[row_idx]

			total_embedding = np.zeros((embedding_table.shape[1]))

			for word_idx in row_word_arr:
				total_embedding += embedding_table[word_idx, :]

			doc_embed_arr[row_idx, :] = total_embedding / len(row_word_arr)

		return doc_embed_arr


	def get_batch_tup_order_list(self, batches_list):

		batch_tup_order_list = []

		for idx1 in range(len(batches_list)):
			for idx2 in range(batches_list[idx1]):
				batch_tup_order_list += [(idx1, idx2)]

		return batch_tup_order_list



	def get_total_params(self):

		total_params = 0

		for variable in tf.trainable_variables():
			print (variable)
			params = 1

			for dim in variable.get_shape():
				params *= dim.value

			total_params += params

		return total_params



	def sess_run_by_batch_size(self, ph_to_eval, feed_dict, average_all, hyper_para_feed_dict):

		samp_num = len(list(feed_dict.items())[0][1])
		samp_idx = 0

		if average_all:
			value_sum = 0
			value_num = 0
		else:
			outcome_list = []

		while samp_idx < samp_num:
			real_batch_size = min(self.batch_size, samp_num - samp_idx)
			outcome = self.sess.run(ph_to_eval, feed_dict = {**hyper_para_feed_dict, **self.get_batch(feed_dict, samp_idx, real_batch_size)})

			if average_all:
				value_sum += real_batch_size * outcome
				value_num += real_batch_size
			else:
				outcome_list += [outcome]

			samp_idx += self.batch_size

		final_return = value_sum / value_num if average_all else np.concatenate(outcome_list, axis = 0)
		return final_return



	def shuffle_dictionary(self, dictionary):

		samples = len(list(dictionary.items())[0][1])
		order = np.random.permutation(samples)

		return {k: v[order] for k, v in dictionary.items()}



	def get_batch(self, dictionary, start_idx, size):

		return {k: v[start_idx: start_idx + size] for k, v in dictionary.items()}



	def dense_layer(self, input_ph, output_dim, var_name):

		input_dim = tf.cast(input_ph.shape[1], tf.int32)

		weight = tf.get_variable(name = var_name + '_weight', initializer = \
				tf.random_normal(shape = [input_dim, output_dim], stddev=self.weight_stddev, dtype = self.float_type))
		
		bias = tf.get_variable(name = var_name + '_bias', initializer = \
				tf.random_normal(shape = [output_dim], stddev = self.bias_stddev, dtype = self.float_type))

		return tf.matmul(input_ph, weight) + bias




	def leaky_relu(self, ph, alpha):
		return tf.maximum(ph, 0.1 * ph)




	def sigma(self, in_ph):
		return 1 / (1 + tf.exp(-in_ph))





