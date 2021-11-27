
import numpy as np
import re
import random


class Doc2vecC(object):


	def __init__(self):
		pass




	def load_text(self, file_name, max_samps):

		word_list_list = []

		with open(file_name, 'r', encoding = 'latin-1') as f:
			for line in f:
				word_list_list += [[word.lower() for word in line.rstrip().replace('\\ n', ' ').split(' ') \
						if len(word) != 0]]

				if len(word_list_list) == max_samps:
					break
				
		self.word_list_list = word_list_list
		self.samps = len(word_list_list)

		print ('samps, word_list_list:	', self.samps)





	def load_label(self, file_name, max_samps):

		label_list_list = []

		with open(file_name, 'r') as f:
			for line in f:
				# label_list_list += [[max(0, int(line.rstrip()))]] # For label -1, assign it to 0 to avoid a fake class.
				label_list_list += [[line.rstrip()]] if line.rstrip() != '-1' else [['0']]

				if len(label_list_list) == max_samps:
					break

		if len(label_list_list) != self.samps:
			raise Exception('Number of labels and word lists do not match.')

		self.label_list_list = label_list_list
		self.labels = len(set([label_list[0] for label_list in label_list_list]))

		print ('labels, label_list_list, label_set:	', self.labels, label_list_list[:10])






	def build_vocab2idx(self, count_thres):

		vocab2count = {}

		for word_list in self.word_list_list:

			# if len(word_list) > 8000:
			# 	print (len(word_list), word_list)

			for word in word_list:
				vocab2count[word] = vocab2count.get(word, 0) + 1

		raw_vocabs = len(vocab2count)



		vocab2idx = {}
		next_idx = 2 # 0 for rare words, 1 for padding.

		for vocab in vocab2count.keys():

			if vocab2count[vocab] < count_thres:
				continue

			vocab2idx[vocab] = next_idx
			next_idx += 1

		self.vocab2idx = vocab2idx
		self.vocabs = len(vocab2idx) + 2

		print ('raw_vocabs, vocabs:	', raw_vocabs, self.vocabs)






	def operate_vocab2idx(self):
		self.word_list_list = [[self.vocab2idx.get(word, 0) for word in word_list] for word_list in self.word_list_list]





	def split_data(self, train_samps_per_lab, val_samps_per_lab):

		label2word_list_list = {}

		for idx in range(self.samps):
			word_list = self.word_list_list[idx]
			label = self.label_list_list[idx][0]
			label2word_list_list[label] = label2word_list_list.get(label, []) + [word_list]



		if min([len(label2word_list_list[key]) for key in label2word_list_list.keys()]) < train_samps_per_lab + val_samps_per_lab:
			print(min([len(label2word_list_list[key]) for key in label2word_list_list.keys()]))
			print ([len(label2word_list_list[key]) for key in label2word_list_list.keys()], train_samps_per_lab, val_samps_per_lab)
			raise Exception('Number of samples of some label is less than sum of train_samps_per_lab + val_samps_per_lab.')


		#label2word_list_list = {k: self.shuffle_list_list([v]) for k, v in label2word_list_list.item()}
		label_list = list(label2word_list_list.keys())


		self.train_word_list_list = self.link_list_list([label2word_list_list[key][:train_samps_per_lab] \
				for key in label_list])
		self.train_label_list_list = self.link_list_list([[[key]] * train_samps_per_lab for key in label_list])

		self.val_word_list_list = self.link_list_list([label2word_list_list[key]\
				[train_samps_per_lab: train_samps_per_lab + val_samps_per_lab] for key in label_list])
		self.val_label_list_list = self.link_list_list([[[key]] * val_samps_per_lab for key in label_list])

		self.unlabeled_word_list_list = self.link_list_list([label2word_list_list[key]\
				[train_samps_per_lab + val_samps_per_lab:] for key in label_list])
		self.unlabeled_label_list_list = self.link_list_list([[[key]] \
				* (len(label2word_list_list[key])-train_samps_per_lab-val_samps_per_lab) for key in label_list])
		print ('train_word_list_list:	', self.train_word_list_list[:5], len(self.train_word_list_list))
		print ('train_label_list_list:	', self.train_label_list_list[:5], len(self.train_label_list_list))
		print ('val_word_list_list:	', self.val_word_list_list[:5], len(self.val_word_list_list))
		print ('val_label_list_list:	', self.val_label_list_list[:5], len(self.val_label_list_list))
		print ('unlabeled_word_list_list:	', self.unlabeled_word_list_list[:5], len(self.unlabeled_word_list_list))
		print ('unlabeled_label_list_list:	', self.unlabeled_label_list_list[:5], len(self.unlabeled_label_list_list))




	def build_rep_data(self, context_len, doc_samp_len, include_val, val_pro, target_at_middle):

		if target_at_middle and context_len % 2 != 0:
			raise Exception('Context_len should be even if target_at_middle is used.')

		self.context_len = context_len
		self.doc_samp_len = doc_samp_len

		word_list_list = self.train_word_list_list + self.unlabeled_label_list_list

		if include_val:
			word_list_list += self.val_word_list_list


		window_size = context_len + 1

		context_list_list = []
		masked_list_list = []
		doc_samp_list_list = []

		for word_list in word_list_list:
			length = len(word_list)

			for start_idx in range(length - window_size):

				if target_at_middle:
					context_list = word_list[start_idx: start_idx + window_size]
					
					masked_list_list += [[context_list.pop(window_size // 2)]]
					context_list_list += [context_list]
					doc_samp_list_list += [random.sample(word_list, doc_samp_len)]

				else:

					for masked_idx in range(window_size):

						context_list = word_list[start_idx: start_idx + window_size]
						
						masked_list_list += [[context_list.pop(masked_idx)]]
						context_list_list += [context_list]

						if doc_samp_len <= len(word_list):
							doc_samp_list_list += [random.sample(word_list, doc_samp_len)]
						else:
							doc_samp_list_list += [word_list * (doc_samp_len // len(word_list)) \
									+ random.sample(word_list, doc_samp_len % len(word_list))]

		masked_list_list, context_list_list, doc_samp_list_list \
				= self.shuffle_list_list([masked_list_list, context_list_list, doc_samp_list_list])

		context_arr = np.array(context_list_list)
		masked_arr = np.array(masked_list_list)
		doc_samp_arr = np.array(doc_samp_list_list)

		total_samps = len(context_arr)
		val_samps = int(total_samps * val_pro)

		self.val_context_arr = context_arr[:val_samps]
		self.val_masked_arr = masked_arr[:val_samps]
		self.val_doc_samp_arr = doc_samp_arr[:val_samps]
		self.train_context_arr = context_arr[val_samps:]
		self.train_masked_arr = masked_arr[val_samps:]
		self.train_doc_samp_arr = doc_samp_arr[val_samps:]



		print ('number of rep samples:	', len(context_list_list))
		print ('val samps, train samps:	', len(self.val_context_arr), len(self.train_context_arr))



	def build_tuning_data_arr(self):

		self.train_word_arr = np.array(self.train_word_list_list)
		self.train_label_arr = np.array(self.train_label_list_list)
		self.val_word_arr = np.array(self.val_word_list_list)
		self.val_label_arr = np.array(self.val_label_list_list)

		train_lab2count = {}
		# for label in self.train_label_arr:
		# 	train_lab2count[label[0]] = train_lab2count.get(label[0], 0) + 1
		# val_lab2count = {}
		# for label in self.val_label_arr:
		# 	val_lab2count[label[0]] = val_lab2count.get(label[0], 0) + 1

		# print ('	train_lab2count:	', train_lab2count)
		# print ('	val_lab2count:	', val_lab2count)




	def build_tuning_data_dict(self, std_lengths, max_doc_len):

		train_len2word_list_list, train_len2label_list_list = \
				self.build_std_length_data(self.train_word_list_list, self.train_label_list_list, std_lengths, max_doc_len)
		val_len2word_list_list, val_len2label_list_list = \
				self.build_std_length_data(self.val_word_list_list, self.val_label_list_list, std_lengths, max_doc_len)
		unlabeled_len2word_list_list, unlabeled_len2label_list_list = \
				self.build_std_length_data(self.unlabeled_word_list_list, self.unlabeled_label_list_list, std_lengths, max_doc_len)

		self.train_len2word_arr = {k: np.array(v) for k, v in train_len2word_list_list.items()}
		self.train_len2label_arr = {k: np.array(v) for k, v in train_len2label_list_list.items()}
		self.val_len2word_arr = {k: np.array(v) for k, v in val_len2word_list_list.items()}
		self.val_len2label_arr = {k: np.array(v) for k, v in val_len2label_list_list.items()}
		self.unlabeled_len2word_arr = {k: np.array(v) for k, v in unlabeled_len2word_list_list.items()}
		self.unlabeled_len2label_arr = {k: np.array(v) for k, v in unlabeled_len2label_list_list.items()}

		print ('train_len2word_arr:	', [(k, v.shape) for k, v in self.train_len2word_arr.items()])
		print ('train_len2label_arr:	', [(k, v.shape) for k, v in self.train_len2label_arr.items()])
		print ('val_len2word_arr:	', [(k, v.shape) for k, v in self.val_len2word_arr.items()])
		print ('val_len2label_arr:	', [(k, v.shape) for k, v in self.val_len2label_arr.items()])
		print ('unlabeled_len2word_arr:	', [(k, v.shape) for k, v in self.unlabeled_len2word_arr.items()])
		print ('unlabeled_len2label_arr:	', [(k, v.shape) for k, v in self.unlabeled_len2label_arr.items()])




	def build_std_length_data(self, word_list_list, label_list_list, std_lengths, max_doc_len):

		samps = len(word_list_list)
		word_list_list = [word_list[:max_doc_len] for word_list in word_list_list]

		sorted_len_list = sorted([len(word_list) for word_list in word_list_list])

		gap_mul = (sorted_len_list[-1] / sorted_len_list[0]) ** (std_lengths ** -1)
		std_len_list = [int(sorted_len_list[-1] / gap_mul ** idx) for idx in range(std_lengths)][::-1]

		print ('std_len_list:	', std_len_list)

		# samps_per_len = samps // std_lengths
		# std_len_list = [sorted_len_list[- len_idx * samps_per_len - 1] for len_idx in range(std_lengths)][::-1]

		len2word_list_list = {}
		len2label_list_list = {}

		for word_list_idx in range(samps):

			word_list = word_list_list[word_list_idx]
			label_list = label_list_list[word_list_idx]

			std_len = self.get_std_len(len(word_list), std_len_list)

			len2word_list_list[std_len] = len2word_list_list.get(std_len, []) + [word_list + [1] * (std_len-len(word_list))]
			len2label_list_list[std_len] = len2label_list_list.get(std_len, []) + [label_list]


		return len2word_list_list, len2label_list_list





	def get_std_len(self, length, std_len_list):

		idx = 0

		while std_len_list[idx] < length:
			idx += 1

		return std_len_list[idx]





	def shuffle_list_list(self, list_list):

		if len(set([len(list_) for list_ in list_list])) != 1:
			raise Exception('list of different lenths found in list_list.')

		order = np.random.permutation(len(list_list[0]))

		return tuple([np.array(list_)[order].tolist() for list_ in list_list])




	def link_list_list(self, list_list):

		output_list = []

		for list_ in list_list:
			output_list += list_

		return output_list



