import numpy as np
import timeit
import os
import datetime
from sklearn.model_selection import train_test_split
from sklearn import svm
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import subprocess
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
### represent a sequence in form of items and sequential patterns (SPs),
### learn sequence vectors using Doc2Vec (PV-DBOW) from items and SPs separately
### take average of two sequence vectors
### use SVM as classifier

### variables ###
data_name = "exo"
path = "./data/" + data_name
minSup = 0.03
gap = 2 # 0: any gap or >0: use gap constraint

n_run = 10
class sep_training:
	def __call__(self):
		getattr(self, "run")()
	### functions ###
	# mine SPs from sequences
	def mine_SPs(self,file_seq, minSup, gap, file_seq_items_sp):
		subprocess.run("sp_miner.exe -dataset {} -minsup {} -gap {} -seqsymsp {}".
		format(file_seq, minSup, gap, file_seq_items_sp),shell=True)

	# load sequences in form of items and their labels
	def load_seq_items(self,file_name):
		labels, sequences = [], []	  
		with open(file_name) as f:
				for line in f:
						label, content = line.split("\t")
						if content != "\n":
								labels.append(label)
								sequences.append(content.rstrip().split(" "))
		return sequences, labels
	# load sequences in form of SPs and their labels
	def load_seq_SPs(self,file_name):
		labels, sequences = [], []
		with open(file_name) as f:
				for line in f:
						label, content = line.split("\t")
						labels.append(label)
						sequences.append(content.rstrip().split(" "))
		return sequences, labels
	# create a sequence id to each sequence
	def assign_sequence_id(self,sequences):
		sequences_with_ids = []
		for idx, val in enumerate(sequences):
				sequence_id = "s_{}".format(idx)
				sequences_with_ids.append(TaggedDocument(val, [sequence_id]))
		return sequences_with_ids
	def ___init___(self):
		pass
	def run(self):
		start_date_time = datetime.datetime.now()
		start_time = timeit.default_timer()
		print("### sqn2vec_sep_classify, data: {}, minSup={}, gap={}, dim={} ###".format(data_name, minSup, gap, dim))
		# mine SPs and associate each sequence with a set of SPs
		in_seq = path + "/{}.txt".format(data_name)
		out_seq_sp = path + "/{}_seq_sp_{}_{}.txt".format(data_name, minSup, gap)
		self.mine_SPs(in_seq, minSup, gap, out_seq_sp)
		# load sequences in the form of items
		data_path = path + "/" + data_name + ".txt"
		data_i_X, data_i_y = self.load_seq_items(data_path)
		# assign a sequence id to each sequence
		data_seq_i = self.assign_sequence_id(data_i_X)
		# load data in the form of patterns
		data_path = path + "/{}_seq_sp_{}_{}.txt".format(data_name, minSup, gap)
		data_p_X, data_p_y = self.load_seq_SPs(data_path)
		# assign a sequence id to each sequence
		data_seq_p = self.assign_sequence_id(data_p_X)
		print(data_seq_p)
		all_acc, all_mic, all_mac = [], [], []
		for run in range(n_run):
				print("run={}".format(run))
				# learn sequence vectors using Doc2Vec (PV-DBOW) from items
				d2v_i = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
				d2v_i.build_vocab(data_seq_i)
				d2v_i.train(data_seq_i, total_examples=d2v_i.corpus_count, epochs=50)
				data_i_vec = [d2v_i.docvecs[idx] for idx in range(len(data_seq_i))]
				del d2v_i  # delete unneeded model memory
				# learn sequence vectors using Doc2Vec (PV-DBOW) from SPs
				d2v_p = Doc2Vec(vector_size=dim, min_count=0, workers=16, dm=0, epochs=50)
				d2v_p.build_vocab(data_seq_p)
				d2v_p.train(data_seq_p, total_examples=d2v_p.corpus_count, epochs=50)
				data_p_vec = [d2v_p.docvecs[idx] for idx in range(len(data_seq_p))]
				del d2v_p  # delete unneeded model memory
				# take average of sequence vectors
				data_i_vec = np.array(data_i_vec).reshape(len(data_i_vec), dim)
				data_p_vec = np.array(data_p_vec).reshape(len(data_p_vec), dim)
				data_vec = (data_i_vec + data_p_vec) / 2

				# generate train and test vectors using 10-fold CV
				train_vec, test_vec, train_y, test_y = \
								train_test_split(data_vec, data_p_y, test_size=0.1, random_state=run, stratify=data_p_y)
				svm_d2v = svm.LinearSVC()
				# classify test data
				svm_d2v.fit(train_vec, train_y)
				test_pred = svm_d2v.predict(test_vec)
				acc = accuracy_score(test_y, test_pred)
				mic = f1_score(test_y, test_pred, pos_label=None, average="micro")
				mac = f1_score(test_y, test_pred, pos_label=None, average="macro")
				all_acc.append(acc)
				all_mic.append(mic)
				all_mac.append(mac)
				# obtain accuracy and F1-scores
				print("accuracy: {}".format(np.round(acc, 4)))
				print("micro: {}".format(np.round(mic, 4)))
				print("macro: {}".format(np.round(mac, 4)))
		print("avg accuracy: {} ({})".format(np.round(np.average(all_acc), 4), np.round(np.std(all_acc), 3)))
		print("avg micro: {} ({})".format(np.round(np.average(all_mic), 4), np.round(np.std(all_mic), 3)))
		print("avg macro: {} ({})".format(np.round(np.average(all_mac), 4), np.round(np.std(all_mac), 3)))
		end_date_time = datetime.datetime.now()
		end_time = timeit.default_timer()
		print("start date time: {} and end date time: {}".format(start_date_time, end_date_time))
		print("runtime: {}(s)".format(round(end_time-start_time, 2)))


