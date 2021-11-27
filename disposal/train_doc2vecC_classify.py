
import sys, getopt
sys.path.insert(0, './sources')

from doc2vecC_model import Doc2vecC_model
from tf_loader import Doc2vecC
import numpy as np



if __name__ == '__main__':

	enwiki100_loader = Doc2vecC()

	enwiki100_loader.load_text('output', max_samps = 200)
	enwiki100_loader.load_label('n.txt', max_samps = 200)
	
	enwiki100_loader.build_vocab2idx(count_thres = 1)
	enwiki100_loader.operate_vocab2idx()

	enwiki100_loader.split_data(train_samps_per_lab = 4, val_samps_per_lab = 4)

	enwiki100_loader.build_rep_data(context_len = 6, doc_samp_len = 10, include_val = False, val_pro = 0.5, target_at_middle = False)
	enwiki100_loader.build_tuning_data_arr()


	data_loader = enwiki100_loader





	doc2vecC_model = Doc2vecC_model(weight_stddev = 0.00001, \
									bias_stddev = 0.00001, \
									gpu_idx = 1, \
									float_type = 'float64')


	doc2vecC_model.build_model(context_len = data_loader.context_len, \
								doc_samp_len = data_loader.doc_samp_len, \
								vocabs = data_loader.vocabs, \
								embed_dims = 256, \
								neg_samps = 50)


	doc2vecC_model.train_model(learning_rate = 0.001, \
								epochs = 51, \
								batch_size = 1024, \

								train_context_arr = data_loader.train_context_arr, \
								train_masked_arr = data_loader.train_masked_arr, \
								train_doc_samp_arr = data_loader.train_doc_samp_arr, \
								val_context_arr = data_loader.val_context_arr, \
								val_masked_arr = data_loader.val_masked_arr, \
								val_doc_samp_arr = data_loader.val_doc_samp_arr, \

								train_word_arr = data_loader.train_word_arr, \
								train_label_arr = data_loader.train_label_arr, \
								val_word_arr = data_loader.val_word_arr, \
								val_label_arr = data_loader.val_label_arr, \

								dropout_keep_prob = 0.1, \
								epoch_per_eval = 1, \
								batch_per_print = 1000, \
								model_save_path = './models/model.ckpt', \
								print_samps = 3)




