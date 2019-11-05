import os

from data_helper import export_word2vec_vectors, build_vocab, read_category, read_vocab
from rnn_model import RNNConfig


base_dir = 'data/'
train_dir = os.path.join(base_dir, 'cnews.train.txt')
test_dir = os.path.join(base_dir, 'cnews.test.txt')
val_dir = os.path.join(base_dir, 'cnews.val.txt')
vocab_dir = os.path.join(base_dir, 'cnews.vocab.txt')
words_embeding=os.path.join(base_dir, 'cnews.embeding.txt')
vector_word_npz= os.path.join(base_dir, 'cnews.embeding.npz')

config = RNNConfig()  # 获取配置参数
if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
    build_vocab(train_dir, vocab_dir, config.vocab_size)
categories, cat_to_id = read_category()
words, word_to_id = read_vocab(vocab_dir)
config.vocab_size = len(words)
export_word2vec_vectors(word_to_id, words_embeding, vector_word_npz)