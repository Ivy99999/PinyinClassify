import codecs
from collections import Counter
import tensorflow.contrib.keras as kr
import numpy as np
import data_helpers


def read_file(filename):
    """读取文件数据"""
    # print(filename)
    contents, labels = [], []
    with open(filename,encoding='utf-8') as f:
        for line in f:
            # print(line)
            try:
                label, content = line.strip().split('\t')
                if content:
                    #list这里是将句子打成字
                    contents.append(list(content))
                    labels.append(label)
            except:
                pass
    return contents, labels

def build_vocab(train_dir, vocab_dir, vocab_size=5000):
    data_train, _ = read_file(train_dir)
    all_data = []
    for content in data_train:
        all_data.extend(content)
    counter = Counter(all_data)
    #将前两千个词进行词和词频的输出
    count_pairs = counter.most_common(vocab_size - 1)
    words, _ = list(zip(*count_pairs))
    # 添加一个 <PAD> 来将所有文本pad为同一长度
    words = ['<PAD>'] + list(words)
    open(vocab_dir, mode='w').write('\n'.join(words) + '\n')


def read_category():
    """读取分类目录，固定"""
    categories = ['娱乐',
    '游戏',
    '音乐',
    '星座运势',
    '体育',
    '食物',
    '时尚',
    '社会万象',
    '汽车',
    '农业',
    '母婴育儿',
    '科技',
    '军事',
    '教育',
    '健康养生',
    '国际视野',
    '搞笑',
    '动漫',
    '宠物',
    '财经',
    '历史',
    '家居',
    '房产']



    categories = [x for x in categories]

    cat_to_id = dict(zip(categories, range(len(categories))))
    # print(cat_to_id,categories)

    return categories, cat_to_id

def read_vocab(vocab_dir):
    """读取词汇表"""
    # words = open_file(vocab_dir).read().strip().split('\n')
    with open(vocab_dir) as fp:
        # 如果是py2 则每个值都转化为unicode
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id
#训练开始时候进行pad转换。
def process_file(filename, word_to_id, cat_to_id, max_length=600):
    """将文件转换为id表示"""
    contents, labels = read_file(filename)

    data_id, label_id = [], []
    for i in range(len(contents)):
        data_id.append([word_to_id[x] for x in contents[i] if x in word_to_id])
        label_id.append(cat_to_id[labels[i]])
    print(label_id)

    # 使用keras提供的pad_sequences来将文本pad为固定长度
    x_pad = kr.preprocessing.sequence.pad_sequences(data_id, max_length)
    y_pad = kr.utils.to_categorical(label_id, num_classes=len(cat_to_id))  # 将标签转换为one-hot表示

    return x_pad, y_pad

def batch_iter(x, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1
    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = x[indices]
    y_shuffle = y[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]


def export_word2vec_vectors(vocab, word2vec_dir,trimmed_filename):
    """
    Args:
        vocab: word_to_id
        word2vec_dir:file path of have trained word vector by word2vec
        trimmed_filename:file path of changing word_vector to numpy file
    Returns:
        save vocab_vector to numpy file
    """
    # ff=open('data/new.embeding','w',encoding='utf-8')
    file_r = codecs.open(word2vec_dir, 'r', encoding='utf-8')
    line = file_r.readline()
    voc_size, vec_dim = map(int, line.strip().split(' '))
    print(vec_dim+26)
    embeddings = np.zeros([len(vocab), vec_dim+26])
    line = file_r.readline()
    while line:
        try:
            items = line.split(' ')
            word = items[0]
            vec = np.asarray(items[1:], dtype='float32')
            if word in vocab:
                word_idx = vocab[word]
                py_vec = data_helpers.word_to_vec(word)
                merge_vec = np.concatenate((vec, py_vec))
                print(word_idx)
                print(np.asarray(merge_vec))
                embeddings[word_idx] = np.asarray(merge_vec)

        except:
            pass
        line = file_r.readline()
    # ff.write(embeddings)
    np.savez_compressed(trimmed_filename, embeddings=embeddings)

def get_training_word2vec_vectors(filename):
    """
    Args:
        filename:numpy file
    Returns:
        data["embeddings"]: a matrix of vocab vector
    """
    with np.load(filename) as data:
        return data["embeddings"]


def get_sequence_length(x_batch):
    """
    Args:
        x_batch:a batch of input_data
    Returns:
        sequence_lenghts: a list of acutal length of  every senuence_data in input_data
    """
    sequence_lengths=[]
    for x in x_batch:
        actual_length = np.sum(np.sign(x))
        sequence_lengths.append(actual_length)
    return sequence_lengths
