# encoding: UTF-8

from pinyindeal import PinYin


def word_to_pinyin(word):
    test = PinYin()
    test.load_word()
    word_py = test.hanzi2pinyin_split(string=word,split="")
    return word_py

#将拼音转化为向量
#并且对于预处理当中没有过滤的字母赋予零向量
def word_to_vec(word):
    string = word_to_pinyin(word)
    # print(string)
    vec = [0]*26
    for char in string:
        if char.isalpha():
            loc = ord(char)- ord('a')
            if loc not in range(0,26):
                vec = [0 for i in range(26)]
                return vec
            if vec[loc]==1:
                vec[loc] +=1
            else:
                vec[loc] = 1
    return vec

# def load_data_and_labels(input_text_file, input_label_file, num_labels):
#     x_text = read_and_clean_zh_file(input_text_file)
#     y = None if not os.path.exists(input_label_file) else map(int, list(open(input_label_file, "r").readlines()))
#     return (x_text, y)
#
# def load_positive_negative_data_files(positive_data_file, negative_data_file):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     positive_examples = read_and_clean_zh_file(positive_data_file, "pos_out.txt")
#     negative_examples = read_and_clean_zh_file(negative_data_file, "neg_out.txt")
#     # Combine data
#     x_text = positive_examples + negative_examples
#     # Generate labels
#     positive_labels = [[0, 1] for _ in positive_examples]
#     negative_labels = [[1, 0] for _ in negative_examples]
#     y = np.concatenate([positive_labels, negative_labels], 0)
#     return [x_text, y]
#
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# #这个是训练集
#
# def load_positive_negative_dev_data_files(positive_dev_data_file, negative_dev_data_file):
#     """
#     Loads MR polarity data from files, splits the data into words and generates labels.
#     Returns split sentences and labels.
#     """
#     # Load data from files
#     positive_examples = read_and_clean_zh_file(positive_dev_data_file, "pos_dev_out.txt")
#     negative_examples = read_and_clean_zh_file(negative_dev_data_file, "neg_dev_out.txt")
#     # Combine data
#     x_text = positive_examples + negative_examples
#     # Generate labels
#     positive_labels = [[0, 1] for _ in positive_examples]
#     negative_labels = [[1, 0] for _ in negative_examples]
#     y = np.concatenate([positive_labels, negative_labels], 0)
#     return [x_text, y]
#
# #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#
# def padding_sentences(input_sentences, padding_token, padding_sentence_length = None):
#     sentences = [sentence.split(' ') for sentence in input_sentences]
#     max_sentence_length = padding_sentence_length if padding_sentence_length is not None else max([len(sentence) for sentence in sentences])
#     for sentence in sentences:
#         if len(sentence) > max_sentence_length:
#             sentence = sentence[:max_sentence_length]
#         else:
#             sentence.extend([padding_token] * (max_sentence_length - len(sentence)))
#     return (sentences, max_sentence_length)
#
# def batch_iter(data, batch_size, num_epochs, shuffle=True):
#     '''
#     Generate a batch iterator for a dataset
#     '''
#     data = np.array(data)
#     data_size = len(data)
#     num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
#     for epoch in range(num_epochs):
#         if shuffle:
#             # Shuffle the data at each epoch
#             shuffle_indices = np.random.permutation(np.arange(data_size))
#             shuffled_data = data[shuffle_indices]
#         else:
#             shuffled_data = data
#         for batch_num in range(num_batches_per_epoch):
#             start_idx = batch_num * batch_size
#             end_idx = min((batch_num + 1) * batch_size, data_size)
#             yield shuffled_data[start_idx: end_idx]
#
# def test():
#     # Test clean_str
#     print("Test")
#     #print(clean_str("This's a huge dog! Who're going to the top."))
#     # Test load_positive_negative_data_files
#     #x_text,y = load_positive_negative_data_files("./tiny_data/rt-polarity.pos", "./tiny_data/rt-polarity.neg")
#     #print(x_text)
#     #print(y)
#     # Test batch_iter
#     #batches = batch_iter(x_text, 2, 4)
#     #for batch in batches:
#     #    print(batch)
#
# def mkdir_if_not_exist(dirpath):
#     if not os.path.exists(dirpath):
#         os.mkdir(dirpath)
#
#
# def seperate_line(line):
#     #当训练词向量就加上这几句话
#     word_list = jieba.lcut(line)
#     return ''.join([word + ' ' for word in word_list])
#     #训练字向量
#     #return ''.join([word + ' ' for word in line])
#
# def read_and_clean_zh_file(input_file, output_cleaned_file = None):
#     lines = list(open(input_file, "r").readlines())
#     lines = [clean_str(seperate_line(line)) for line in lines]
#     if output_cleaned_file is not None:
#         with codecs.open(output_cleaned_file, mode='w', encoding='utf-8') as f:
#             for line in lines:
#                 f.write((line + '\n'))
#     return lines
#
# def clean_str(string):
#     """
#     Tokenization/string cleaning for all datasets except for SST.
#     Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
#     """
#     #这句话：是只保留中文汉字
#     string = re.sub(r"[^\u4e00-\u9fff]", " ", string)
#     #string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
#     #string = re.sub(r"\'s", " \'s", string)
#     #string = re.sub(r"\'ve", " \'ve", string)
#     #string = re.sub(r"n\'t", " n\'t", string)
#     #string = re.sub(r"\'re", " \'re", string)
#     #string = re.sub(r"\'d", " \'d", string)
#     #string = re.sub(r"\'ll", " \'ll", string)
#     #string = re.sub(r",", " , ", string)
#     #string = re.sub(r"!", " ! ", string)
#     #string = re.sub(r"\(", " \( ", string)
#     #string = re.sub(r"\)", " \) ", string)
#     #string = re.sub(r"\?", " \? ", string)
#     string = re.sub(r"\s{2,}", " ", string)
#     #return string.strip().lower()
#     return string.strip()
#
# def saveDict(input_dict, output_file):
#     with open(output_file, 'wb') as f:
#         pickle.dump(input_dict, f)
#
# def loadDict(dict_file):
#     output_dict = None
#     with open(dict_file, 'rb') as f:
#         output_dict = pickle.load(f)
#     return output_dict
