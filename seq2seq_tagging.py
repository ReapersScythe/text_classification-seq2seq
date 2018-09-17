import tensorflow as tf
import json
import jieba
from hyperparams import Hyperparams as hp


source = []
target = []

with open("/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_data/training_data_1.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        source.append(line[0])
        target.append(line[1])

special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

def source_extract_character_vocab():
    with open('/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_op/source_trained_model_sequence.json') as f:
        data = list(f)
    source_int_to_vocab = json.loads(data[0])
    source_vocab_to_int = json.loads(data[1])

    return source_int_to_vocab, source_vocab_to_int

def target_extract_character_vocab():
    with open('/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_op/target_trained_model_sequence.json') as f:
        data = list(f)
    target_int_to_vocab = json.loads(data[0])
    target_vocab_to_int = json.loads(data[1])

    return target_int_to_vocab, target_vocab_to_int


# 得到输入和输出的字符映射表
source_int_to_letter, source_letter_to_int = source_extract_character_vocab()
target_int_to_letter, target_letter_to_int = target_extract_character_vocab()

# 将每一行转换成字符id的list
source_int = []
target_int = []


for line in source:
    temp = []
    line = line.strip('\n')
    line = list(jieba.cut(line))
    for j in line:
        temp.append(source_letter_to_int[j])
    source_int.append(temp)

for line in target:
    line = line.strip('\n')
    target_int.append([target_letter_to_int[line]])


#======================================================================================================================#
def source_to_seq(text):
    sequence_length = 7
    temp = [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in jieba.cut(text)] + [
        source_letter_to_int['<PAD>']] * (sequence_length - len(text))
    return [int(i) for i in temp]


checkpoint = "/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_op/trained_model_test.ckpt"
loaded_graph = tf.Graph()

sess = tf.InteractiveSession(graph=loaded_graph)
loader = tf.train.import_meta_graph(checkpoint + '.meta')
loader.restore(sess, checkpoint)

input_data = loaded_graph.get_tensor_by_name('inputs:0')
logits = loaded_graph.get_tensor_by_name('predictions:0')

source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')


def tagging(input_word):

    text = source_to_seq(input_word)
    answer_logits = sess.run(logits, {input_data: [text] * hp.batch_size,
                                  target_sequence_length: [len(input_word)] * hp.batch_size,
                                  source_sequence_length: [len(input_word)] * hp.batch_size})[0]

    pad = source_letter_to_int["<PAD>"]
    result = list(set([target_int_to_letter[str(i)] for i in answer_logits if i != pad]))
    if " ".join([i for i in result if i not in special_words]) == '':
        return 'None'
    else:
        return (" ".join([i for i in result if i not in special_words]))


print(tagging('保洁服务费'))