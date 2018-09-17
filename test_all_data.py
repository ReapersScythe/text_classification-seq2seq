import tensorflow as tf
import json
import jieba

batch_size = 50


source = []
target = []

with open("training_data/training_data.txt", 'r') as f:
    lines = f.readlines()
    for line in lines:
        line = line.strip().split("\t")
        source.append(line[0])
        target.append(line[1])


def source_extract_character_vocab():
    with open('training_op/source_trained_model_sequence.json') as f:
        data = list(f)
    source_int_to_vocab = json.loads(data[0])
    source_vocab_to_int = json.loads(data[1])

    return source_int_to_vocab, source_vocab_to_int

def target_extract_character_vocab():
    with open('training_op/target_trained_model_sequence.json') as f:
        data = list(f)
    target_int_to_vocab = json.loads(data[0])
    target_vocab_to_int = json.loads(data[1])

    return target_int_to_vocab, target_vocab_to_int

special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']

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
# 输入层
def get_inputs():
    inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')

    # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
    target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
    max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
    source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

    return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length


def source_to_seq(text):
    sequence_length = 7
    temp = [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in jieba.cut(text)] + [
        source_letter_to_int['<PAD>']] * (sequence_length - len(text))
    return [int(i) for i in temp]


checkpoint = "training_op/trained_model_test.ckpt"
loaded_graph = tf.Graph()




correct_count = 0
with tf.Session(graph=loaded_graph) as sess:
    loader = tf.train.import_meta_graph(checkpoint + '.meta')
    loader.restore(sess, checkpoint)
    for i in range(len(source)):
        text = source_to_seq(source[i])
        input_data = loaded_graph.get_tensor_by_name('inputs:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                      target_sequence_length: [len(source[i])] * batch_size,
                                      source_sequence_length: [len(source[i])] * batch_size})[0]


        pad = source_letter_to_int["<PAD>"]

        #print('原始输入:', i)

        #print('Source: ', end=' ')
        #print('  Word 编号:    {}'.format([i for i in text]))
        #print('  Input Words: {}'.format(source[i]),end='\t')
        #print('Target:', end=' ')
        #print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
        output_temp = []
        for j in [target_int_to_letter[str(i_2)] for i_2 in answer_logits if i_2 != pad]:
            if j not in special_words:
                output_temp.append(j)
        if len(output_temp) == 0:
            #print(answer_logits)
            print('Input Words: {}'.format(source[i])+' Target: '+target[i]+' | '+'None\t'+str(correct_count)+'/958')
        else:
            if ''.join(output_temp) == target[i].strip('\r\t\n'):
                correct_count += 1
                #print()
                #print('Input Words: {}'.format(source[i])+' Target: '+target[i]+' | '+''.join(output_temp) + '\t'+str(correct_count)+'/836')
            else:
                print('Input Words: {}'.format(source[i])+' Target: ' +target[i]+' | '+''.join(output_temp) + '\t' + str(correct_count) + '/958')
