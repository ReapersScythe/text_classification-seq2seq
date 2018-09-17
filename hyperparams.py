#Hyperparams for Seq2Seq Tagging

class Hyperparams:
    #data source
    source_train = '/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_op/source.txt'
    target_train = '/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_op/target.txt'
    source_test = '/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_op/source.txt'
    target_test = '/home/frankqi/Desktop/NLP_Deloitte/SPO_test/SPO_test/Seq2Seq_tool/seq2seq/training_op/target.txt'

    # Number of Epochs
    epochs = 300
    # Batch Size
    batch_size = 500
    # RNN Size
    rnn_size = 200
    # Number of Layers
    encoder_layers = 2
    decoding_layers = 2
    # Embedding Size
    encoding_embedding_size = 300
    decoding_embedding_size = 300
    # Learning Rate
    learning_rate = 0.001
