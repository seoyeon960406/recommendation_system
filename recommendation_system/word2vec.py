import numpy as np
import gensim

class Word2Vec():
    def __init__(self):
        self.model = None

    def load_model(self, model_name):
        model = gensim.models.word2vec.Word2Vec.load(model_name)
        self.model = model

    def corpus2vec(self, model_name, doc):
        word_vec = []
        self.load_model(model_name)

        for sentence in doc:
            sub = []
            for word in sentence:
                if word in self.model.wv.vocab:
                    sub.append(self.model.wv[word])
                else:
                    sub.append(np.random.uniform(-0.25, 0.25, 300))
            word_vec.append(sub)

        return np.array(word_vec)

    def zero_padding(self, train_batch_X, batch_size, maxseq_length, vector_size):
        zero_pad = np.zeros((batch_size, maxseq_length, vector_size))
        for i in range(batch_size):
            try:
                zero_pad[i, :np.shape(train_batch_X[i])[0],:np.shape(train_batch_X[i])[1]] = train_batch_X[i]
            except:
                continue
        return zero_pad
        
    def onehot(self, labels):
        result = []
        for label in labels:
            line = np.zeros(2)
            line[int(label)]=1
            result.append(line)
        return np.array(result)
