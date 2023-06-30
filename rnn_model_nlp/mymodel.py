from typing import List
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import copy
import nltk
from nltk.tokenize import (word_tokenize,
                           sent_tokenize)
import torch.nn.functional as nnf

random.seed(911001)
np.random.seed(911001)
torch.manual_seed(911001)

loss_fn = nn.CrossEntropyLoss()
# 超参设置
BATCH_SIZE = 1                # critical：一个batch大小，代表训练的每个step输入BATCH_SIZE个句子到模型中。一般来说，提高BATCH_SIZE有助于提高模型性能
EMBEDDING_SIZE = 128        # critical：词嵌入向量大小
MAX_VOCAB_SIZE = 50000        # critical：字典大小，代表有多少个词/字

GRAD_CLIP = 1               # 梯度裁剪，防止模型的backward的梯度过大，导致模型训偏
NUM_EPOCHS = 100             # critical：训练的轮次

words = set()
word2id = {"<pad>":0}
id2word = {0: "<pad>"}
lines = []

# critical：定义数据集
class Article_Dataset(Dataset):
    def __init__(self, word2id:dict, id2word:dict, texts:List[List[str]]) -> None:
        super().__init__()
        self.word2id = word2id
        self.id2word = id2word
        
        process_text = []
        for text in texts:
            process_text.append(self.convert_words_to_ids(text))
        self.texts = process_text

    def convert_ids_to_words(self, input_ids:List[int]) -> str:
        text = [self.id2word[id] for id in input_ids]
        return ''.join(text)

    def convert_words_to_ids(self, words:List[str]) -> List[int]:
        text = [self.word2id[word] for word in words]
        return text

    def __len__(self):
        return len(self.texts)

    # critical：提供输入和标签
    # 思考1：target为什么取输入sent的[1:]，0又代表了什么？
    # 思考2：当BATCH_SIZE>1时，多个输入的句子可能不等长，
    # 不等长的句子forward会报错，应该怎么处理？
    def __getitem__(self, idx):
        sent = self.texts[idx]
        target = sent[1:] + [0]
        sent = torch.tensor(sent)
        target = torch.tensor(target)
        return sent, target    

def predict(model, dataloader, data_lines, sentences):
    model.eval()
    with torch.no_grad():
        hidden = model.init_hidden(BATCH_SIZE, requires_grad = False)
        for i, batch in enumerate(dataloader):
            data, target = batch
            
            hidden = repackage_hidden(hidden)
            with torch.no_grad():
                output, hidden = model(data, hidden)
            sentence_probability = 1
            prob = nnf.softmax(output, dim=1)
            for word in data_lines[i]:
                print(word, end='')
                sentence_probability *= prob.numpy()[0][i][word2id[word]]
            print("Sentence: ", sentences[i], "probability = ", sentence_probability)            
 
    
def repackage_hidden(h):
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)
    
class RecurrentNeuralNet(nn.Module):
    def __init__(self, ntoken, ninp, nhid, nonlinearity = 'tanh', dropout = 0.5):
        super(RecurrentNeuralNet, self).__init__()

        # The distribution is Bernoilli distribution, every neurons have probability of p once
        self.drop = nn.Dropout(dropout)

        # note  embedding is the first layer of language model, it already converts input into one hot vectors such that the computer can understand
        self.encoder = nn.Embedding(ntoken, ninp)

        self.rnn = nn.RNN(ninp, nhid, nonlinearity = nonlinearity, dropout = dropout, batch_first = True)

                     
        self.decoder = nn.Linear(nhid, ntoken)
        self.init_weights()
        self.nhid = nhid
        self.vocab_size = ntoken
    def init_weights(self):
        self.encoder.weight.data.uniform_(-0.5, 0.5)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-0.5, 0.5)
    # define forward propagation function
    def forward(self, inputs, hidden):
        embedding = self.drop(self.encoder(inputs))
        # output is each output after input, hidden is last output
        output, hidden = self.rnn(embedding)
        decoded = self.decoder(output.view(output.size(0) * output.size(1), output.size(2)))
        return decoded.view(output.size(0), output.size(1), decoded.size(1)), hidden

    def init_hidden(self, bsz, requires_grad = True):
        weight = next(self.parameters())
        return weight.new_zeros((1,bsz, self.nhid), requires_grad = requires_grad)



def main():
    # preprocessing data
    with open("train_en.txt", "r") as file:
        content = file.read()
    sentences = sent_tokenize(content)
    for sent in sentences:
        split_sent = word_tokenize(sent)
        lines.append(split_sent)
        for word in split_sent:
            words.add(word)

    with open("eval_en.txt", "r") as file:
        eval_content = file.read()
    e_lines = []
    e_sentences = sent_tokenize(eval_content)
    for sentence in e_sentences:
        e_split_sentence = word_tokenize(sentence)
        e_lines.append(e_split_sentence)
        for word in e_split_sentence:
            words.add(word)

    with open("test_en.txt", "r") as file:
        test_content = file.read()
    test_lines = []
    test_sentences = sent_tokenize(test_content)

    for sentence in test_sentences:
        test_split_sentence = word_tokenize(sentence)
        test_lines.append(test_split_sentence)
        for word in test_split_sentence:
            words.add(word)


    # note that here we first add all the words from the three dataset
    # into words dictionary so that we will be able to index them
    for i, word in enumerate(list(words)):
        word2id[word] = i+1
        id2word[i+1] = word

    
    VOCAB_SIZE = len(word2id)

    # define dataloader for training, evaluationg, and prediction
    news_ds = Article_Dataset(word2id, id2word, lines)
    train_dataloader = DataLoader(news_ds)

    eval_ds = Article_Dataset(word2id, id2word, e_lines)
    eval_dataloader = DataLoader(eval_ds)

    test_ds = Article_Dataset(word2id, id2word, test_lines)
    test_dataloader = DataLoader(test_ds)

    # model parameters
    nhid = 300
    model = RecurrentNeuralNet(VOCAB_SIZE, EMBEDDING_SIZE, nhid, dropout = 0.5)

    # training parameters
    learningRate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr = learningRate)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    evaluation_losses = []    
    # training and evaluation
    for epoch in range(100):
        model.train()
        hidden = model.init_hidden(BATCH_SIZE)

        for i, batch in enumerate(train_dataloader):
            data, target = batch

            hidden = repackage_hidden(hidden)
            model.zero_grad()
            # in every step, we clear backward from the last step,
            # such that gradient information is accurate 
            output, hidden = model(data, hidden)
            loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
            # updating parameters 
            loss.backward()

            optimizer.step()
        if (epoch +1) %5 == 0:
            print("epoch ", epoch, " iteration: ", i, "loss = ", loss.item())
        if(epoch + 1) % 10 == 0:
            # evaluation
            model.eval()
            total_loss = 0
            total_count = 0

            # 不是训练，关闭梯度加快运行速度
            with torch.no_grad():
                hidden = model.init_hidden(BATCH_SIZE, requires_grad = False)
                # 将数据按batch输入
                for i, batch in enumerate(eval_dataloader):
                    data, target = batch
                    hidden = repackage_hidden(hidden)

                    with torch.no_grad():
                        output, hidden = model(data, hidden)                            

                    # Calculuate Loss 
                    evaluation_loss = loss_fn(output.view(-1, VOCAB_SIZE), target.view(-1))
                    total_count += np.multiply(*data.size())

                    total_loss += loss.item() * np.multiply(*data.size())

                evaluation_loss = total_loss / total_count
                model.train()        
            if len(evaluation_losses) == 0 or evaluation_loss < min(evaluation_losses):
                print("Best model, epoch: ", epoch, "loss: ", evaluation_loss)
                # save best model, we can then use torch.load() to read saved parameters
                torch.save(model.state_dict(), "lm-best.th")
            evaluation_losses.append(evaluation_loss)
    # prediction (or testing)
    predict(model, test_dataloader, test_lines, test_sentences)

if __name__ == "__main__":
    main()
