from __future__ import unicode_literals, print_function, division
from seq2seq_model import *
from io import open
import unicodedata
import re
import random
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import jieba
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from config import *



class Lang:
    def __init__(self,name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1:"EOS"}
        self.n_words = 2
        self.num_sentences = 0
    def addSentence(self, sentence):
        self.num_sentences = self.num_sentences + 1
        for word in list(word_tokenize(sentence)):
            self.addWord(word)
    def addSentence_cn(self, sentence):
        self.num_sentences = self.num_sentences + 1        
        for word in list(jieba.cut(sentence)):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


def readLines(datatype):
    print("reading lines for %s..."%datatype)
    output_lang_lines = open('C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/data/%s_en.txt'% datatype, encoding = 'utf-8').read().strip().split('\n')
    input_lang_lines = open('C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/data/%s_ch.txt' %datatype, encoding = 'utf-8').read().strip().split('\n')
    pairs = []
    for i in range(len(input_lang_lines)):
        pairs.append([input_lang_lines[i], output_lang_lines[i]])
    return input_lang_lines, output_lang_lines, pairs

def filterPair(p):
    return len(list(jieba.cut(p[0]))) < MAX_LENGTH and len(list(word_tokenize(p[1]))) < MAX_LENGTH

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

                                                              
def prepare_data():
    input_lang_lines = []
    output_lang_lines = []
    pairs = []

    train_input_lang_lines, train_output_lang_lines, train_pairs = readLines('train')
    print("Read %s sentence pairs for train" % len(train_pairs))
    train_pairs = filterPairs(train_pairs)
    for i in range(len(train_pairs)):
        input_lang_lines.append(train_pairs[i][0])
        output_lang_lines.append(train_pairs[i][1])
    print("num of sentences for train dataloader: %s" % len(train_pairs)) 
    val_input_lang_lines, val_output_lang_lines, val_pairs = readLines('val')
    print("Read %s sentence pairs for val" % len(val_pairs))
    val_pairs = filterPairs(val_pairs)
    for i in range(len(val_pairs)):
        input_lang_lines.append(val_pairs[i][0])
        output_lang_lines.append(val_pairs[i][1])
    print("num of sentences for val dataloader: %s" % len(val_pairs))     
    test_input_lang_lines, test_output_lang_lines, test_pairs = readLines('test')
    print("Read %s sentence pairs for test" % len(test_pairs))
    test_pairs = filterPairs(test_pairs)
    for i in range(len(test_pairs)):
        input_lang_lines.append(test_pairs[i][0])
        output_lang_lines.append(test_pairs[i][1])    
    print("num of sentences for test dataloader: %s" % len(test_pairs))       
    print("Counting words...")
    input_lang = Lang(input_lang_lines)
    output_lang = Lang(output_lang_lines)

    for pair in train_pairs:
        input_lang.addSentence_cn(pair[0])
        output_lang.addSentence(pair[1])
    for pair in val_pairs:
        input_lang.addSentence_cn(pair[0])
        output_lang.addSentence(pair[1])
    for pair in test_pairs:
        input_lang.addSentence_cn(pair[0])
        output_lang.addSentence(pair[1])    
    print("Counted words: ")
    print(input_lang.n_words)
    print(output_lang.n_words)
    return input_lang, output_lang, train_pairs, val_pairs, test_pairs




def indexesFromSentence_cn(lang, sentence):
    return [lang.word2index[word] for word in list(jieba.cut(sentence))]
def tensorFromSentence_cn(lang, sentence):
    indexes = indexesFromSentence_cn(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in list(word_tokenize(sentence))]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence_cn(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(pairs, batch_size):

    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence_cn(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return train_dataloader



def train_epoch(dataloader, encoder, decoder, encoder_optimizer, 
          decoder_optimizer, criterion):
    
    total_loss = 0
    for data in dataloader:
        input_tensor, target_tensor = data

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, _, _ = decoder(encoder_outputs, encoder_hidden, target_tensor)
        
        loss = criterion(
            decoder_outputs.view(-1, decoder_outputs.size(-1)),
            target_tensor.view(-1)
        )
        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()

        total_loss += loss.item()
    
    return total_loss / len(dataloader)

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


def evaluate(encoder, decoder, sentence):
    with torch.no_grad():
        input_tensor = tensorFromSentence_cn(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(encoder_outputs, encoder_hidden)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()
        
        decoded_words = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                break
            decoded_words.append(output_lang.index2word[idx.item()])
    return decoded_words, decoder_attn




def train_and_eval(train_dataloader, encoder, decoder, n_epochs, 
               print_every=100, plot_every=100, eval_every=100):
    list_of_references = [ [list(word_tokenize(target_tensor))] for (input_tensor, target_tensor) in val_pairs]
    evaluation_scores = []
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    
    #encoder_optimizer = optim.Adam(encoder.parameters(), lr=ADAM_LR)
    #decoder_optimizer = optim.Adam(decoder.parameters(), lr=ADAM_LR)
    
    encoder_optimizer = torch.optim.SGD(encoder.parameters(), lr=SGD_LR, momentum=MOMENTUM)
    decoder_optimizer = torch.optim.SGD(decoder.parameters(), lr=SGD_LR, momentum=MOMENTUM)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        encoder.train()
        decoder.train()
        
        loss = train_epoch(train_dataloader, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
        print('training epoch: ', epoch,' loss = ', loss)
        print_loss_total += loss
        plot_loss_total += loss
        
        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (Epoch: %d %d%%) average loss = %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

        if epoch % eval_every == 0:
            encoder.eval()
            decoder.eval()
            list_of_hypotheses = []
            for input_sentence, _ in val_pairs:     
                #print(input_sentence)           
                decoded_words, _ = evaluate(encoder, decoder, input_sentence)
                #print(decoded_words)
                list_of_hypotheses.append(decoded_words)
            bleu_score = corpus_bleu(list_of_references, list_of_hypotheses)

            print('Evaluating data: iter = ', epoch,' bleu score = ',bleu_score)
            if(len(evaluation_scores) == 0 or max(evaluation_scores) < bleu_score):
                print('Save best model from evaluation: ')
                evaluation_scores.append(bleu_score)
                # save best model
                torch.save(encoder.state_dict(), 'C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/model/best_encoder_model.pth') 
                torch.save(decoder.state_dict(), 'C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/model/best_decoder_model.pth')               


if __name__ == '__main__':
    input_lang, output_lang, train_pairs, val_pairs, test_pairs = prepare_data()
    torch.save(output_lang, "C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/test_data/output_eng.lang")
    torch.save(input_lang, "C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/test_data/input_chinese.lang")
    torch.save(test_pairs, "C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/test_data/test_pair.lang")
    print("finished saving test dataloader")
    train_dataloader = get_dataloader(train_pairs,batch_size)

    encoder = EncoderRNN(input_lang.n_words, hidden_size).to(device)
    decoder = AttnDecoderRNN(hidden_size
        , output_lang.n_words).to(device)

    train_and_eval(train_dataloader, encoder, decoder, 100, print_every=5, plot_every=5, eval_every = 5)
