import random
from torch import optim, nn
from config import *
from data import *
from seq2seq_model import *
import jieba
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import corpus_bleu
from config import *
from train_eval import Lang, tensorFromSentence_cn


input_lang = torch.load("C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/test_data/input_chinese.lang")
output_lang = torch.load("C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/test_data/output_eng.lang")
test_pairs = torch.load("C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/test_data/test_pair.lang")

best_encoder_model = torch.load("C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/model/best_encoder_model.pth", map_location = device)
best_decoder_model = torch.load("C:/Users/user/OneDrive/Desktop/jim_neural_network/final_project/model/best_decoder_model.pth", map_location = device)
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words)
encoder.load_state_dict(best_encoder_model)
encoder.to(device)
decoder.load_state_dict(best_decoder_model)
decoder.to(device)
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


def evaluate_test_dataset_bleu_score():
	encoder.eval()
	decoder.eval()
	list_of_references = [ [list(word_tokenize(target_tensor))] for (input_tensor, target_tensor) in test_pairs]	
	list_of_hypotheses = []

	for i in range(len(test_pairs)):
		decoded_words, _ = evaluate(encoder, decoder, test_pairs[i][0])

		print(list_of_references[i])
		print(decoded_words)
		print()
		list_of_hypotheses.append(decoded_words)
		#print(input_sentence, ' ', decoded_words)
	bleu_score = corpus_bleu(list_of_references, list_of_hypotheses)
	print("Bleu score of test dataset = ", bleu_score)


# calculate bleu score for test dataset
evaluate_test_dataset_bleu_score()