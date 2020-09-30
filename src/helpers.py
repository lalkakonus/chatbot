import itertools
import json
import torch

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


with open("../config.json", "r") as f:
    config = json.load(f)

MAX_SENTENCE_LENGTH = config["MAX_SENTENCE_LENGTH"]  # Maximum sentence length to consider
MIN_WORD_CNT = config["MIN_WORD_CNT"]    # Minimum word count threshold for trimming


def indexesFromSentence(vocabulary, sentence):
    return [vocabulary.word2index[word] for word in sentence.split()] + [EOS_token]


def zeroPadding(seqs, fillvalue=PAD_token):
    return list(itertools.zip_longest(*seqs, fillvalue=fillvalue))


# Returns padded input sequence tensor and lengths
def inputVar(seqs, vocabulary):
    indexes_batch = [indexesFromSentence(vocabulary, sentence) for sentence in seqs]
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    return padVar, lengths

# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(seqs, vocabulary):
    indexes_batch = [indexesFromSentence(vocabulary, sentence) for sentence in seqs]
    max_target_len = max([len(indexes) for indexes in indexes_batch])
    padList = zeroPadding(indexes_batch)
    padVar = torch.LongTensor(padList)
    mask = torch.BoolTensor(padVar != PAD_token)
    return padVar, mask, max_target_len


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()