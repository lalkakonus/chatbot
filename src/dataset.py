import os
import pickle
from torch.utils.data import Dataset, DataLoader

from helpers import MIN_WORD_CNT, inputVar, outputVar
from prepare_data import loadPreprocessedData, buildVocabulary, trimRareWords, Vocabulary


class ConversationsDataset(Dataset):
    def __init__(self, corpus_name="cornell movie-dialogs corpus", build=False):

        corpus = os.path.join("../data", corpus_name)
        datafile = os.path.join(corpus, "formatted_movie_lines.txt")
        dataset_filepath = os.path.join(corpus, "dataset.pkl")

        self.vocabulary = Vocabulary(corpus_name)
        self.pairs = []

        if os.path.exists(dataset_filepath) and build is False:
            with open(dataset_filepath, "rb") as f:
                self.vocabulary, self.pairs = pickle.load(f)
        else:
            pairs = loadPreprocessedData(corpus_name, datafile)
            # pairs = pairs[:1000]
            vocabulary = buildVocabulary(corpus_name, pairs)
            self.vocabulary, self.pairs = trimRareWords(vocabulary, pairs, MIN_WORD_CNT)
            self.vocabulary.init_embedding("../data/embedding/wiki-news-300d-1M.vec")

            with open(dataset_filepath, "wb") as f:
                pickle.dump([self.vocabulary, self.pairs], f)

    def batch_to_train_data(self, pairs_batch):
        pairs_batch.sort(key=lambda x: len(x[0].split()), reverse=True)
        input_batch, output_batch = [], []
        for pair in pairs_batch:
            input_batch.append(pair[0])
            output_batch.append(pair[1])
        inp, lengths = inputVar(input_batch, self.vocabulary)
        output, mask, max_target_len = outputVar(output_batch, self.vocabulary)
        return inp, lengths, output, mask, max_target_len

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        return self.pairs[index]


if __name__ == "__main__":
    dataset = ConversationsDataset()
    print(dataset[1])
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=dataset.batch_to_train_data)
    print(len(dataloader))
    # for batch in dataloader:
    #     print(batch)
    #     break
