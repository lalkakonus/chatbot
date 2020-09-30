from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import json
import os
import random

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from dataset import ConversationsDataset
from nn_modules import EncoderRNN, DecoderRNN, GreedySearch
from helpers import SOS_token, maskNLLLoss, device, indexesFromSentence, MAX_SENTENCE_LENGTH


class ChatModel:

    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.epoch = 1
        self.train_loss = []
        self.checkpoint_dir = "../data/checkpoints"

        with open("../config.json", "r") as f:
            config = json.load(f)

        self.model_name = config["MODEL_NAME"]
        self.attn_model = config["ATTN_MODEL"]
        self.hidden_size = config["HIDDEN_SIZE"]
        self.encoder_n_layers = config["ENCODER_N_LAYERS"]
        self.decoder_n_layers = config["DECODER_N_LAYERS"]
        self.dropout = config["DROPOUT"]
        self.batch_size = config["BATCH_SIZE"]
        self.n_epoch = config["N_EPOCH"]
        self.clip = config["CLIP"]
        self.teacher_forcing_ratio = config["TEACHER_FORCING_RATIO"]
        self.learning_rate = config["LEARNING_RATE"]
        self.decoder_learning_ratio = config["DECODER_LEARNING_RATIO"]
        self.save_every = config["SAVE_EVERY"]

        # Initialize word embeddings
        self.embedding = nn.Embedding(len(self.vocabulary), self.hidden_size)
        self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
        self.decoder = DecoderRNN(self.attn_model, self.embedding, self.hidden_size, len(self.vocabulary),
                                  self.decoder_n_layers, self.dropout)

        self.searcher = GreedySearch(self.encoder, self.decoder)
        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=self.learning_rate * self.decoder_learning_ratio)

        print('Models built and ready to go!')

    def __configure_cuda__(self):
        for state in self.encoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

        for state in self.decoder_optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    def __train_iteration__(self, input_variable, lengths, target_variable, mask, max_target_len, batch_size):

        # Zero gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Initialize variables
        loss = 0
        print_losses = []
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        # Determine if we are using teacher forcing this iteration
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        # Forward batch of sequences through decoder one time step at a time
        if use_teacher_forcing:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # Teacher forcing: next input is current target
                decoder_input = target_variable[t].view(1, -1)
                # Calculate and accumulate loss
                mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total
        else:
            for t in range(max_target_len):
                decoder_output, decoder_hidden = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs
                )
                # No teacher forcing: next input is decoder's own current output
                _, topi = decoder_output.topk(1)
                decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
                decoder_input = decoder_input.to(device)
                # Calculate and accumulate loss
                mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
                loss += mask_loss
                print_losses.append(mask_loss.item() * n_total)
                n_totals += n_total

        # Perform backpropatation
        loss.backward()

        # Clip gradients: gradients are modified in place
        _ = torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), self.clip)
        _ = torch.nn.utils.clip_grad_norm_(self.decoder.parameters(), self.clip)

        # Adjust model weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return sum(print_losses) / n_totals

    def train(self, dataloader):
        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()

        # Run training iterations
        print("Starting Training!")

        # training_batches = [batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
        #                     for _ in range(self.n_iteration)]

        # Initializations
        print('Initializing ...')
        print_loss = 0

        # Training loop
        print_every = 1
        print("Training...")
        for epoch in range(self.epoch, self.n_epoch + 1):
            for train_batch in dataloader:
                # Extract fields from batch
                input_variable, lengths, target_variable, mask, max_target_len = train_batch

                # Run a training iteration with batch
                loss = self.__train_iteration__(input_variable, lengths, target_variable, mask, max_target_len,
                                                dataloader.batch_size)
                print_loss += loss

            # Print progress
            print_loss_avg = print_loss / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Average loss: {:.4f}".format(epoch,
                                                                                          epoch / self.n_epoch * 100,
                                                                                          print_loss_avg))
            print_loss = 0

            # Save checkpoint
            # if epoch % self.save_every == 0:
            #     self.save(epoch, print_loss)

    def evaluate(self, sentence):
        self.encoder.eval()
        self.decoder.eval()

        indexes_batch = [indexesFromSentence(self.vocabulary, sentence)]
        lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
        input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)

        input_batch = input_batch.to(device)
        lengths = lengths.to(device)

        tokens, scores = self.searcher(input_batch, lengths, MAX_SENTENCE_LENGTH)
        decoded_words = [self.vocabulary.index2word[token.item()] for token in tokens]
        return decoded_words

    def load(self, epoch=100):
        filename = '{}-{}_{}_{}_checkpoint.tar'.format(self.encoder_n_layers, self.decoder_n_layers,
                                                       self.hidden_size, self.epoch)

        filepath = os.path.join(self.checkpoint_dir, filename)

        # If loading on same machine the model was trained on
        checkpoint = torch.load(filepath)
        self.epoch = checkpoint['epoch']
        self.train_loss = checkpoint["train_loss"]

        encoder_sd = checkpoint['en']
        decoder_sd = checkpoint['de']
        encoder_optimizer_sd = checkpoint['en_opt']
        decoder_optimizer_sd = checkpoint['de_opt']
        embedding_sd = checkpoint['embedding']

        # Initialize word embeddings
        self.embedding.load_state_dict(embedding_sd)
        self.encoder.load_state_dict(encoder_sd)
        self.decoder.load_state_dict(decoder_sd)

        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
        self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)
        return self

    def save(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        filename = '{}-{}_{}_{}_checkpoint.tar'.format(self.encoder_n_layers, self.decoder_n_layers,
                                                       self.hidden_size, self.epoch)
        torch.save({
            'epoch': self.epoch,
            'en': self.encoder.state_dict(),
            'de': self.decoder.state_dict(),
            'en_opt': self.encoder_optimizer.state_dict(),
            'de_opt': self.decoder_optimizer.state_dict(),
            'train_loss': self.train_loss,
            'embedding': self.embedding.state_dict()
        }, os.path.join(self.checkpoint_dir, filename))
        return self


if __name__ == "__main__":
    dataset = ConversationsDataset()
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=dataset.batch_to_train_data,
                            drop_last=True, shuffle=True)
    model = ChatModel(dataset.vocabulary)
    # model.train(dataloader)
    print(model.evaluate("hello dear !"))
