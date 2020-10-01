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
from torch.utils.data import DataLoader, random_split

from dataset import ConversationsDataset
from nn_modules import EncoderRNN, DecoderRNN, GreedySearch
from helpers import SOS_token, maskNLLLoss, device, indexesFromSentence, MAX_SENTENCE_LENGTH


class ChatModel:

    def __init__(self, config):
        self.checkpoint_dir = "../data/checkpoints"

        self.epoch = config.get("EPOCH", 1)
        self.train_loss = config.get("TRAIN_LOSS", [])
        self.model_name = config["MODEL_NAME"]
        self.vocabulary = config["VOCABULARY"]
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

        self.embedding = nn.Embedding.from_pretrained(self.vocabulary.embeddings)
        # self.embedding = nn.Embedding(len(self.vocabulary), self.hidden_size)
        self.encoder = EncoderRNN(self.hidden_size, self.embedding, self.encoder_n_layers, self.dropout)
        self.decoder = DecoderRNN(self.attn_model, self.embedding, self.hidden_size, len(self.vocabulary),
                                  self.decoder_n_layers, self.dropout)

        if config.get("LOAD_WEIGHT", False):
            self.embedding.load_state_dict(config["EMBEDDING"])
            self.encoder.load_state_dict(config["ENCODER"])
            self.decoder.load_state_dict(config["DECODER"])

        # Use appropriate device
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=self.learning_rate)
        self.decoder_optimizer = optim.Adam(self.decoder.parameters(),
                                            lr=self.learning_rate * self.decoder_learning_ratio)

        if config.get("LOAD_WEIGHT", False):
            self.encoder_optimizer.load_state_dict(config["ENCODER_OPTIMIZER"])
            self.decoder_optimizer.load_state_dict(config["DECODER_OPTIMIZER"])

        self.searcher = GreedySearch(self.encoder, self.decoder)

        print('Model initialized')

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
        # Ensure dropout layers are in train mode
        self.encoder.train()
        self.decoder.train()

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

    def __test_iteration__(self, input_variable, lengths, target_variable, mask, max_target_len, batch_size):
        # Ensure dropout layers are in eval mode
        self.encoder.eval()
        self.decoder.eval()

        # Set device options
        input_variable = input_variable.to(device)
        lengths = lengths.to(device)
        target_variable = target_variable.to(device)
        mask = mask.to(device)

        # Initialize variables
        loss = 0
        n_totals = 0

        # Forward pass through encoder
        encoder_outputs, encoder_hidden = self.encoder(input_variable, lengths)

        # Create initial decoder input (start with SOS tokens for each sentence)
        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch_size)]])
        decoder_input = decoder_input.to(device)

        # Set initial decoder hidden state to the encoder's final hidden state
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]

        for t in range(max_target_len):
            decoder_output, decoder_hidden = self.decoder(decoder_input, decoder_hidden, encoder_outputs)

            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, n_total = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss.item() * n_total
            n_totals += n_total

        return loss / n_totals


    def train(self, train_dataloader, test_dataloader):
        # Training loop
        print("Training...")
        for self.epoch in range(self.epoch, self.n_epoch + 1):
            train_loss = 0
            test_loss = 0
            for train_batch in train_dataloader:
                # Extract fields from batch
                input_variable, lengths, target_variable, mask, max_target_len = train_batch

                # Run a training iteration with batch
                train_loss += self.__train_iteration__(input_variable, lengths, target_variable, mask, max_target_len,
                                                       train_dataloader.batch_size)
            train_loss /= len(train_dataloader)
            for test_batch in test_dataloader:
                input_variable, lengths, target_variable, mask, max_target_len = test_batch
                # Run a training iteration with batch
                test_loss += self.__test_iteration__(input_variable, lengths, target_variable, mask, max_target_len,
                                                     train_dataloader.batch_size)
            test_loss /= len(test_dataloader)

            # Print progress
            self.train_loss.append(train_loss)
            print("Iteration: {}; Percent complete: {:.1f}%; "
                  "Average train loss: {:.4f}; test loss: {:.4f}".format(self.epoch, self.epoch / self.n_epoch * 100,
                                                                         train_loss, test_loss))
            # Save checkpoint
            if self.epoch % self.save_every == 0:
                self.save()



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

    @classmethod
    def load(cls, checkpoint_dir, encoder_n_layers, decoder_n_layers, hidden_size, epoch):
        filename = '{}-{}_{}_{}_checkpoint.tar'.format(encoder_n_layers, decoder_n_layers, hidden_size, epoch)
        checkpoint = torch.load(os.path.join(checkpoint_dir, filename))
        return cls(checkpoint)

    def save(self):
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
        filename = '{}-{}_{}_{}_checkpoint.tar'.format(self.encoder_n_layers, self.decoder_n_layers,
                                                       self.hidden_size, self.epoch)
        torch.save({
            "LOAD_WEIGHT": True,
            "EPOCH": self.epoch,
            "TRAIN_LOSS": self.train_loss,
            "MODEL_NAME": self.model_name,
            "VOCABULARY": self.vocabulary,
            "ATTN_MODEL": self.attn_model,
            "HIDDEN_SIZE": self.hidden_size,
            "ENCODER_N_LAYERS": self.encoder_n_layers,
            "DECODER_N_LAYERS": self.decoder_n_layers,
            "DROPOUT": self.dropout,
            "BATCH_SIZE": self.batch_size,
            "N_EPOCH": self.n_epoch,
            "CLIP": self.clip,
            "TEACHER_FORCING_RATIO": self.teacher_forcing_ratio,
            "LEARNING_RATE": self.learning_rate,
            "DECODER_LEARNING_RATIO": self.decoder_learning_ratio,
            "SAVE_EVERY": self.save_every,
            "ENCODER": self.encoder.state_dict(),
            "DECODER": self.decoder.state_dict(),
            "EMBEDDING": self.embedding.state_dict(),
            "ENCODER_OPTIMIZER": self.encoder_optimizer.state_dict(),
            "DECODER_OPTIMIZER": self.decoder_optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, filename))
        return self


if __name__ == "__main__":
    dataset = ConversationsDataset(build=True)
    with open("../config.json", "r") as f:
        config = json.load(f)
    config["VOCABULARY"] = dataset.vocabulary

    train_ration = 0.8
    size_list = [round(len(dataset) * train_ration), round(len(dataset) * (1 - train_ration))]
    train_dataset, test_dataset = random_split(dataset, size_list, generator=torch.Generator().manual_seed(42))

    train_dataloader = DataLoader(train_dataset, batch_size=config["BATCH_SIZE"],
                                  collate_fn=dataset.batch_to_train_data,
                                  drop_last=True, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=config["BATCH_SIZE"],
                                 collate_fn=dataset.batch_to_train_data,
                                 drop_last=True, shuffle=True)

    model = ChatModel(config)
    model.train(train_dataloader, test_dataloader)
    # model.save()
    # model = ChatModel.load("../data/checkpoints/", 2, 2, 500, 10)
    # print(model.evaluate("hello dear !"))
