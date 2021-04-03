# -*- coding: utf-8 -*-
from json import decoder
import math
import numpy as np
import operator
from queue import PriorityQueue

import torch
import torch.nn as nn
from torch.nn import functional as F

import pytorch_lightning as pl
from torch.optim.lr_scheduler import LambdaLR, ReduceLROnPlateau
from lib.constant import IDs, Tokens

from allennlp.nn.beam_search import BeamSearch

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        """[summary]

        Args:
            d_model ([type]): [description]
            dropout (float, optional): [description]. Defaults to 0.1.
            max_len (int, optional): [description]. Defaults to 5000.
        """
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.scale * self.pe[:x.size(0), :]
        return self.dropout(x)

class BeamSearchNode(object):
    """ Reference: https://github.com/312shan/Pytorch-seq2seq-Beam-Search
    """
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.h = hiddenstate
        self.prevNode = previousNode
        self.wordid = wordId
        self.logp = logProb
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward
        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward 

    def __lt__(self, other):
        return self.leng < other.leng 

    def __gt__(self, other):
        return self.leng > other.leng

class MLPEncoder(nn.Module):
    # 数値用のEncoder
    def __init__(self, input_size, hidden_size, dropout_p):
        super(MLPEncoder, self).__init__()
        self.linear = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(input_size, hidden_size),
            nn.Tanh())
    def __call__(self, x):
        return self.linear(x)

class WeatherLabelClassifier(nn.Module):
    def __init__(self, d_model, num_label, dropout=0.2):
        super(WeatherLabelClassifier, self).__init__()
        """Binary Classifer for Weather Label

        Args:
            d_model (tensor): dimension size of model
            num_label (tensor): number of label types
        """        
        self.linear = nn.Linear(d_model, d_model)
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.output = nn.Linear(d_model, num_label)
    
    def forward(self, hidden):
        hidden = self.dropout(self.tanh(self.linear(hidden)))
        output = self.output(hidden)
        return output, hidden

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.normal_(mean=0, std=stdv)

    def forward(self, hidden, encoder_outputs, src_len=None):
        '''
        :param hidden: 
            previous hidden state of the decoder, in shape (layers*directions,B,H)
        :param encoder_outputs:
            encoder outputs from Encoder, in shape (T,B,H)
        :param src_len:
            used for masking. NoneType or tensor in shape (B) indicating sequence length
        :return
            attention energies in shape (B,T)
        '''
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        H = hidden.repeat(max_len,1,1).transpose(0,1)
        encoder_outputs = encoder_outputs.transpose(0,1) # [B*T*H]
        attn_energies = self.score(H,encoder_outputs) # compute attention score
        
        if src_len is not None:
            mask = []
            for b in range(src_len.size(0)):
                mask.append([0] * src_len[b].item() + [1] * (encoder_outputs.size(1) - src_len[b].item()))
            mask = cuda_(torch.ByteTensor(mask).unsqueeze(1)) # [B,1,T]
            attn_energies = attn_energies.masked_fill(mask, -1e18)
        return F.softmax(attn_energies, dim=1).unsqueeze(1) # normalize with softmax

    def score(self, hidden, encoder_outputs):
        energy = torch.tanh(self.attn(torch.cat([hidden, encoder_outputs], 2))) # [B*T*2H]->[B*T*H]
        energy = energy.transpose(2,1) # [B*H*T]
        v = self.v.repeat(encoder_outputs.data.shape[0],1).unsqueeze(1) #[B*1*H]
        energy = torch.bmm(v,energy) # [B*1*T]
        return energy.squeeze(1) #[B*T]

class TokenAttnDecoderRNN(nn.Module):
    """ https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq
    """
    def __init__(self, hidden_size, meta_amedas_hidden_size, embed_size, output_size, n_layers=1, weather=None, dropout_p=0.2):
        super(TokenAttnDecoderRNN, self).__init__()
        # Define parameters
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.weather = weather
        self.dropout_p = dropout_p
        # Define layers
        self.embedding = nn.Embedding(output_size, embed_size, padding_idx=IDs.PAD.value)
        self.dropout = nn.Dropout(dropout_p)
        self.relu = nn.ReLU()
        
        self.gpv_attn = Attn('concat', hidden_size)
        self.amedas_linear = nn.Linear(hidden_size, meta_amedas_hidden_size)
        self.amedas_attn = Attn('concat', meta_amedas_hidden_size)
        self.meta_linear = nn.Linear(hidden_size, meta_amedas_hidden_size)
        self.meta_attn = Attn('concat', meta_amedas_hidden_size)

        if weather is None:
            self.attn_combine = nn.Linear(embed_size + hidden_size + meta_amedas_hidden_size * 2, hidden_size)
        else:
            self.weather_attn = Attn('concat', hidden_size)
            self.attn_combine = nn.Linear(embed_size + hidden_size * 2 + meta_amedas_hidden_size * 2, hidden_size)
            
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=dropout_p)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, word_input, last_hidden, 
        enc_gpv_outputs, enc_amedas_outputs, enc_meta_outputs, enc_weather_outputs):
        '''
        :param word_input:
            word input for current time step, in shape (B)
        :param last_hidden:
            last hidden stat of the decoder, in shape (layers*direction*B*H)
        :param encoder_outputs:
            encoder outputs in shape (T*B*H)
        :return
            decoder output
        Note: we run this one step at a time i.e. you should use a outer loop 
            to process the whole sequence
        Tip(update):
        EncoderRNN may be bidirectional or have multiple layers, so the shape of hidden states can be 
        different from that of DecoderRNN
        You may have to manually guarantee that they have the same dimension outside this function,
        e.g, select the encoder hidden state of the foward/backward pass.
        '''
        # Get the embedding of the current input word (last output word)
        word_embedded = self.embedding(word_input).view(1, word_input.size(0), -1) # (1,B,V)
        word_embedded = self.dropout(word_embedded)

        # Calculate attention weights and apply to encoder outputs
        gpv_attn_weights = self.gpv_attn(last_hidden[-1], enc_gpv_outputs)
        gpv_context = gpv_attn_weights.bmm(enc_gpv_outputs.transpose(0, 1))  # (B,1,V)
        gpv_context = gpv_context.transpose(0, 1)  # (1,B,V)
        
        last_hidden_for_amedas = self.amedas_linear(last_hidden[-1])
        amedas_attn_weights = self.amedas_attn(last_hidden_for_amedas, enc_amedas_outputs)
        amedas_context = amedas_attn_weights.bmm(enc_amedas_outputs.transpose(0, 1))  # (B,1,V)
        amedas_context = amedas_context.transpose(0, 1)  # (1,B,V)

        last_hidden_for_meta = self.meta_linear(last_hidden[-1])
        meta_attn_weights = self.meta_attn(last_hidden_for_meta, enc_meta_outputs)
        meta_context = meta_attn_weights.bmm(enc_meta_outputs.transpose(0, 1))  # (B,1,V)
        meta_context = meta_context.transpose(0, 1)  # (1,B,V)

        # introduce weather_context into context vector
        if self.weather is not None:
            weather_attn_weights = self.weather_attn(last_hidden[-1], enc_weather_outputs)
            weather_context = weather_attn_weights.bmm(enc_weather_outputs.transpose(0, 1))  # (B,1,V)
            weather_context = weather_context.transpose(0, 1)  # (1,B,V)
            context = torch.cat([gpv_context, amedas_context, meta_context, weather_context], dim=2)
        else:
            context = torch.cat([gpv_context, amedas_context, meta_context], dim=2)

        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat((word_embedded, context), 2)
        rnn_input = self.attn_combine(rnn_input) # use it in case your size of rnn_input is different
        rnn_input = self.relu(rnn_input) 
        output, hidden = self.gru(rnn_input, last_hidden.contiguous())
        output = output.squeeze(0)  # (1,B,V)->(B,V)
        # context = context.squeeze(0)
        # update: "context" input before final layer can be problematic.
        # output = F.log_softmax(self.out(torch.cat((output, context), 1)))
        output = F.log_softmax(self.out(output), dim=-1)
        # Return final output, hidden state
        return output, hidden, word_embedded

class WNCGBaseModel(pl.LightningModule):
    def __init__(self, d_model, weather, cl, dropout_p):
        super().__init__()
        self.weather = weather
        self.cl = cl

        # weather label classifier
        if weather == "label":
            self.sunny_decoder = WeatherLabelClassifier(d_model, 2, dropout_p)
            self.cloudy_decoder = WeatherLabelClassifier(d_model, 2, dropout_p)
            self.rain_decoder = WeatherLabelClassifier(d_model, 2, dropout_p)
            self.snow_decoder = WeatherLabelClassifier(d_model, 2, dropout_p)

    @staticmethod
    def get_data(batch, device):
        # input
        src_gpv = batch[0].to(device) # (batch_size, seq_len, num_gpv_types, lat_size, long_size)
        src_amedas = batch[1].to(device) # (batch_size, amedas_seq_len, num_amedas_types)
        src_meta = batch[2].to(device)  # (batch_size, num_meta_types)
        # output
        ref_label = batch[3].to(device) # (batch_size, num_label_types)
        ref_comment = batch[4].to(device) # (batch_size, seq_len)

        # reshape tensor
        src_gpv = src_gpv.view(src_gpv.size(0), src_gpv.size(1), -1).transpose(0, 1) # (seq_len, batch_size, d_model)
        src_amedas = src_amedas.transpose(1, 2).transpose(0, 1) # (num_amedas_types, batch_size, amedas_seq_len)
        src_meta = src_meta.transpose(0, 1) # (num_meta_types, batch_size)
        tgt_sunny_label, tgt_cloudy_label, tgt_rain_label, tgt_snow_label = \
            ref_label[:, 0], ref_label[:, 1], ref_label[:, 2], ref_label[:, 3]
        ref_comment = ref_comment.transpose(0, 1) # (seq_len, batch_size)
        src_comment = ref_comment[:-1, :]
        tgt_comment = ref_comment[1:, :] 
        return (src_gpv, src_amedas, src_meta, src_comment,
            tgt_sunny_label, tgt_cloudy_label, tgt_rain_label,
            tgt_snow_label, tgt_comment)

    def training_step(self, batch, batch_idx):
        # get data from batch
        src_gpv, src_amedas, src_meta, src_comment, \
        tgt_sunny_label, tgt_cloudy_label, tgt_rain_label, \
        tgt_snow_label, tgt_comment = self.get_data(batch, device=self.device)

        # forward pass
        # (seq_len, batch_size, vocab_size)
        token_out, sunny_out, cloudy_out, rain_out, snow_out, \
            weather_hidden, tgt_text_embed = self(src_gpv, src_amedas, src_meta, src_comment) 

        # calculate label loss
        if self.weather == "label":
            # calculate label loss
            sunny_loss = F.nll_loss(sunny_out, tgt_sunny_label) / 4
            cloudy_loss = F.nll_loss(cloudy_out, tgt_cloudy_label) / 4
            rain_loss = F.nll_loss(rain_out, tgt_rain_label) / 4
            snow_loss = F.nll_loss(snow_out, tgt_snow_label) / 4
            label_loss = sunny_loss + cloudy_loss + rain_loss + snow_loss
        else:
            sunny_loss, cloudy_loss, rain_loss, snow_loss, label_loss = 0, 0, 0, 0, 0

        # calculate content_agreement loss
        if self.cl:
            weather_hidden_mean = torch.mean(weather_hidden, dim=0)
            tgt_text_embed_mean = torch.mean(tgt_text_embed, dim=0)
            agreement_loss = F.mse_loss(weather_hidden_mean, tgt_text_embed_mean)
        else:
            agreement_loss = 0

        # calculate token loss
        token_out = token_out.view(-1, token_out.size(-1)) # (seq_len * batch_size, vocab_size)
        token_loss = F.nll_loss(token_out, tgt_comment.reshape(-1))        

        label_loss_weight = 0.2
        agreement_loss_weight = 0.2
        if self.training:
            self.log('token_loss', token_loss, prog_bar=True)
            self.log('label_loss', label_loss * label_loss_weight, prog_bar=True)
            self.log('label_sunny_loss', sunny_loss * label_loss_weight, prog_bar=False)
            self.log('label_cloudy_loss', cloudy_loss * label_loss_weight, prog_bar=False)
            self.log('label_rain_loss', rain_loss * label_loss_weight, prog_bar=False)
            self.log('label_snow_loss', snow_loss * label_loss_weight, prog_bar=False)
            self.log('agreement_loss', agreement_loss * agreement_loss_weight, prog_bar=True)
            self.log('lr', self.lr, prog_bar=True)
        return token_loss + (label_loss * label_loss_weight) + (agreement_loss * agreement_loss_weight)

    def validation_step(self, batch, batch_idx):
        val_loss = self.training_step(batch, batch_idx)
        self.log('val_loss', val_loss)
        return val_loss

    def predict_label(self, hidden):
        """ predict weather labels from hidden state of encoders

        Args:
            hidden ([type]): [description]

        Returns:
            [type]: [description]
        """
        sunny_out, sunny_hidden = self.sunny_decoder(hidden)
        cloudy_out, cloudy_hidden = self.cloudy_decoder(hidden)
        rain_out, rain_hidden = self.rain_decoder(hidden)
        snow_out, snow_hidden = self.snow_decoder(hidden)

        sunny_topv, sunny_topi = sunny_out.data.topk(1)
        cloudy_topv, cloudy_topi = cloudy_out.data.topk(1)
        rain_topv, rain_topi = rain_out.data.topk(1)
        snow_topv, snow_topi = snow_out.data.topk(1)

        sunny_topi = sunny_topi.squeeze(1).detach().tolist()
        cloudy_topi = cloudy_topi.squeeze(1).detach().tolist()
        rain_topi = rain_topi.squeeze(1).detach().tolist()
        snow_topi = snow_topi.squeeze(1).detach().tolist()
        # concatenate hidden states of weather label classifer
        weather_hidden = torch.stack([sunny_hidden, cloudy_hidden, rain_hidden, snow_hidden], dim=0)
        return sunny_topi, cloudy_topi, rain_topi, snow_topi, weather_hidden

class WNCGRnnModel(WNCGBaseModel):
    def __init__(self, input_gpv_dim, d_model, num_layers, n_vocab, 
        input_amedas_seqlen, weather, cl, lr=0.001, dropout_p=0.2):
        super().__init__(d_model, weather, cl, dropout_p)
        d_meta_amedas = 64
        self.num_layers = num_layers
        self.d_model = d_model

        """ Encoder """
        # encoder for gpv
        self.gpv_encoder = MLPEncoder(input_gpv_dim, d_model, dropout_p=dropout_p) # single-layer MLP for encoding a gpv data
        self.rnn_encoder = nn.GRU(d_model, d_model, num_layers=1, bidirectional=True) # single-layer BiGRU for encoding sequence of gpv data
        self.gpv_to_dmodel = nn.Linear(d_model * 2, d_model)
        # encoder for amedas
        self.amedas_to_dmodel = nn.Linear(input_amedas_seqlen, d_meta_amedas)
        # encoder for meta-data
        metaenc = {}
        metaenc["area"] = nn.Embedding(277, d_meta_amedas)
        metaenc["month"] = nn.Embedding(12, d_meta_amedas) 
        metaenc["day"] = nn.Embedding(31, d_meta_amedas)
        metaenc["time"] = nn.Embedding(24, d_meta_amedas)
        metaenc["week"] = nn.Embedding(7, d_meta_amedas)
        self.meta_encoders = nn.ModuleDict(metaenc)
        # BiGRU for gpv, Linear for Amedas, Linear for Metadata
        self.input_to_dmodel = nn.Linear((d_model * 2) + (d_meta_amedas * 4) + (d_meta_amedas * 5), d_model)
        self.relu = nn.ReLU()

        """ Decoder """
        # word decoder
        self.token_decoder = TokenAttnDecoderRNN(
            d_model, d_meta_amedas, d_model, n_vocab, num_layers, weather, dropout_p)
        # weather label
        self.weather = weather
        # option for content agreement loss
        self.cl = cl
        # make the arguments global
        self.lr = lr
        # save the arguments
        self.save_hyperparameters()

    def encode(self, src_gpv, src_amedas, src_meta, src_comment):
        """ encode 
        """
        _, batch_size = src_comment.size()
        # encode gpv-data
        src_gpv = self.gpv_encoder(src_gpv)
        gpv_output, gpv_hidden = self.rnn_encoder(src_gpv)
        gpv_output = self.gpv_to_dmodel(gpv_output)
        # encode amedas-data
        src_amedas = self.amedas_to_dmodel(src_amedas)
        # encode meta-data
        emb_area = self.meta_encoders["area"](src_meta[0, :])
        emb_month = self.meta_encoders["month"](src_meta[1, :])
        emb_day = self.meta_encoders["day"](src_meta[2, :])
        emb_time = self.meta_encoders["time"](src_meta[3, :])
        emb_week = self.meta_encoders["week"](src_meta[4, :])
        src_meta = torch.stack([emb_area, emb_month, emb_day, emb_time, emb_week], dim=0)

        gpv_hidden = torch.cat([gpv_output[0, :, :], gpv_output[-1, :, :]], dim=1) # (batch_size, d_model * 2 * 2)
        amedas_hidden = src_amedas.transpose(0, 1).reshape(batch_size, -1) # (batch_size, num_amedas_types * d_model)
        meta_hidden = src_meta.transpose(0, 1).reshape(batch_size, -1) # (batch_size, num_meta_types * d_model)

        # initital state of decoder
        data_h = self.relu(self.input_to_dmodel(torch.cat([gpv_hidden, amedas_hidden, meta_hidden], dim=1))) # (batch_size, d_model)
        encoder_hidden = self.reset(data_h)
        return gpv_output, src_amedas, src_meta, encoder_hidden

    def reset(self, hidden_state):
        # initialize hidden states of word decoder
        batch_size = hidden_state.size(0)
        decoder_hidden = torch.zeros((self.num_layers, batch_size, self.d_model), dtype=torch.float32).to(self.device)
        nn.init.normal_(decoder_hidden, mean=0, std=0.05)
        decoder_hidden[0, :, :] = hidden_state
        return decoder_hidden

    def forward(self, src_gpv, src_amedas, src_meta, src_comment):
        """[summary]

        Args:
            src_gpv ([type]): [description]
            src_amedas ([type]): [description]
            src_meta ([type]): [description]
            src_comment ([type]): [description]

        Returns:
            [type]: [description]
        """

        """ encode GPV/AMeDAS/Meta"""
        gpv_output, amedas_output, meta_output, encoder_hidden = \
            self.encode(src_gpv, src_amedas, src_meta, src_comment)

        # initialize outputs of weather labels and weather hidden
        ZERO = torch.zeros(1, 1).to(self.device)
        sunny_out, cloudy_out, rain_out, snow_out, weather_hidden = \
            ZERO, ZERO, ZERO, ZERO, None

        """ decode weather labels """
        if self.weather == "label":
            sunny_out, sunny_hidden = self.sunny_decoder(encoder_hidden[0])
            cloudy_out, cloudy_hidden = self.cloudy_decoder(encoder_hidden[0])
            rain_out, rain_hidden = self.rain_decoder(encoder_hidden[0])
            snow_out, snow_hidden = self.snow_decoder(encoder_hidden[0])
            weather_hidden = torch.stack([sunny_hidden, cloudy_hidden, rain_hidden, snow_hidden], dim=0)

        """ decode tokens """
        token_out = []
        tgt_word_embeddings = []
        hidden = encoder_hidden # initial state of decoder
        for word_input in src_comment:
            output, hidden, word_emb = self.token_decoder(
                word_input, hidden, gpv_output, amedas_output, meta_output, weather_hidden)
            token_out.append(output)
            tgt_word_embeddings.append(word_emb)

        token_out = torch.stack(token_out, dim=0)
        tgt_text_embed = torch.cat(tgt_word_embeddings, dim=0)

        return (F.log_softmax(token_out, dim=-1), \
            F.log_softmax(sunny_out, dim=-1), F.log_softmax(cloudy_out, dim=-1), \
            F.log_softmax(rain_out, dim=-1), F.log_softmax(snow_out, dim=-1), \
            weather_hidden, tgt_text_embed)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs): 
        """[summary]

        Args:
            epoch ([type]): [description]
            batch_idx ([type]): [description]
            optimizer ([type]): [description]
            optimizer_idx ([type]): [description]
            optimizer_closure ([type]): [description]
            on_tpu ([type]): [description]
            using_native_amp ([type]): [description]
            using_lbfgs ([type]): [description]
        """        
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        lr_scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=1, min_lr=1e-5, verbose=True)
        scheduler = {
            'scheduler': lr_scheduler, # The LR scheduler instance (required)
            'interval': 'epoch', # The unit of the scheduler's step size
            'frequency': 1, # The frequency of the scheduler
            'reduce_on_plateau': True, # For ReduceLROnPlateau scheduler
            'monitor': 'val_loss', # Metric for ReduceLROnPlateau to monitor
            'strict': True, # Whether to crash the training if `monitor` is not found
            'name': None, # Custom name for LearningRateMonitor to use
        }
        return [optimizer], [scheduler]

    def greedy_token_decode(self, hidden, gpv_output, amedas_output, meta_output, weather_hidden, token_generation_limit=128):
        _, batch_size, hidden_size = hidden.size()
        decoded_batch = torch.zeros((batch_size, token_generation_limit))
        word_input = torch.tensor([[IDs.BOS.value] for _ in range(batch_size)], dtype=torch.long).to(self.device)
        for idx in range(token_generation_limit):
            output, hidden, _ = self.token_decoder(word_input, hidden, gpv_output, amedas_output, meta_output, weather_hidden)
            topv, topi = output.data.topk(1)  # [batch_size, vocab_size] get candidates
            decoded_batch[:, idx] = topi.view(-1)
            word_input = topi
        return decoded_batch.detach().tolist()

    @torch.no_grad()
    def beam_token_decode(self, hidden, gpv_output, amedas_output, meta_output, weather_hidden, beam_width=5):
        max_steps = 128 # The maximum number of decoding steps to take,
        self.beam_search = BeamSearch(end_index=IDs.EOS.value, max_steps=max_steps, beam_size=beam_width)        
        batch_size = hidden.size(1)

        start_predictions = torch.tensor([IDs.BOS.value] * batch_size, dtype=torch.long, device=self.device)
        start_state = {
            "prev_tokens": torch.zeros(batch_size, 0, dtype=torch.long, device=self.device),
            "decoder_hidden": hidden
        }
        def step(last_tokens, current_state, t):
            """
            Args:
                last_tokens: (group_size,)
                current_state: {}
                t: int
            """
            nonlocal gpv_output
            nonlocal amedas_output
            nonlocal meta_output
            nonlocal weather_hidden
            group_size = last_tokens.size(0)
            # cocatenate prev_tokens with last_tokens
            prev_tokens = torch.cat([current_state["prev_tokens"], last_tokens.unsqueeze(1)], dim=-1)  # [B*k, t+1]

            # expand context hiddens for beam search decoding
            if group_size != gpv_output.size(1):
                gpv_output = gpv_output.unsqueeze(2)\
                    .expand(gpv_output.size(0), gpv_output.size(1), beam_width, gpv_output.size(-1))\
                    .reshape(gpv_output.size(0), gpv_output.size(1) * beam_width, gpv_output.size(-1))
                amedas_output = amedas_output.unsqueeze(2)\
                    .expand(amedas_output.size(0), amedas_output.size(1), beam_width, amedas_output.size(-1))\
                    .reshape(amedas_output.size(0), amedas_output.size(1) * beam_width, amedas_output.size(-1))
                meta_output = meta_output.unsqueeze(2)\
                    .expand(meta_output.size(0), meta_output.size(1), beam_width, meta_output.size(-1))\
                    .reshape(meta_output.size(0), meta_output.size(1) * beam_width, meta_output.size(-1))
                weather_hidden = weather_hidden.unsqueeze(2)\
                    .expand(weather_hidden.size(0), weather_hidden.size(1), beam_width, weather_hidden.size(-1))\
                    .reshape(weather_hidden.size(0), weather_hidden.size(1) * beam_width, weather_hidden.size(-1)) if weather_hidden is not None else None

            # decode for one step using decoder
            decoder_output, decoder_hidden, _ = self.token_decoder(
                prev_tokens[:, -1], current_state["decoder_hidden"], gpv_output, amedas_output, meta_output, weather_hidden)

            current_state["prev_tokens"] = prev_tokens # update prev_tokens
            current_state["decoder_hidden"] = decoder_hidden # update decoder_hidden

            return (decoder_output, current_state)

        predictions, log_probs = self.beam_search.search(
            start_predictions=start_predictions, 
            start_state=start_state, 
            step=step)

        return predictions, log_probs

class WNCGTransformerModel(WNCGBaseModel):
    def __init__(self, input_gpv_dim, d_model, nhead, num_layers, n_vocab, 
        input_amedas_seqlen, weather, cl, lr=0.001, dropout_p=0.2, warm_up_steps=4000):
        super().__init__(d_model, weather, cl, dropout_p)
        """ Encoder """
        # encoder for gpv
        self.gpv_encoder = MLPEncoder(input_gpv_dim, d_model, dropout_p=dropout_p)
        self.pos_encoder = PositionalEncoding(d_model)
        # encoder for amedas
        self.amedas_to_dmodel = nn.Linear(input_amedas_seqlen, d_model)
        # encoder for meta-data
        metaenc = {}
        metaenc["area"] = nn.Embedding(277, d_model)
        metaenc["month"] = nn.Embedding(12, d_model) 
        metaenc["day"] = nn.Embedding(31, d_model)
        metaenc["time"] = nn.Embedding(24, d_model)
        metaenc["week"] = nn.Embedding(7, d_model)
        self.meta_encoders = nn.ModuleDict(metaenc)
        # encoder
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers)

        """ Decoder """
        # word decoder
        self.token_embedder = nn.Embedding(n_vocab, d_model, padding_idx=IDs.PAD.value)
        self.token_position = PositionalEncoding(d_model)
        self.token_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead), num_layers=num_layers)
        self.token_output = nn.Linear(d_model, n_vocab)

        # make the arguments global
        self.lr = lr
        # weather label
        self.weather = weather
        # content agreement loss
        self.cl = cl
        # warm up steps for learning rate
        self.warm_up_steps = warm_up_steps
        # save the arguments
        self.save_hyperparameters()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask==1, float('-inf'))
        return mask

    def encode(self, src_gpv, src_amedas, src_meta, src_comment):
        """ encode """
        # encode gpv-data
        src_gpv = self.gpv_encoder(src_gpv)
        src_gpv  = self.pos_encoder(src_gpv)
        # encode amedas-data
        src_amedas = self.amedas_to_dmodel(src_amedas)
        # encode meta-data
        emb_area = self.meta_encoders["area"](src_meta[0, :])
        emb_month = self.meta_encoders["month"](src_meta[1, :])
        emb_day = self.meta_encoders["day"](src_meta[2, :])
        emb_time = self.meta_encoders["time"](src_meta[3, :])
        emb_week = self.meta_encoders["week"](src_meta[4, :])
        src_meta = torch.stack([emb_area, emb_month, emb_day, emb_time, emb_week], dim=0)
        # concatenate input-data (gpv, amedas, meta) 
        src_data = torch.cat([src_gpv, src_amedas, src_meta], dim=0) # (seq_len[9(gpv) + 4(amedas) + 5(meta)], batch_size, d_model)
        # encode input-data by transformer
        src_memory = self.transformer_encoder(src_data) # (seq_len, batch_size, d_model)
        return src_gpv, src_amedas, src_meta, src_memory

    def forward(self, src_gpv, src_amedas, src_meta, src_comment):
        """[summary]

        Args:
            src_gpv ([type]): [description]
            src_amedas ([type]): [description]
            src_meta ([type]): [description]
            src_comment ([type]): [description]

        Returns:
            [type]: [description]
        """

        """ encode gpv/amedas/meta """
        src_gpv, src_amedas, src_meta, src_memory = \
            self.encode(src_gpv, src_amedas, src_meta, src_comment)

        # initialize outputs of weather labels and weather hidden
        ZERO = torch.zeros(1, 1).to(self.device)
        sunny_out, cloudy_out, rain_out, snow_out, weather_hidden = \
            ZERO, ZERO, ZERO, ZERO, ZERO, None

        """ decode weather labels """
        if self.weather == "label":
            sunny_out, sunny_hidden = self.sunny_decoder(src_memory[0])
            cloudy_out, cloudy_hidden = self.cloudy_decoder(src_memory[0])
            rain_out, rain_hidden = self.rain_decoder(src_memory[0])
            snow_out, snow_hidden = self.snow_decoder(src_memory[0])
            weather_hidden = torch.stack([sunny_hidden, cloudy_hidden, rain_hidden, snow_hidden], dim=0)

        # induce weather_hidden into input
        if self.weather is not None:
            src_memory = torch.cat([src_memory, weather_hidden], dim=0)

        """ decode tokens """
        # prepare masks for word decoder        
        src_comment_len = src_comment.size(0) # seq_len
        # mask for padding token 
        src_comment_padd_mask = (src_comment == IDs.PAD.value).transpose(0, 1).to(self.device) # (batch_size, seq_len)
        # mask for subsequence
        src_comment_attn_mask = self.generate_square_subsequent_mask(src_comment_len).to(self.device) # (seq_len, seq_len)
        # embedding
        src_comment_emb = self.token_embedder(src_comment) # (seqlen, batch_size, d_model)
        src_comment_emb_pos = self.token_position(src_comment_emb)        
        # decode
        token_hidden = self.token_decoder(src_comment_emb_pos, src_memory, 
            tgt_mask=src_comment_attn_mask, tgt_key_padding_mask=src_comment_padd_mask) # (seqlen, batch_size, d_model)
        # output distribution over vocabularies
        token_out = self.token_output(token_hidden)

        return (F.log_softmax(token_out, dim=-1), \
            F.log_softmax(sunny_out, dim=-1), F.log_softmax(cloudy_out, dim=-1), \
            F.log_softmax(rain_out, dim=-1), F.log_softmax(snow_out, dim=-1),
            weather_hidden, src_comment_emb)

    # learning rate warm-up
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                    optimizer_closure, on_tpu, using_native_amp, using_lbfgs): 
        """[summary]

        Args:
            epoch ([type]): [description]
            batch_idx ([type]): [description]
            optimizer ([type]): [description]
            optimizer_idx ([type]): [description]
            optimizer_closure ([type]): [description]
            on_tpu ([type]): [description]
            using_native_amp ([type]): [description]
            using_lbfgs ([type]): [description]
        """
        # warm up lr
        for pg in optimizer.param_groups:
            self.lr = (self.hparams.d_model**-0.5) * min(float(self.trainer.global_step + 1)**-0.5, float(self.trainer.global_step + 1) * self.warm_up_steps**-1.5)
            pg['lr'] = self.lr
        # update params
        optimizer.step(closure=optimizer_closure)
        optimizer.zero_grad()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, betas=(0.9, 0.98), eps=1e-09)
        return optimizer

    def greedy_token_decode(self, src_memory, gpv_output, amedas_output, meta_output, weather_hidden, token_generation_limit=128):
        """ decode tokens """
        _, batch_size, d_model = src_memory.size()
        src_comment = torch.tensor([[IDs.BOS.value for _ in range(batch_size)]], dtype=torch.long).to(self.device)
        decoded_batch = torch.zeros((batch_size, token_generation_limit))

        # induce weather_hidden into input
        if self.weather is not None:
            src_memory = torch.cat([src_memory, weather_hidden], dim=0)

        for idx in range(token_generation_limit):
            # prepare masks for word decoder        
            src_comment_len = idx + 1# seq_len
            # mask for padding token 
            src_comment_padd_mask = (src_comment == IDs.PAD.value).transpose(0, 1).to(self.device) # (batch_size, seq_len)
            # mask for subsequence
            src_comment_attn_mask = self.generate_square_subsequent_mask(src_comment_len).to(self.device) # (seq_len, seq_len)
            # embedding
            src_comment_emb = self.token_embedder(src_comment) # (seqlen, batch_size, d_model)
            src_comment_emb = self.token_position(src_comment_emb)        
            # decode
            token_hidden = self.token_decoder(src_comment_emb, src_memory, 
                tgt_mask=src_comment_attn_mask, tgt_key_padding_mask=src_comment_padd_mask) # (seqlen, batch_size, d_model)
            # output distribution over vocabularies
            token_out = self.token_output(token_hidden)

            topv, topi = token_out[-1, :, :].data.topk(1)
            decoded_batch[:, idx] = topi.view(-1)
            topi = topi.transpose(0, 1)

            # concat source with output
            src_comment = torch.cat([src_comment, topi], dim=0)

        return decoded_batch.detach().tolist()

    @torch.no_grad()
    def beam_token_decode(self, src_memory, gpv_output, amedas_output, meta_output, weather_hidden, beam_width=5):
        max_steps = 128 # The maximum number of decoding steps to take,
        self.beam_search = BeamSearch(end_index=IDs.EOS.value, max_steps=max_steps, beam_size=beam_width)        
        batch_size = src_memory.size(1)

        # induce weather_hidden into input
        if self.weather is not None:
            src_memory = torch.cat([src_memory, weather_hidden], dim=0)

        start_predictions = torch.tensor([IDs.BOS.value] * batch_size, dtype=torch.long, device=self.device)
        start_state = {
            "prev_tokens": torch.zeros(batch_size, 0, dtype=torch.long, device=self.device), # set none of prev_tokens
            "decoder_hidden": src_memory # (seq_len, batch_size, d_model)
        }
        def step(last_tokens, current_state, t):
            """
            Args:
                last_tokens: (group_size,)
                current_state: {}
                t: int
            """
            # concatenate prev_tokens with last_tokens
            prev_tokens = torch.cat([current_state["prev_tokens"], last_tokens.unsqueeze(1)], dim=-1)  # [batch_size * beam_width, t+1]
            # embedding
            prev_tokens_emb = self.token_embedder(prev_tokens).transpose(0, 1) # (seqlen, batch_size, d_model)
            prev_tokens_emb = self.token_position(prev_tokens_emb)   
            prev_tokens_len = prev_tokens.size(1)
            # mask for padding token 
            prev_token_padd_mask = (prev_tokens == IDs.PAD.value).to(self.device) # (batch_size, seq_len)
            # mask for subsequence
            prev_token_attn_mask = self.generate_square_subsequent_mask(prev_tokens_len).to(self.device) # (seq_len, seq_len)
            # decode
            token_hidden = self.token_decoder(prev_tokens_emb, current_state["decoder_hidden"], 
                tgt_mask=prev_token_attn_mask, tgt_key_padding_mask=prev_token_padd_mask) # (seqlen, batch_size, d_model)

            # output distribution over vocabularies
            token_out = self.token_output(token_hidden)
            # get outout distribution for last token
            decoder_output = F.log_softmax(token_out[-1, :, :], dim=-1)
            # update prev_tokens
            current_state["prev_tokens"] = prev_tokens 
            return (decoder_output, current_state)

        predictions, log_probs = self.beam_search.search(
            start_predictions=start_predictions, 
            start_state=start_state, 
            step=step)

        return predictions, log_probs