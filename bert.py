from typing import Dict, List, Optional, Union, Tuple, Callable
import math
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F
from base_bert import BertPreTrainedModel
from utils import *

#  module BertSelfAttention: implement a module for calculating attention score from 
#  an embedding input like in the paper Attention is all you need
class BertSelfAttention(nn.Module):
  def __init__(self, config):    
    super().__init__()
    # initialize config
    self.num_attention_heads = config.num_attention_heads
    self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
    self.all_head_size = self.num_attention_heads * self.attention_head_size

    # initialize matrix for key, value, query 
    self.query = nn.Linear(config.hidden_size, self.all_head_size)
    self.key = nn.Linear(config.hidden_size, self.all_head_size)
    self.value = nn.Linear(config.hidden_size, self.all_head_size)

    # adding dropout score to the attention score 
    # (according to the CMU's original code template: ...'empirically observe that it yields better performance')
    self.dropout = nn.Dropout(config.attention_probs_dropout_prob)


  # purpose: transform key, value, query embedding matrix of a sentence into smaller heads for multi-head attention dot product later
  # input: x: [bs, seq_len, hidden_state_size], linear_layer: the corresponding matrix of k, v, q
  # output: the tensors of k, v, q after transformation (splitting into multiple heads with the size: [bs, num_attention_heads, seq_len, attention_head_size])
  def transform(self, x, linear_layer):
    bs, seq_len = x.shape[:2]
    # first, transform the hidden state to the key, value, query matrix by multiplying with the corresponding matrix
    proj = linear_layer(x)
    # second, split the hidden state to self.num_attention_heads, each of size self.attention_head_size (hidden_state_size = num_attention_heads * attention_head_size)
    # ('view' function of pytorch: reshape the tensor to the new shape)
    proj = proj.view(bs, seq_len, self.num_attention_heads, self.attention_head_size)
    proj = proj.transpose(1, 2)
    return proj

  # purpose: calculate the attention score for each token in a sentence
  # input: key, query, value: the transformed key, value, query matrix (from a big matrix into smaller heads) of sentences in a batch
  #        (key, query, value: [bs, num_attention_heads, seq_len, attention_head_size] => output of the transform function)
  # output: a tensor containing attention score matrixs of multiple heads of the sentences in a batch
  #        (attention score matrix: [bs, num_attention_heads, seq_len, seq_len])
  def attention(self, key, query, value, attention_mask):
    # attention scores are calculated by multiply query and key, following eq (1) of https://arxiv.org/pdf/1706.03762.pdf 
    # S[*, i, j, k] represents the (unnormalized)attention score between the j-th and k-th token, given by i-th attention head
    # before normalizing the scores, use the attention mask to mask out the padding token scores
    # (As in the attention_mask non-padding tokens with 0 and padding tokens with a large negative number)
    # attention_mask = torch.where(attention_mask == 0, 1.0, -1e10)

    # query: [bs, num_attention_heads, seq_len, attention_head_size]
    # torch.transpose(query, 2, 3): [bs, num_attention_heads, attention_head_size, seq_len]
    # scores: [bs, num_attention_heads, seq_len, seq_len]
    # key.shape[-1] = attention_head_size (hidden_state_size gets divided into num_attention_heads * attention_head_size)
    scores = torch.matmul(query, torch.transpose(key, 2, 3)) / math.sqrt(key.shape[-1])

    # in case the seq has more than 512 tokens, we need to truncate the attention_mask
    attention_mask = attention_mask[:, :, :, :512]
    scores = scores.masked_fill(attention_mask < 0, -10000)
    
    # normalize the scores as the original implementation in google-research/bert
    normed = torch.softmax(scores, -1)

    # multiply the attention scores to the value and get back 
    per_head = torch.matmul(normed, value)

    # next, we need to concat multi-heads and recover the original shape [bs, seq_len, num_attention_heads * attention_head_size = hidden_size]
    atten = torch.cat([per_head[:, i, :, :] for i in range(per_head.shape[1])], -1)
    return atten

  # purpose: as the name
  # input:     hidden_states: [bs, seq_len, hidden_state]
  #            attention_mask: [bs, 1, 1, seq_len]
  #             output: [bs, seq_len, hidden_state]
  # output: embedding with attention score of a sentence
  def forward(self, hidden_states, attention_mask):
    # first, generate the key, value, query for each token for multi-head attention w/ transform (more details inside the function)
    # of *_layers are of [bs, num_attention_heads, seq_len, attention_head_size]
    key_layer = self.transform(hidden_states, self.key)
    value_layer = self.transform(hidden_states, self.value)
    query_layer = self.transform(hidden_states, self.query)
    attention_mask = attention_mask[:, :, :, :512]

    # second, calculate the attention score for each token in a sentence
    attn_value = self.attention(key_layer, query_layer, value_layer, attention_mask)
    return attn_value

# module BertLayer: implement a module for a encoder block in the bert model
class BertLayer(nn.Module):
  def __init__(self, config):
    super().__init__()
    # self attention
    self.self_attention = BertSelfAttention(config)
    self.attention_dense = nn.Linear(config.hidden_size, config.hidden_size)
    self.attention_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.attention_dropout = nn.Dropout(config.hidden_dropout_prob)
    # feed forward
    self.interm_dense = nn.Linear(config.hidden_size, config.intermediate_size)
    self.interm_af = F.gelu
    # layer out
    self.out_dense = nn.Linear(config.intermediate_size, config.hidden_size)
    self.out_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.out_dropout = nn.Dropout(config.hidden_dropout_prob)

  # purpose: add-norm
  # input: input: the input
  #        output: the input that requires the sublayer to transform
  #        dense_layer, dropout: the sublayer
  #        ln_layer: layer norm that takes input+sublayer(output)
  # output: the output after the add-norm layer
  def add_norm(self, input, output, dense_layer, dropout, ln_layer):
    return ln_layer(input + dropout(dense_layer(output)))


  # purpose: forward pass of the BertLayer (an encoder block)
  # input: hidden_states: the output from the previous bert layer or the embedding layer
  #        attention_mask: the mask for padding tokens
  # output: the output of the BertLayer (an encoder block)
  def forward(self, hidden_states, attention_mask):
    """
    hidden_states: either from the embedding layer (first bert layer) or from the previous bert layer
    each block consists of 
    1. a multi-head attention layer (BertSelfAttention)
    2. a add-norm that takes the output of BertSelfAttention and the input of BertSelfAttention
    3. a feed forward layer
    4. a add-norm that takes the output of feed forward layer and the input of feed forward layer
    """
    # multi-head attention w/ self.self_attention
    atten = self.self_attention(hidden_states, attention_mask)

    # add-norm layer
    norm_atten = self.add_norm(hidden_states, atten, self.attention_dense, self.attention_dropout, self.attention_layer_norm)

    # feed forward
    interim = self.interm_af(self.interm_dense(norm_atten))

    # another add-norm layer
    ffn = self.add_norm(norm_atten, interim, self.out_dense, self.out_dropout, self.out_layer_norm)

    return ffn

# module BertModel: a complete bert model
class BertModel(BertPreTrainedModel):
  """
  the bert model returns the final embeddings for each token in a sentence, not include extra layers for downstream tasks.
  it consists
  1. embedding (used in self.embed)
  2. a stack of n bert layers (used in self.encode)
  3. a linear transformation layer for [CLS] token (used in self.forward, as given)
  """
  def __init__(self, config):
    super().__init__(config)
    self.config = config

    # embedding
    self.word_embedding = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
    self.pos_embedding = nn.Embedding(config.max_position_embeddings, config.hidden_size)
    self.tk_type_embedding = nn.Embedding(config.type_vocab_size, config.hidden_size)
    self.embed_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
    self.embed_dropout = nn.Dropout(config.hidden_dropout_prob)
    # position_ids (1, len position emb) is a constant, register to buffer
    position_ids = torch.arange(config.max_position_embeddings).unsqueeze(0)
    self.register_buffer('position_ids', position_ids)

    # bert encoder
    self.bert_layers = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])

    # for [CLS] token
    self.pooler_dense = nn.Linear(config.hidden_size, config.hidden_size)
    # activation function for [CLS] token as in the original implementation
    self.pooler_af = nn.Tanh()

    self.init_weights()

  # purpose: get the embedding for each token in a sentence
  # input: input_ids: the input token ids
  # output: the embedding for each token in a sentence
  def embed(self, input_ids):
    input_ids = input_ids[:, :512]  # truncate input sequence to 512  
    input_shape = input_ids.size()
    seq_length = input_shape[1]

    # get word embedding from self.word_embedding
    inputs_embeds = self.word_embedding(input_ids)

    # get position index and position embedding from self.pos_embedding
    pos_ids = self.position_ids[:, :seq_length]
    pos_embeds = self.pos_embedding(pos_ids)

    # get token type ids, since we are not consider token type, just a placeholder
    tk_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device)
    tk_type_embeds = self.tk_type_embedding(tk_type_ids)

    # add three embeddings together
    embeds = inputs_embeds + tk_type_embeds + pos_embeds

    # layer norm and dropout
    embeds = self.embed_layer_norm(embeds)
    embeds = self.embed_dropout(embeds)

    return embeds

  # purpose: encode the hidden states through the bert layers (a stack of encoder blocks)
  # input: hidden_states: the output from the embedding layer [batch_size, seq_len, hidden_size]
  #        attention_mask: [batch_size, seq_len]
  # output: the output of the bert layers 
  def encode(self, hidden_states, attention_mask):
    # get the extended attention mask for self attention
    # returns extended_attention_mask of [batch_size, 1, 1, seq_len]
    # non-padding tokens with 0 and padding tokens with a large negative number 
    extended_attention_mask: torch.Tensor = get_extended_attention_mask(attention_mask, self.dtype)

    # pass the hidden states through the encoder layers
    for i, layer_module in enumerate(self.bert_layers):
      # feed the encoding from the last bert_layer to the next
      hidden_states = layer_module(hidden_states, extended_attention_mask)

    return hidden_states

  # purpose: forward the input through the bert model
  # input: input_ids: the input token ids, 
  #                   shape: [batch_size, seq_len] - seq_len is the max length of the batch
  #       attention_mask: the mask for padding tokens,
  #                      shape: [batch_size, seq_len] - 1 represents non-padding tokens, 0 represents padding tokens
  # output: the last hidden state of the output embedding from the input, shape: [batch_size, seq_len, hidden_size]
  #         and the hidden state of the [CLS] token, shape: [batch_size, hidden_size]
  def forward(self, input_ids, attention_mask):
    # get the embedding for each input token
    embedding_output = self.embed(input_ids=input_ids)

    # feed to a transformer (a stack of BertLayers)
    sequence_output = self.encode(embedding_output, attention_mask=attention_mask)

    # get cls token hidden state
    first_tk = sequence_output[:, 0]
    first_tk = self.pooler_dense(first_tk)
    first_tk = self.pooler_af(first_tk)
    # sequence_output: [batch_size, seq_len, hidden_size]
    # first_tk: [batch_size, hidden_size]
    return {'last_hidden_state': sequence_output, 'pooler_output': first_tk}
