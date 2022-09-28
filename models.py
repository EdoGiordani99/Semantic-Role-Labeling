import torch
import numpy as np
import torch.nn.functional as F

from torch import nn
from transformers import AutoModel


class BERT_Model(nn.Module):

    def __init__(self,
                 language_model_name: str,
                 num_labels: int,
                 rnn_size: int,
                 num_rnn: int,
                 bidirectional: bool,
                 dropout: float,
                 fine_tune: bool,
                 class_weights=None):

        super().__init__()

        self.num_labels = num_labels
        self.class_weights = class_weights

        self.bert_model = AutoModel.from_pretrained(language_model_name,
                                                    output_hidden_states=True)

        if not fine_tune:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.dropout = torch.nn.Dropout(dropout)

        bert_output_size = self.bert_model.config.hidden_size

        self.rnn = nn.LSTM(input_size=2 * bert_output_size,
                           hidden_size=rnn_size,
                           bidirectional=bidirectional,
                           num_layers=num_rnn,
                           dropout=dropout,
                           batch_first=True)

        if bidirectional:
            linear_size = 2 * rnn_size
            mid_size = rnn_size
        else:
            linear_size = rnn_size
            mid_size = rnn_size / 2

        self.linear1 = torch.nn.Linear(linear_size, mid_size, bias=False)
        self.linear2 = torch.nn.Linear(mid_size, self.num_labels, bias=False)

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                token_types_ids: torch.Tensor = None,
                predicate_idx: torch.Tensor = None,
                labels: torch.Tensor = None,
                compute_predictions: bool = False,
                compute_loss: bool = True,
                *args,
                **kwargs):

        bert_model_inputs = {'input_ids': input_ids,
                             'attention_mask': attention_mask}

        if token_types_ids:
            bert_model_inputs['token_types_ids'] = token_types_ids

        model_outputs = self.bert_model(**bert_model_inputs)
        model_outputs_sum = torch.stack(model_outputs.hidden_states[-4:], dim=0).sum(dim=0)

        # concatenating words bert embeddings with predicate bert embedding
        bert_outputs = model_outputs_sum.tolist()
        predicate_idx = predicate_idx.tolist()
        concat = []
        for i, sentence in enumerate(bert_outputs):
            new_sentence = []
            idx = predicate_idx[i]
            for embed in sentence:
                new_sentence.append(embed + bert_outputs[i][idx])
            concat.append(new_sentence)

        concat = torch.FloatTensor(concat).to(input_ids.device)

        rnn_out, _ = self.rnn(concat)

        lin_out = self.linear1(rnn_out)
        lin_out = self.dropout(lin_out)
        logits = self.linear2(lin_out)

        output = {'logits': logits}

        if compute_predictions:
            preds = output['logits'].argmax(dim=-1)
            output['preds'] = preds

        if compute_loss:
            loss = self.compute_loss(logits=output['logits'],
                                     labels=labels)
            output['loss'] = loss

        return output

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor):

        cross_entropy_loss = F.cross_entropy(logits.view(-1, self.num_labels),
                                             labels.view(-1),
                                             ignore_index=-100,
                                             weight=self.class_weights)
        return cross_entropy_loss


class PoS_BiLSTM_Model(nn.Module):

    def __init__(self,
                 language_model_name: str,
                 num_labels: int,
                 rnn_size: int,
                 num_rnn: int,
                 pos_rnn_size: int,
                 pos_num_rnn: int,
                 bidirectional: bool,
                 dropout: float,
                 fine_tune: bool,
                 class_weights=None,
                 *args,
                 **kwargs):

        super().__init__()

        # Setting the bert model
        self.bert_model = AutoModel.from_pretrained(language_model_name,
                                                    output_hidden_states=True)
        if not fine_tune:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        bert_output_size = self.bert_model.config.hidden_size

        self.num_labels = num_labels
        self.class_weights = class_weights
        self.dropout = torch.nn.Dropout(dropout)

        # input size = lenght of pos tags vocabulary
        self.pos_rnn = nn.LSTM(input_size=18,
                               hidden_size=pos_rnn_size,
                               bidirectional=bidirectional,
                               num_layers=pos_num_rnn,
                               dropout=dropout,
                               batch_first=True)
        if bidirectional:
            pos_out_size = 2 * pos_rnn_size
        else:
            pos_out_size = pos_rnn_size

        self.rnn = nn.LSTM(input_size=bert_output_size + pos_out_size + 1,
                           hidden_size=rnn_size,
                           bidirectional=bidirectional,
                           num_layers=num_rnn,
                           dropout=dropout,
                           batch_first=True)
        if bidirectional:
            linear_size = 2 * rnn_size
        else:
            linear_size = rnn_size

        mid_size = int(linear_size / 2)

        self.linear1 = torch.nn.Linear(linear_size, mid_size, bias=False)
        self.linear2 = torch.nn.Linear(mid_size, self.num_labels, bias=False)

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                predicates: torch.Tensor = None,
                token_types_ids: torch.Tensor = None,
                pos_tags: torch.Tensor = None,
                labels: torch.Tensor = None,
                compute_predictions: bool = False,
                compute_loss: bool = True,
                *args,
                **kwargs):

        bert_model_inputs = {'input_ids': input_ids,
                             'attention_mask': attention_mask}

        # since not all the models support token types
        if token_types_ids:
            bert_model_inputs['token_types_ids'] = token_types_ids

        model_outputs = self.bert_model(**bert_model_inputs)
        model_outputs_sum = torch.stack(model_outputs.hidden_states[-4:], dim=0).sum(dim=0)

        # applying dropout
        bert_outputs_sum = self.dropout(model_outputs_sum)

        ohe_pos_tags = []

        for tag in pos_tags:
            pos_sample = []
            for t in tag:
                a = np.zeros(18)
                a[t] = 1
                pos_sample.append(list(a))
            ohe_pos_tags.append(pos_sample)

        ohe_pos_tags = torch.Tensor(ohe_pos_tags)

        # inputing pos_tags
        out_pos_rnn, _ = self.pos_rnn(ohe_pos_tags)

        concat = torch.cat((bert_outputs_sum, out_pos_rnn), 2)
        rnn_in = torch.cat((predicates, concat), 2)

        rnn_out, _ = self.rnn(rnn_in)

        out = self.linear1(rnn_out)
        logits = self.linear2(out)

        output = {'logits': logits}

        if compute_predictions:
            preds = output['logits'].argmax(dim=-1)
            output['preds'] = preds

        if compute_loss:
            loss = self.compute_loss(logits=output['logits'],
                                     labels=labels)
            output['loss'] = loss

        return output

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor):

        cross_entropy_loss = F.cross_entropy(logits.view(-1, self.num_labels),
                                             labels.view(-1),
                                             ignore_index=-100)
        return cross_entropy_loss


class BiaffineAttention(torch.nn.Module):

    def __init__(self, in_dim1, in_dim2, out_dim):
        super(BiaffineAttention, self).__init__()

        self.in_dim1 = in_dim1
        self.in_dim2 = in_dim2
        self.out_dim = out_dim

        self.bilinear = nn.Bilinear(in_dim1, in_dim2, out_dim, bias=False)
        self.linear = nn.Linear(in_dim1+in_dim2, out_dim, bias=True)

        self.reset_parameters()

    def forward(self, start, end):
        bil_out = self.bilinear(start, end)
        lin_out = self.linear(torch.cat((start, end), dim=-1))

        return bil_out + lin_out

    def reset_parameters(self):
        self.bilinear.reset_parameters()
        self.linear.reset_parameters()


class PoS_Biaffine_Model(nn.Module):

    def __init__(self,
                 language_model_name: str,
                 num_labels: int,
                 rnn_size: int,
                 num_rnn: int,
                 pos_rnn_size: int,
                 pos_num_rnn: int,
                 bidirectional: bool,
                 dropout: float,
                 fine_tune: bool,
                 class_weights=None,
                 *args,
                 **kwargs):

        super().__init__()

        # Setting the bert model
        self.bert_model = AutoModel.from_pretrained(language_model_name,
                                                    output_hidden_states=True)
        if not fine_tune:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        bert_output_size = self.bert_model.config.hidden_size

        self.num_labels = num_labels
        self.class_weights = class_weights
        self.dropout = torch.nn.Dropout(dropout)

        # input size = lenght of pos tags vocabulary
        self.pos_rnn = nn.LSTM(input_size=18,
                               hidden_size=pos_rnn_size,
                               bidirectional=bidirectional,
                               num_layers=pos_num_rnn,
                               dropout=dropout,
                               batch_first=True)
        if bidirectional:
            pos_out_size = 2 * pos_rnn_size
        else:
            pos_out_size = pos_rnn_size

        self.rnn1 = nn.LSTM(input_size=bert_output_size + pos_out_size + 1,
                            hidden_size=rnn_size,
                            bidirectional=bidirectional,
                            num_layers=num_rnn,
                            dropout=dropout,
                            batch_first=True)

        self.rnn2 = nn.LSTM(input_size=2 * rnn_size,
                            hidden_size=rnn_size,
                            bidirectional=True,
                            num_layers=2,
                            dropout=dropout,
                            batch_first=True)
        if bidirectional:
            linear_size = 2 * rnn_size
        else:
            linear_size = rnn_size

        self.start_linear = torch.nn.Linear(linear_size, linear_size, bias=False)
        self.start_activation = nn.ReLU()

        self.end_linear = torch.nn.Linear(linear_size, linear_size, bias=False)
        self.end_activation = nn.ReLU()

        self.biaffine = BiaffineAttention(linear_size, linear_size, self.num_labels)

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                predicates: torch.Tensor = None,
                token_types_ids: torch.Tensor = None,
                pos_tags: torch.Tensor = None,
                labels: torch.Tensor = None,
                compute_predictions: bool = False,
                compute_loss: bool = True,
                *args,
                **kwargs):

        bert_model_inputs = {'input_ids': input_ids,
                             'attention_mask': attention_mask}

        # since not all the models support token types
        if token_types_ids:
            bert_model_inputs['token_types_ids'] = token_types_ids

        model_outputs = self.bert_model(**bert_model_inputs)
        model_outputs_sum = torch.stack(model_outputs.hidden_states[-4:], dim=0).sum(dim=0)

        # applying dropout
        bert_outputs_sum = self.dropout(model_outputs_sum)

        ohe_pos_tags = []

        for tag in pos_tags:
            pos_sample = []
            for t in tag:
                a = np.zeros(18)
                a[t] = 1
                pos_sample.append(list(a))
            ohe_pos_tags.append(pos_sample)

        ohe_pos_tags = torch.Tensor(ohe_pos_tags)

        # inputing pos_tags
        out_pos_rnn, _ = self.pos_rnn(ohe_pos_tags)

        concat = torch.cat((bert_outputs_sum, out_pos_rnn), 2)
        rnn_in = torch.cat((predicates, concat), 2)

        rnn_out1, _ = self.rnn1(rnn_in)
        rnn_out2, _ = self.rnn2(rnn_out1)

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor):

        cross_entropy_loss = F.cross_entropy(logits.view(-1, self.num_labels),
                                             labels.view(-1),
                                             ignore_index=-100)
        return cross_entropy_loss


class BERT_OHE_Model(nn.Module):

    def __init__(self,
                 language_model_name: str,
                 num_labels: int,
                 rnn_size: int,
                 bidirectional: bool,
                 num_rnn: int,
                 dropout: float,
                 fine_tune: bool,
                 class_weights=None,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_labels = num_labels
        self.class_weights = class_weights

        self.bert_model = AutoModel.from_pretrained(language_model_name,
                                                    output_hidden_states=True)
        if not fine_tune:
            for param in self.bert_model.parameters():
                param.requires_grad = False

        self.dropout = torch.nn.Dropout(dropout)

        bert_output_size = self.bert_model.config.hidden_size

        # we add + 1 to input_size for the predicate token
        self.rnn = nn.LSTM(input_size=bert_output_size + 1,
                           hidden_size=rnn_size,
                           bidirectional=bidirectional,
                           num_layers=num_rnn,
                           batch_first=True)

        if bidirectional:
            linear_size = 2 * rnn_size
        else:
            linear_size = rnn_size

        self.linear = torch.nn.Linear(linear_size, num_labels, bias=False)

    def forward(self,
                input_ids: torch.Tensor = None,
                attention_mask: torch.Tensor = None,
                predicates: torch.Tensor = None,
                token_types_ids: torch.Tensor = None,
                labels: torch.Tensor = None,
                compute_predictions: bool = False,
                compute_loss: bool = True,
                *args,
                **kwargs):

        bert_model_inputs = {'input_ids': input_ids,
                             'attention_mask': attention_mask}
        # since not all the models support token types
        if token_types_ids:
            bert_model_inputs['token_types_ids'] = token_types_ids

        model_outputs = self.bert_model(**bert_model_inputs)
        bert_outputs_sum = torch.stack(model_outputs.hidden_states[-4:], dim=0).sum(dim=0)

        rnn_in = torch.cat((predicates, bert_outputs_sum), 2)

        rnn_out, _ = self.rnn(rnn_in)

        rnn_out = self.dropout(rnn_out)

        logits = self.linear(rnn_out)

        output = {'logits': logits}

        if compute_predictions:
            preds = output['logits'].argmax(dim=-1)
            output['preds'] = preds

        if compute_loss:
            loss = self.compute_loss(logits=output['logits'],
                                     labels=labels)
            output['loss'] = loss

        return output

    def compute_loss(self,
                     logits: torch.Tensor,
                     labels: torch.Tensor):

        cross_entropy_loss = F.cross_entropy(logits.view(-1, self.num_labels),
                                             labels.view(-1),
                                             ignore_index=-100,
                                             weight=self.class_weights)
        return cross_entropy_loss
