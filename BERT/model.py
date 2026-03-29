#!/usr/bin/env python3
"""
BERT分类模型
基于预训练BERT的中文医学文献分类模型
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertClassifier(nn.Module):
    def __init__(self, num_classes=7, model_name='bert-base-chinese', dropout=0.3):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        self.num_classes = num_classes
        self.model_name = model_name
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        return logits


class BertForMultiLabel(nn.Module):
    def __init__(self, num_classes=7, model_name='bert-base-chinese', dropout=0.3):
        super(BertForMultiLabel, self).__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
        self.num_classes = num_classes
        self.model_name = model_name
    
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        pooled_output = outputs.last_hidden_state[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        
        logits = self.classifier(pooled_output)
        return logits


def get_model(num_classes=7, model_name='bert-base-chinese', device='cuda'):
    model = BertClassifier(num_classes=num_classes, model_name=model_name)
    model = model.to(device)
    return model


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = get_model(num_classes=7, device=device)
    
    total, trainable = count_parameters(model)
    print(f"Total parameters: {total:,}")
    print(f"Trainable parameters: {trainable:,}")
    
    batch_size = 8
    seq_length = 256
    input_ids = torch.randint(0, 21128, (batch_size, seq_length)).to(device)
    attention_mask = torch.ones(batch_size, seq_length).to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
    
    print(f"Output shape: {outputs.shape}")
