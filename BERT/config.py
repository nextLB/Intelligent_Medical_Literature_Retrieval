#!/usr/bin/env python3
"""
配置文件
可在此处修改训练参数
"""

CONFIG = {
    'model_name': 'bert-base-chinese',
    'max_length': 256,
    'batch_size': 16,
    'num_epochs': 5,
    'learning_rate': 2e-5,
    'warmup_ratio': 0.1,
    'dropout': 0.3,
    'num_classes': 7,
    'seed': 42,
    'data_dir': '../Data_Preprocessing',
    'output_dir': './checkpoints',
    'log_dir': './logs'
}

LABEL_MAPPING = {
    '肿瘤学': 0,
    '内分泌代谢': 1,
    '心血管': 2,
    '其他': 3,
    '感染性疾病': 4,
    '中医药': 5,
    '神经内科': 6
}

LABEL_MAPPING_REVERSE = {
    0: '肿瘤学',
    1: '内分泌代谢',
    2: '心血管',
    3: '其他',
    4: '感染性疾病',
    5: '中医药',
    6: '神经内科'
}
