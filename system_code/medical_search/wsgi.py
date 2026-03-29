#!/usr/bin/env python3
import os
import sys
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_search.settings')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from django.core.wsgi import get_wsgi_application

application = get_wsgi_application()

def preload_bert():
    try:
        print("\n[预加载] 使用镜像加载BERT模型...")
        from transformers import BertTokenizer, BertModel
        import torch
        
        print("[预加载] 加载分词器...")
        BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
        print("[预加载] ✓ 分词器已就绪")
        
        print("[预加载] 加载模型...")
        model = BertModel.from_pretrained('bert-base-chinese', local_files_only=True)
        print("[预加载] ✓ 模型已就绪")
        
        if torch.cuda.is_available():
            model = model.to('cuda')
            print("[预加载] ✓ 模型已转移到GPU")
        
        print("[预加载] BERT模型预加载完成\n")
    except Exception as e:
        print(f"[预加载] BERT预加载跳过: {e}\n")

try:
    preload_bert()
except:
    pass
