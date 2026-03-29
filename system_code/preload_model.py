#!/usr/bin/env python3
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

print("预加载BERT模型到缓存...")
print("使用镜像: https://hf-mirror.com")

try:
    from transformers import BertTokenizer, BertModel
    import torch
    
    print("加载分词器...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
    print("✓ 分词器加载完成")
    
    print("加载模型...")
    model = BertModel.from_pretrained('bert-base-chinese', local_files_only=True)
    print("✓ 模型加载完成")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        model = model.to(device)
        print("✓ 模型已转移到GPU")
    
    print("\n预加载完成！训练时将更快启动。")
except Exception as e:
    print(f"预加载出错: {e}")
    print("这不影响训练，只是首次加载会稍慢。")
