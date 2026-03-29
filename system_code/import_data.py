#!/usr/bin/env python3
import os
import sys
import django
import json
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_search.settings')
django.setup()

from literature.models import Literature, LiteratureCategory
from ml_models.bert_classifier import get_bert_classifier


def init_categories():
    categories = [
        {'name': '肿瘤学', 'name_en': 'Oncology', 'description': '肿瘤相关研究'},
        {'name': '内分泌代谢', 'name_en': 'Endocrine', 'description': '内分泌和代谢疾病研究'},
        {'name': '心血管', 'name_en': 'Cardiovascular', 'description': '心血管疾病研究'},
        {'name': '其他', 'name_en': 'Other', 'description': '其他医学研究'},
        {'name': '感染性疾病', 'name_en': 'Infectious Disease', 'description': '感染性疾病研究'},
        {'name': '中医药', 'name_en': 'TCM', 'description': '中医药研究'},
        {'name': '神经内科', 'name_en': 'Neurology', 'description': '神经系统疾病研究'},
    ]
    
    for cat_data in categories:
        LiteratureCategory.objects.get_or_create(
            name=cat_data['name'],
            defaults={
                'name_en': cat_data['name_en'],
                'description': cat_data['description']
            }
        )
    
    print(f"已初始化 {len(categories)} 个分类")


def import_from_existing_dataset():
    base_path = Path(__file__).resolve().parent.parent.parent.parent
    data_dir = base_path / 'Data_Preprocessing'
    csv_path = data_dir / 'bert_train_dataset.csv'
    
    if not csv_path.exists():
        print(f"数据集不存在: {csv_path}")
        return
    
    print(f"正在从 {csv_path} 导入数据...")
    
    df = pd.read_csv(csv_path, encoding='utf-8-sig')
    print(f"共 {len(df)} 条记录")
    
    classifier = get_bert_classifier()
    
    categories = {cat.name: cat for cat in LiteratureCategory.objects.all()}
    
    imported = 0
    for idx, row in df.iterrows():
        try:
            label = row.get('label', '其他')
            category = categories.get(label)
            
            if not category:
                text = f"{row.get('text', '')}"
                pred = classifier.predict(text[:500])
                category = categories.get(pred['category_name'])
            
            literature, created = Literature.objects.get_or_create(
                title=row.get('text', '')[:200] if pd.notna(row.get('text')) else '未命名',
                defaults={
                    'abstract': str(row.get('text', ''))[200:1500] if pd.notna(row.get('text')) else '',
                    'category': category,
                    'language': 'zh'
                }
            )
            
            if created:
                imported += 1
            
            if (idx + 1) % 100 == 0:
                print(f"已处理 {idx + 1}/{len(df)} 条")
                
        except Exception as e:
            print(f"处理第 {idx + 1} 条时出错: {e}")
            continue
    
    print(f"成功导入 {imported} 条新记录")


def main():
    print("=" * 60)
    print("医学文献数据导入工具")
    print("=" * 60)
    
    print("\n1. 初始化分类...")
    init_categories()
    
    print("\n2. 导入现有数据集...")
    import_from_existing_dataset()
    
    print("\n3. 统计信息:")
    print(f"   - 分类总数: {LiteratureCategory.objects.count()}")
    print(f"   - 文献总数: {Literature.objects.count()}")
    
    print("\n" + "=" * 60)
    print("导入完成!")
    print("=" * 60)


if __name__ == '__main__':
    main()
