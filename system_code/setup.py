#!/usr/bin/env python3
import os
import sys
from pathlib import Path
import django

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_search.settings')
django.setup()

from django.core.management import call_command


def main():
    print("=" * 60)
    print("医学文献智能检索系统 - 安装向导")
    print("=" * 60)
    
    print("\n[1/4] 创建数据库迁移...")
    call_command('makemigrations', 'literature', verbosity=1)
    
    print("\n[2/4] 执行数据库迁移...")
    call_command('migrate', verbosity=1)
    
    print("\n[3/4] 初始化分类数据...")
    from literature.models import LiteratureCategory
    
    categories = [
        {'name': '肿瘤学', 'name_en': 'Oncology'},
        {'name': '内分泌代谢', 'name_en': 'Endocrine'},
        {'name': '心血管', 'name_en': 'Cardiovascular'},
        {'name': '其他', 'name_en': 'Other'},
        {'name': '感染性疾病', 'name_en': 'Infectious Disease'},
        {'name': '中医药', 'name_en': 'TCM'},
        {'name': '神经内科', 'name_en': 'Neurology'},
    ]
    
    for cat in categories:
        LiteratureCategory.objects.get_or_create(
            name=cat['name'],
            defaults={'name_en': cat['name_en']}
        )
    print(f"已创建 {len(categories)} 个分类")
    
    print("\n[4/4] 收集静态文件...")
    call_command('collectstatic', '--noinput', verbosity=1)
    
    print("\n" + "=" * 60)
    print("安装完成！")
    print("=" * 60)
    print("\n启动服务器:")
    print("  python manage.py runserver")
    print("\n访问地址: http://127.0.0.1:8000")
    print("\n后台管理: http://127.0.0.1:8000/admin")


if __name__ == '__main__':
    main()
