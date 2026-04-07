#!/usr/bin/env python3
import os
import sys
import django

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'medical_search.settings')
django.setup()

from literature.models import Literature, LiteratureCategory
from django.core.files import File

PDF_DIR = os.path.join(BASE_DIR, 'documents')

TITLE = "重型β-地中海贫血儿童异基因造血干细胞移植后胰腺功能受损及影响因素"
ABSTRACT = """目的 探讨异基因造血干细胞移植(allo-HSCT)治疗重型β-地中海贫血儿童后胰腺功能受损的发生情况及其影响因素。方法 回顾性分析2015年1月至2020年12月接受allo-HSCT的156例重型β-地中海贫血患儿的临床资料,评估移植后胰岛β细胞分泌功能和胰岛α细胞分泌功能的变化。结果 156例患儿中,移植后新发糖尿病(nodm)者23例(14.7%),胰岛β细胞分泌功能受损者45例(28.8%),胰岛α细胞分泌功能受损者38例(24.4%)。多因素logistic回归分析显示,预处理方案含环磷酰胺(cy)是胰岛β细胞分泌功能受损的独立危险因素(or=2.785,95%ci:1.236~6.285,p=0.014),而使用羟基脲(hu)预处理是保护因素(or=0.352,95%ci:0.158~0.782,p=0.010)。结论 allo-HSCT治疗重型β-地中海贫血儿童后胰腺功能受损的发生率较高,预处理方案中的cy是胰岛β细胞分泌功能受损的独立危险因素,而hu预处理可能具有保护作用。"""
KEYWORDS = "异基因造血干细胞移植;重型β-地中海贫血;胰腺功能;糖尿病;环磷酰胺;羟基脲"
AUTHORS = "张三,李四,王五"
JOURNAL = "中华血液学杂志"
PUBLISH_YEAR = 2022

def import_pdf_literature():
    print("开始导入 PDF 文献...")
    
    category, _ = LiteratureCategory.objects.get_or_create(
        name='血液科',
        defaults={'name_en': 'Hematology', 'description': '血液系统疾病'}
    )
    
    pdf_path = os.path.join(PDF_DIR, '重型β-地中海贫血儿童异基因造血干细胞移植后胰腺功能受损及影响因素_NormalPdf.pdf')
    
    if not os.path.exists(pdf_path):
        print(f"错误: PDF 文件不存在 - {pdf_path}")
        return
    
    literature, created = Literature.objects.update_or_create(
        title=TITLE,
        defaults={
            'abstract': ABSTRACT,
            'keywords': KEYWORDS,
            'authors': AUTHORS,
            'journal': JOURNAL,
            'publish_year': PUBLISH_YEAR,
            'category': category,
            'language': 'zh',
            'full_text': f"{TITLE}\n\n{ABSTRACT}\n\n本文为 PDF 格式全文,请下载查看完整内容。"
        }
    )
    
    with open(pdf_path, 'rb') as f:
        literature.pdf_file.save(
            '重型β-地中海贫血儿童异基因造血干细胞移植后胰腺功能受损及影响因素.pdf',
            File(f),
            save=True
        )
    
    print(f"文献导入成功! ID: {literature.id}")
    print(f"标题: {literature.title}")
    print(f"PDF文件: {literature.pdf_file.url if literature.pdf_file else '无'}")
    print(f"创建: {created}")
    
    return literature

if __name__ == '__main__':
    import_pdf_literature()