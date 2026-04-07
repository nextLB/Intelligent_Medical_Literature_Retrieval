#!/usr/bin/env python3
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import re
import jieba
import numpy as np
from django.db.models import Q
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.http import require_http_methods
from django.core.paginator import Paginator
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.decorators import api_view

from .models import Literature, LiteratureCategory, SearchHistory, SimilarLiterature
from .serializers import (
    LiteratureListSerializer, 
    LiteratureDetailSerializer,
    LiteratureCategorySerializer,
    SearchHistorySerializer,
    ModelEvaluationSerializer
)
from ml_models.bert_classifier import get_bert_classifier
from ml_models.text_vectorizer import get_text_vectorizer
from ml_models.similarity_search import get_similarity_search
from ml_models.summarizer import get_text_summarizer

import pandas as pd
import os


def compute_similarity(text1, text2):
    vectorizer = get_text_vectorizer()
    try:
        vec1 = vectorizer.get_vector(text1[:512])
        vec2 = vectorizer.get_vector(text2[:512])
        vec1 = vec1 / np.linalg.norm(vec1)
        vec2 = vec2 / np.linalg.norm(vec2)
        return float(np.dot(vec1, vec2))
    except:
        return 0.0


LABEL_MAPPING_REVERSE = {
    0: '肿瘤学', 1: '内分泌代谢', 2: '心血管', 3: '神经内科',
    4: '感染性疾病', 5: '中医药', 6: '呼吸系统', 7: '消化系统',
    8: '泌尿肾脏', 9: '儿科', 10: '医疗AI/技术', 11: '公共卫生',
    12: '风湿免疫', 13: '精神心理', 14: '其他'
}

PDF_DOCUMENTS = {
    999: {
        'title': '重型β-地中海贫血儿童异基因造血干细胞移植后胰腺功能受损及影响因素',
        'pdf_url': '/static/documents/重型β-地中海贫血儿童异基因造血干细胞移植后胰腺功能受损及影响因素_NormalPdf.pdf',
        'abstract': '''目的 探讨异基因造血干细胞移植(allo-HSCT)治疗重型β-地中海贫血儿童后胰腺功能受损的发生情况及其影响因素。方法 回顾性分析2015年1月至2020年12月接受allo-HSCT的156例重型β-地中海贫血患儿的临床资料,评估移植后胰岛β细胞分泌功能和胰岛α细胞分泌功能的变化。结果 156例患儿中,移植后新发糖尿病(nodm)者23例(14.7%),胰岛β细胞分泌功能受损者45例(28.8%),胰岛α细胞分泌功能受损者38例(24.4%)。多因素logistic回归分析显示,预处理方案含环磷酰胺(cy)是胰岛β细胞分泌功能受损的独立危险因素(or=2.785,95%ci:1.236~6.285,p=0.014),而使用羟基脲(hu)预处理是保护因素(or=0.352,95%ci:0.158~0.782,p=0.010)。结论 allo-HSCT治疗重型β-地中海贫血儿童后胰腺功能受损的发生率较高,预处理方案中的cy是胰岛β细胞分泌功能受损的独立危险因素,而hu预处理可能具有保护作用。''',
        'keywords': '异基因造血干细胞移植;重型β-地中海贫血;胰腺功能;糖尿病;环磷酰胺;羟基脲',
        'authors': '张三,李四,王五',
        'journal': '中华血液学杂志',
        'publish_year': 2022,
        'category': '血液科',
    },
    998: {
        'title': '中医外治法治疗糖尿病胃轻瘫的研究进展',
        'pdf_url': '/static/documents/中医外治法治疗糖尿病胃轻瘫的研究进展_NormalPdf.pdf',
        'abstract': '''糖尿病胃轻瘫是糖尿病常见的慢性并发症之一,严重影响患者的生活质量。目前西医治疗主要以促胃动力药物为主,但存在疗效不佳及不良反应等问题。中医外治法包括针灸、推拿、中药敷贴、离子导入等方法,在治疗糖尿病胃轻瘫方面显示出独特的优势。本文综述了近年来中医外治法治疗糖尿病胃轻瘫的研究进展,为临床治疗提供参考。''',
        'keywords': '糖尿病胃轻瘫;中医外治法;针灸;推拿;中药敷贴',
        'authors': '中医内科研究人员',
        'journal': '中华中医药杂志',
        'publish_year': 2023,
        'category': '中医药',
    }
}


def keyword_search_tokens(query):
    """从用户输入拆出可用于匹配的关键词（整串 + jieba），避免前缀误输入导致零命中。"""
    q = (query or '').strip()
    if not q:
        return []
    tokens = []
    if len(q) >= 2:
        tokens.append(q)
    for w in jieba.cut_for_search(q):
        w = w.strip()
        if len(w) >= 2:
            tokens.append(w)
    seen = set()
    out = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def literature_q_for_keyword_tokens(tokens):
    if not tokens:
        return Q(pk__in=[])
    combined = Q()
    for t in tokens:
        combined |= Q(title__icontains=t) | Q(abstract__icontains=t) | Q(keywords__icontains=t)
    return combined


def search_literature_db(query, limit=500):
    """在数据库 Literature 中做关键词检索（标题/摘要/关键词，OR 匹配分词结果）。"""
    tokens = keyword_search_tokens(query)
    if not tokens:
        return []
    qs = Literature.objects.filter(literature_q_for_keyword_tokens(tokens)).order_by('-updated_at')[:limit]
    results = []
    for lit in qs:
        results.append({
            'id': lit.id,
            'title': lit.title,
            'abstract': lit.abstract or '',
            'keywords': lit.keywords or '',
            'journal': lit.journal or '',
            'publish_year': lit.publish_year,
            'category': lit.category.name if lit.category else '',
            'source': 'db',
        })
    return results


def get_csv_path():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base_dir, '..', 'datasets', '数据集_2', 'medical_literature_dataset.csv')
    return csv_path


def get_csv_columns():
    return ['PMID', '标题', '摘要', '关键词', '主题词', '期刊', '年份']


def search_from_csv(query, top_k=100):
    csv_path = get_csv_path()
    
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        tokens = keyword_search_tokens(query)
        if not tokens:
            return []
        mask = pd.Series(False, index=df.index)
        for t in tokens:
            pat = re.escape(t)
            search_cols = ['title', 'abstract', 'keywords']
            col_mask = pd.Series(False, index=df.index)
            for col in search_cols:
                if col in df.columns:
                    col_mask = col_mask | df[col].astype(str).str.contains(pat, case=False, na=False, regex=True)
            mask = mask | col_mask
        results_df = df[mask].head(top_k)
        
        results = []
        for _, row in results_df.iterrows():
            title = str(row.get('title', ''))
            abstract = str(row.get('abstract', ''))
            keywords = str(row.get('keywords', ''))
            journal = str(row.get('journal', ''))
            year = row.get('year', None)
            
            results.append({
                'id': None,
                'title': title,
                'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                'text': abstract,
                'keywords': keywords,
                'journal': journal,
                'publish_year': int(year) if pd.notna(year) else None,
                'category': str(row.get('topic', '其他')),
                'label_id': 3,
                'source': 'csv',
            })
        
        return results
    except Exception as e:
        print(f"CSV检索错误: {e}")
        return []


def semantic_search_from_csv(query, top_k=20):
    csv_path = get_csv_path()
    
    if not os.path.exists(csv_path):
        return []
    
    try:
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        query_tokens = set(keyword_search_tokens(query))
        if not query_tokens and len((query or '').strip()) >= 2:
            query_tokens = {(query or '').strip()}
        
        results = []
        for idx, row in df.iterrows():
            title = str(row.get('title', ''))
            abstract = str(row.get('abstract', ''))
            text = title + ' ' + abstract
            if len(text) < 10:
                continue
            
            similarity = 0.0
            query_lower = (query or '').lower()
            text_lower = text.lower()
            
            if query_lower and query_lower in text_lower:
                similarity = 0.5 + 0.5 * (len(query_lower) / max(len(text_lower), 1))
            
            doc_tokens = {w.strip() for w in jieba.cut_for_search(text) if len(w.strip()) >= 2}
            overlap = len(query_tokens & doc_tokens)
            similarity += 0.12 * overlap
            
            for word in (query or '').split():
                if len(word) >= 2 and word.lower() in text_lower:
                    similarity += 0.1
            
            lit = Literature.objects.filter(title=title).first()
            row_id = lit.id if lit else None
            
            results.append({
                'id': row_id,
                'title': title,
                'abstract': abstract[:500] + '...' if len(abstract) > 500 else abstract,
                'text': abstract,
                'keywords': str(row.get('keywords', '')),
                'journal': str(row.get('journal', '')),
                'publish_year': int(row.get('year')) if pd.notna(row.get('year')) else None,
                'category': str(row.get('topic', '其他')),
                'label_id': 3,
                'similarity': round(similarity, 4),
                'source': 'csv',
            })
        
        results.sort(key=lambda x: x['similarity'], reverse=True)
        results = [r for r in results if r['similarity'] > 0][:top_k]
        return results
    except Exception as e:
        print(f"语义检索错误: {e}")
        return []


def index(request):
    categories = LiteratureCategory.objects.all()
    
    csv_path = get_csv_path()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            literature_count = len(df)
            
            recent_texts = df['title'].tolist()[:6] if 'title' in df.columns else []
            recent_literatures = []
            for i, title in enumerate(recent_texts):
                recent_literatures.append({
                    'id': i,
                    'title': title if len(title) <= 150 else title[:150] + '...',
                    'text': str(df.iloc[i]['abstract']) if 'abstract' in df.columns else '',
                    'category': str(df.iloc[i]['topic']) if 'topic' in df.columns else '其他',
                    'journal': str(df.iloc[i]['journal']) if 'journal' in df.columns else '',
                    'publish_year': int(df.iloc[i]['year']) if 'year' in df.columns and pd.notna(df.iloc[i]['year']) else None
                })
        except Exception as e:
            print(f"首页CSV读取错误: {e}")
            literature_count = Literature.objects.count()
            recent_literatures = Literature.objects.all()[:6]
    else:
        literature_count = Literature.objects.count()
        recent_literatures = Literature.objects.all()[:6]
    
    context = {
        'categories': categories,
        'literature_count': literature_count,
        'recent_literatures': recent_literatures,
    }
    return render(request, 'literature/index.html', context)


def literature_list(request):
    category_id = request.GET.get('category')
    query = request.GET.get('q', '').strip()
    page = int(request.GET.get('page', 1))
    
    csv_path = get_csv_path()
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            if query:
                df = df[df['title'].astype(str).str.contains(query, case=False, na=False) | 
                        df['abstract'].astype(str).str.contains(query, case=False, na=False)]
            
            total_count = len(df)
            
            page_size = 20
            start_idx = (page - 1) * page_size
            end_idx = min(start_idx + page_size, total_count)
            page_df = df.iloc[start_idx:end_idx]
            
            literatures = []
            for idx, row in page_df.iterrows():
                literatures.append({
                    'id': idx,
                    'title': row['title'] if pd.notna(row['title']) else '无标题',
                    'text': row['abstract'] if pd.notna(row['abstract']) else '',
                    'category': row['topic'] if pd.notna(row['topic']) else '其他',
                    'keywords': row['keywords'] if pd.notna(row['keywords']) else '',
                    'journal': row['journal'] if pd.notna(row['journal']) else '未知期刊',
                    'publish_year': row['year'] if pd.notna(row['year']) else None,
                    'pmid': str(row['pmid']) if pd.notna(row['pmid']) else ''
                })
            
            has_prev = page > 1
            has_next = end_idx < total_count
            num_pages = (total_count + page_size - 1) // page_size
            
        except Exception as e:
            print(f"读取CSV出错: {e}")
            literatures = []
            has_prev = False
            has_next = False
            num_pages = 1
            total_count = 0
    else:
        literatures = Literature.objects.all()
        if category_id:
            literatures = literatures.filter(category_id=category_id)
        
        paginator = Paginator(literatures, 20)
        page_obj = paginator.get_page(page)
        
        literatures = []
        for lit in page_obj:
            literatures.append({
                'id': lit.id,
                'title': lit.title,
                'text': lit.abstract,
                'category': lit.category.name if lit.category else '',
                'label_id': lit.category.id if lit.category else 3,
                'journal': lit.journal,
                'publish_year': lit.publish_year
            })
        
        has_prev = page_obj.has_previous()
        has_next = page_obj.has_next()
        num_pages = page_obj.paginator.num_pages
        total_count = paginator.count
    
    categories = LiteratureCategory.objects.all()
    
    lit_999 = {
        'id': 999,
        'title': '重型β-地中海贫血儿童异基因造血干细胞移植后胰腺功能受损及影响因素',
        'text': PDF_DOCUMENTS[999]['abstract'][:200] + '...',
        'category': '血液科',
        'keywords': PDF_DOCUMENTS[999]['keywords'],
        'journal': PDF_DOCUMENTS[999]['journal'],
        'publish_year': PDF_DOCUMENTS[999]['publish_year'],
    }
    lit_998 = {
        'id': 998,
        'title': '中医外治法治疗糖尿病胃轻瘫的研究进展',
        'text': PDF_DOCUMENTS[998]['abstract'][:200] + '...',
        'category': '中医药',
        'keywords': PDF_DOCUMENTS[998]['keywords'],
        'journal': PDF_DOCUMENTS[998]['journal'],
        'publish_year': PDF_DOCUMENTS[998]['publish_year'],
    }
    literatures.insert(0, lit_999)
    literatures.insert(1, lit_998)
    total_count = total_count + 2 if total_count else 2
    
    context = {
        'literatures': literatures,
        'categories': categories,
        'current_category': int(category_id) if category_id else None,
        'has_prev': has_prev,
        'has_next': has_next,
        'current_page': page,
        'num_pages': num_pages,
        'total_count': total_count,
        'query': query
    }
    return render(request, 'literature/literature_list.html', context)


def literature_detail(request, pk):
    from django.http import Http404
    
    csv_path = get_csv_path()
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            if pk < len(df):
                row = df.iloc[pk]
                literature_data = {
                    'id': pk,
                    'title': row['title'] if pd.notna(row['title']) else '无标题',
                    'abstract': row['abstract'] if pd.notna(row['abstract']) else '暂无摘要',
                    'text': row['abstract'] if pd.notna(row['abstract']) else '',
                    'category': row['topic'] if pd.notna(row['topic']) else '其他',
                    'keywords': row['keywords'] if pd.notna(row['keywords']) else '',
                    'journal': row['journal'] if pd.notna(row['journal']) else '未知期刊',
                    'publish_year': row['year'] if pd.notna(row['year']) else None,
                    'pmid': str(row['pmid']) if pd.notna(row['pmid']) else '',
                    'authors': row['authors'] if pd.notna(row['authors']) else '',
                    'doi': row['doi'] if pd.notna(row['doi']) else '',
                    'language': 'zh',
                }
                return render(request, 'literature/literature_detail.html', {
                    'literature': literature_data,
                    'similar_literatures': [],
                    'from_csv': True,
                })
        except Exception as e:
            print(f"读取CSV详情出错: {e}")
    
    pk_int = int(pk)
    if pk_int == 999:
        lit_info = {
            'id': 999,
            'title': '重型β-地中海贫血儿童异基因造血干细胞移植后胰腺功能受损及影响因素',
            'abstract': PDF_DOCUMENTS[999]['abstract'],
            'text': PDF_DOCUMENTS[999]['abstract'],
            'category': '血液科',
            'keywords': PDF_DOCUMENTS[999]['keywords'],
            'journal': PDF_DOCUMENTS[999]['journal'],
            'publish_year': PDF_DOCUMENTS[999]['publish_year'],
            'authors': PDF_DOCUMENTS[999]['authors'],
            'pmid': 'PMID123456',
            'doi': '',
            'language': 'zh',
        }
        pdf_info = PDF_DOCUMENTS[999]
        return render(request, 'literature/literature_detail.html', {
            'literature': lit_info,
            'similar_literatures': [],
            'from_csv': False,
            'show_pdf': True,
            'pdf_url': pdf_info['pdf_url'],
        })
    
    try:
        literature = Literature.objects.get(pk=pk)
        similar_literatures = SimilarLiterature.objects.filter(source=literature).order_by('-similarity_score')[:5]
    except Literature.DoesNotExist:
        raise Http404("文献不存在")
    
    pdf_url = None
    show_pdf = False
    if literature.pdf_file:
        pdf_url = literature.pdf_file.url
        show_pdf = True
    
    return render(request, 'literature/literature_detail.html', {
        'literature': literature,
        'similar_literatures': similar_literatures,
        'from_csv': False,
        'pdf_url': pdf_url,
        'show_pdf': show_pdf,
    })


@require_http_methods(["GET", "POST"])
def keyword_search(request):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get('query', '').strip()
            page = int(data.get('page', 1))
        except:
            query = request.GET.get('q', '').strip()
            page = int(request.GET.get('page', 1))
    else:
        query = request.GET.get('q', '').strip()
        page = int(request.GET.get('page', 1))
    
    if not query:
        return JsonResponse({'error': '请输入搜索词'}, status=400)
    
    print(f"\n执行关键词检索: '{query}'")
    
    db_results = search_literature_db(query)
    csv_results = search_from_csv(query)
    seen_ids = set()
    merged = []
    for r in db_results:
        rid = r.get('id')
        if rid is not None and rid not in seen_ids:
            seen_ids.add(rid)
            merged.append(r)
    titles_in_merged = {m['title'] for m in merged}
    for r in csv_results:
        rid = r.get('id')
        if rid is not None:
            if rid in seen_ids:
                continue
            seen_ids.add(rid)
            merged.append(r)
        else:
            if r['title'] in titles_in_merged:
                continue
            titles_in_merged.add(r['title'])
            merged.append(r)
    
    results = merged
    paginator_page_size = 20
    total = len(results)
    total_pages = (total + paginator_page_size - 1) // paginator_page_size if total else 0
    start_idx = (page - 1) * paginator_page_size
    end_idx = start_idx + paginator_page_size
    page_results = results[start_idx:end_idx]
    
    try:
        SearchHistory.objects.create(
            query=query,
            search_type='keyword',
            results_count=total,
            ip_address=request.META.get('REMOTE_ADDR')
        )
    except Exception as e:
        print(f"保存搜索历史失败: {e}")
    
    return JsonResponse({
        'query': query,
        'search_type': 'keyword',
        'total': total,
        'page': page,
        'total_pages': total_pages,
        'results': page_results
    })


@require_http_methods(["POST"])
def semantic_search_view(request, query=None):
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            query = data.get('query', '')
            top_k = int(data.get('top_k', 20))
            page = int(data.get('page', 1))
        except:
            query = request.POST.get('q', '')
            top_k = int(request.GET.get('top_k', 20))
            page = 1
    else:
        query = request.GET.get('q', '')
        top_k = int(request.GET.get('top_k', 20))
        page = 1
    
    if not query:
        return JsonResponse({'error': '请输入搜索词'}, status=400)
    
    print(f"\n执行语义检索: '{query}'")
    
    results = semantic_search_from_csv(query, top_k=top_k)
    
    SearchHistory.objects.create(
        query=query,
        search_type='semantic',
        results_count=len(results),
        ip_address=request.META.get('REMOTE_ADDR')
    )
    
    return JsonResponse({
        'query': query,
        'total': len(results),
        'page': page,
        'total_pages': 1,
        'results': results
    })


@require_http_methods(["POST"])
def classify_literature(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
    except:
        text = request.POST.get('text', '')
    
    if not text:
        return JsonResponse({'error': '请输入文本'}, status=400)
    
    classifier = get_bert_classifier()
    result = classifier.predict(text)
    
    return JsonResponse(result)


@require_http_methods(["POST"])
def auto_classify_all(request):
    literatures = Literature.objects.filter(category__isnull=True)
    classifier = get_bert_classifier()
    
    results = []
    for lit in literatures:
        text = f"{lit.title} {lit.abstract}"
        pred = classifier.predict(text)
        
        category = LiteratureCategory.objects.filter(name=pred['category_name']).first()
        if category:
            lit.category = category
            lit.save()
        
        results.append({
            'id': lit.id,
            'title': lit.title[:50],
            'category': pred['category_name'],
            'confidence': pred['confidence']
        })
    
    return JsonResponse({
        'total': len(results),
        'results': results
    })


@require_http_methods(["POST"])
def generate_summary(request):
    try:
        data = json.loads(request.body)
        text = data.get('text', '')
        literature_id = data.get('literature_id')
    except:
        text = request.POST.get('text', '')
        literature_id = request.POST.get('literature_id')
    
    if not text:
        if literature_id:
            csv_path = get_csv_path()
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path, encoding='utf-8-sig')
                    lit_id = int(literature_id)
                    if lit_id < len(df):
                        row = df.iloc[lit_id]
                        text = row['abstract'] if pd.notna(row['abstract']) else ''
                except Exception as e:
                    print(f"读取摘要出错: {e}")
                    pass
            
            if not text:
                try:
                    lit = Literature.objects.get(pk=literature_id)
                    text = lit.abstract
                except Literature.DoesNotExist:
                    pass
        
        if not text:
            return JsonResponse({'error': '请提供文本或文献ID'}, status=400)
    
    summarizer = get_text_summarizer()
    summary = summarizer.generate_abstract(text)
    keywords = summarizer.keyword_extraction(text)
    
    return JsonResponse({
        'summary': summary,
        'keywords': keywords
    })


@require_http_methods(["POST"])
def find_similar(request):
    try:
        data = json.loads(request.body)
        literature_id = data.get('literature_id')
        top_k = int(data.get('top_k', 5))
    except:
        literature_id = request.POST.get('literature_id')
        top_k = int(request.POST.get('top_k', 5))
    
    if not literature_id:
        return JsonResponse({'error': '请提供文献ID'}, status=400)
    
    csv_path = get_csv_path()
    
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            
            if literature_id >= len(df):
                return JsonResponse({'error': '文献不存在'}, status=404)
            
            target_row = df.iloc[literature_id]
            target_text = str(target_row['title']) + ' ' + str(target_row['abstract'])
            
            similarities = []
            for idx, row in df.iterrows():
                if idx == literature_id:
                    continue
                text = str(row['title']) + ' ' + str(row['abstract'])
                sim = compute_similarity(target_text, text)
                similarities.append({
                    'literature_id': idx,
                    'title': row['title'] if pd.notna(row['title']) else '无标题',
                    'abstract': str(row['abstract'])[:200] + '...' if pd.notna(row['abstract']) and len(str(row['abstract'])) > 200 else (str(row['abstract']) if pd.notna(row['abstract']) else ''),
                    'journal': row['journal'] if pd.notna(row['journal']) else '',
                    'publish_year': row['year'] if pd.notna(row['year']) else None,
                    'similarity': round(sim, 4)
                })
            
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:top_k]
            
            return JsonResponse({'similar_literatures': top_results})
            
        except Exception as e:
            return JsonResponse({'error': f'查找相似文献失败: {str(e)}'}, status=500)
    
    try:
        literature = Literature.objects.get(pk=literature_id)
    except Literature.DoesNotExist:
        return JsonResponse({'error': '文献不存在'}, status=404)
    
    search_engine = get_similarity_search()
    search_engine.add_literature(literature.id, literature.title, literature.abstract)
    
    all_literatures = Literature.objects.exclude(pk=literature_id)[:200]
    for lit in all_literatures:
        search_engine.add_literature(lit.id, lit.title, lit.abstract)
    
    similar_results = search_engine.find_similar_literatures(literature.id, top_k=top_k)
    
    results = []
    for r in similar_results:
        try:
            lit = Literature.objects.get(pk=r['literature_id'])
            results.append({
                'id': lit.id,
                'title': lit.title,
                'abstract': lit.abstract[:200] + '...' if len(lit.abstract) > 200 else lit.abstract,
                'journal': lit.journal,
                'publish_year': lit.publish_year,
                'similarity': round(r['similarity'], 4)
            })
            
            SimilarLiterature.objects.update_or_create(
                source=literature,
                similar=lit,
                defaults={'similarity_score': r['similarity']}
            )
        except Literature.DoesNotExist:
            continue
    
    return JsonResponse({
        'source': {
            'id': literature.id,
            'title': literature.title
        },
        'total': len(results),
        'similar_literatures': results
    })


class CategoryStatisticsView(APIView):
    def get(self, request):
        categories = LiteratureCategory.objects.all()
        data = []
        
        for cat in categories:
            count = Literature.objects.filter(category=cat).count()
            data.append({
                'id': cat.id,
                'name': cat.name,
                'count': count
            })
        
        return Response(data)


class SearchHistoryView(APIView):
    def get(self, request):
        histories = SearchHistory.objects.all()[:50]
        return Response(SearchHistorySerializer(histories, many=True).data)


@require_http_methods(["GET"])
def search_page(request):
    return render(request, 'literature/search.html')


@require_http_methods(["GET"])
def classification_page(request):
    categories = LiteratureCategory.objects.all()
    context = {'categories': categories}
    return render(request, 'literature/classification.html', context)


@require_http_methods(["GET"])
def evaluation_page(request):
    return render(request, 'literature/evaluation.html')


@require_http_methods(["GET"])
def data_import_page(request):
    return render(request, 'literature/data_import.html')


@api_view(['POST'])
def import_literature(request):
    try:
        data = request.data
        pmid = data.get('pmid', '')
        title = data.get('title', '')
        abstract = data.get('abstract', '')
        keywords = data.get('keywords', '')
        authors = data.get('authors', '')
        journal = data.get('journal', '')
        publish_year = data.get('publish_year')
        
        if not title:
            return Response({'error': '标题不能为空'}, status=400)
        
        literature, created = Literature.objects.update_or_create(
            pmid=pmid if pmid else None,
            defaults={
                'title': title,
                'abstract': abstract,
                'keywords': keywords,
                'authors': authors,
                'journal': journal,
                'publish_year': publish_year
            }
        )
        
        classifier = get_bert_classifier()
        text = f"{title} {abstract}"
        pred = classifier.predict(text)
        
        category = LiteratureCategory.objects.filter(name=pred['category_name']).first()
        if category:
            literature.category = category
            literature.save()
        
        return Response({
            'id': literature.id,
            'category': pred['category_name'],
            'created': created
        })
    except Exception as e:
        return Response({'error': str(e)}, status=500)


@require_http_methods(["GET"])
def training_page(request):
    categories = LiteratureCategory.objects.all()
    context = {'categories': categories}
    return render(request, 'literature/training.html', context)


@require_http_methods(["POST"])
def train_model(request):
    import os
    import sys
    import torch
    import random
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertModel
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE_DIR)
    
    LABEL_MAPPING_REVERSE = {
        0: '肿瘤学', 1: '内分泌代谢', 2: '心血管', 3: '神经内科',
        4: '感染性疾病', 5: '中医药', 6: '呼吸系统', 7: '消化系统',
        8: '泌尿肾脏', 9: '儿科', 10: '医疗AI/技术', 11: '公共卫生',
        12: '风湿免疫', 13: '精神心理', 14: '其他'
    }
    
    try:
        data = json.loads(request.body)
        num_epochs = int(data.get('num_epochs', 5))
        batch_size = int(data.get('batch_size', 16))
        learning_rate = float(data.get('learning_rate', 2e-5))
        max_length = int(data.get('max_length', 256))
    except:
        num_epochs = 5
        batch_size = 16
        learning_rate = 2e-5
        max_length = 256
    
    print("=" * 60)
    print("开始BERT模型训练")
    print("=" * 60)
    print(f"训练参数: epochs={num_epochs}, batch_size={batch_size}, lr={learning_rate}, max_length={max_length}")
    
    try:
        csv_path = os.path.join(BASE_DIR, '..', 'Data_Preprocessing', 'bert_train_dataset.csv')
        print(f"加载数据: {csv_path}")
        
        if not os.path.exists(csv_path):
            return JsonResponse({'error': f'训练数据不存在: {csv_path}'}, status=400)
        
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        print(f"数据加载完成，共 {len(df)} 条记录")
        
        if len(df) == 0:
            return JsonResponse({'error': '训练数据为空'}, status=400)
        
        texts = df['text'].tolist()
        labels = df['label_id'].tolist()
        
        print("\n标签分布:")
        for label_id in sorted(set(labels)):
            count = labels.count(label_id)
            print(f"  {LABEL_MAPPING_REVERSE.get(label_id, '未知')}: {count}")
        
        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.15, random_state=seed, stratify=labels
        )
        
        print(f"\n训练集: {len(train_texts)} 条, 验证集: {len(val_texts)} 条")
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
        
        print("\n加载BERT分词器...")
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        
        class MedicalDataset(Dataset):
            def __init__(self, texts, labels, tokenizer, max_length):
                self.texts = texts
                self.labels = labels
                self.tokenizer = tokenizer
                self.max_length = max_length
            
            def __len__(self):
                return len(self.texts)
            
            def __getitem__(self, idx):
                text = str(self.texts[idx])
                label = self.labels[idx]
                encoding = self.tokenizer(
                    text, add_special_tokens=True, max_length=self.max_length,
                    padding='max_length', truncation=True, return_tensors='pt'
                )
                return {
                    'input_ids': encoding['input_ids'].squeeze(0),
                    'attention_mask': encoding['attention_mask'].squeeze(0),
                    'label': torch.tensor(label, dtype=torch.long)
                }
        
        train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, max_length)
        val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, max_length)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        class BertClassifier(nn.Module):
            def __init__(self, num_classes=15, dropout=0.3):
                super(BertClassifier, self).__init__()
                self.bert = BertModel.from_pretrained('bert-base-chinese')
                self.dropout = nn.Dropout(dropout)
                self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
            
            def forward(self, input_ids, attention_mask):
                outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                pooled_output = outputs.pooler_output
                pooled_output = self.dropout(pooled_output)
                return self.classifier(pooled_output)
        
        print("\n初始化模型...")
        model = BertClassifier(num_classes=15, dropout=0.3)
        model = model.to(device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"模型参数: 总数={total_params:,}, 可训练={trainable_params:,}")
        
        criterion = nn.CrossEntropyLoss()
        optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
        
        total_steps = len(train_loader) * num_epochs
        warmup_steps = int(total_steps * 0.1)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
        
        best_val_acc = 0.0
        training_history = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        print("\n" + "=" * 60)
        print("开始训练")
        print("=" * 60)
        
        for epoch in range(num_epochs):
            model.train()
            total_loss = 0
            correct = 0
            total = 0
            
            print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
            
            for i, batch in enumerate(train_loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].to(device)
                
                optimizer.zero_grad()
                outputs = model(input_ids, attention_mask)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                if (i + 1) % 10 == 0 or i == 0:
                    batch_loss = loss.item()
                    batch_acc = correct / total
                    progress = (i + 1) / len(train_loader) * 100
                    print(f"  Step {i+1}/{len(train_loader)} [{progress:.1f}%] - Loss: {batch_loss:.4f} - Acc: {batch_acc:.4f}")
            
            train_loss = total_loss / len(train_loader)
            train_acc = correct / total
            
            print(f"\n验证中...")
            model.eval()
            val_loss = 0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / len(val_loader)
            val_acc = correct / total
            
            print(f"Epoch {epoch+1} 完成: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")
            
            training_history['epochs'].append(epoch + 1)
            training_history['train_loss'].append(round(train_loss, 4))
            training_history['train_acc'].append(round(train_acc, 4))
            training_history['val_loss'].append(round(val_loss, 4))
            training_history['val_acc'].append(round(val_acc, 4))
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
                os.makedirs(checkpoint_dir, exist_ok=True)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': {
                        'num_epochs': num_epochs,
                        'batch_size': batch_size,
                        'learning_rate': learning_rate,
                        'max_length': max_length
                    }
                }, os.path.join(checkpoint_dir, 'best_model.pt'))
                print(f"*** 保存最佳模型, Val Acc={val_acc:.4f} ***")
        
        print("\n" + "=" * 60)
        print(f"训练完成! 最佳验证准确率: {best_val_acc:.4f}")
        print("=" * 60)
        
        return JsonResponse({
            'status': 'success',
            'message': '模型训练完成',
            'best_val_acc': round(best_val_acc, 4),
            'history': training_history,
            'train_samples': len(train_texts),
            'val_samples': len(val_texts),
            'device': str(device)
        })
        
    except Exception as e:
        import traceback
        print(f"训练出错: {str(e)}")
        print(traceback.format_exc())
        return JsonResponse({'error': str(e), 'trace': traceback.format_exc()}, status=500)


@require_http_methods(["GET"])
def get_training_history(request):
    import os
    history_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'BERT', 'logs', 'training_history.json')
    if os.path.exists(history_path):
        with open(history_path, 'r') as f:
            return JsonResponse(json.load(f))
    return JsonResponse({'message': '暂无训练历史'})


@require_http_methods(["POST"])
def train_model_stream(request):
    import os
    import sys
    import time
    import torch
    import random
    import numpy as np
    import pandas as pd
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import BertTokenizer, get_linear_schedule_with_warmup, BertModel
    from torch.optim import AdamW
    from sklearn.model_selection import train_test_split
    from django.http import StreamingHttpResponse
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, BASE_DIR)
    
    LABEL_MAP = {
        0: '肿瘤学', 1: '内分泌代谢', 2: '心血管', 3: '神经内科',
        4: '感染性疾病', 5: '中医药', 6: '呼吸系统', 7: '消化系统',
        8: '泌尿肾脏', 9: '儿科', 10: '医疗AI/技术', 11: '公共卫生',
        12: '风湿免疫', 13: '精神心理', 14: '其他'
    }
    
    try:
        data = json.loads(request.body)
        num_epochs = int(data.get('num_epochs', 3))
        batch_size = int(data.get('batch_size', 16))
        learning_rate = float(data.get('learning_rate', 2e-5))
        max_length = int(data.get('max_length', 256))
    except:
        num_epochs = 3
        batch_size = 16
        learning_rate = 2e-5
        max_length = 256
    
    def send_event(event_type, event_data):
        return "data: " + json.dumps({'type': event_type, 'data': event_data}) + "\n\n"
    
    lock_file = os.path.join(BASE_DIR, 'checkpoints', 'training.lock')
    training_stop_flag = os.path.join(BASE_DIR, 'checkpoints', 'training_stop.flag')
    
    if os.path.exists(lock_file):
        return StreamingHttpResponse(
            iter([send_event('error', '已有训练任务正在进行，请等待完成或重启服务'),
                  send_event('complete', {'error': '训练任务冲突'})]),
            content_type='text/event-stream'
        )
    
    os.makedirs(os.path.dirname(lock_file), exist_ok=True)
    with open(lock_file, 'w') as f:
        f.write(str(os.getpid()))
    
    if os.path.exists(training_stop_flag):
        os.remove(training_stop_flag)
    
    def generate():
        try:
            yield send_event('log', '=' * 60)
            yield send_event('log', '开始BERT模型训练')
            yield send_event('log', '=' * 60)
            yield send_event('info', '训练参数: epochs=%d, batch_size=%d, lr=%s, max_length=%d' % (num_epochs, batch_size, learning_rate, max_length))
            
            csv_path = os.path.join(BASE_DIR, '..', 'Data_Preprocessing', 'bert_train_dataset.csv')
            yield send_event('log', '加载数据: ' + csv_path)
            
            if not os.path.exists(csv_path):
                yield send_event('error', '训练数据不存在: ' + csv_path)
                yield send_event('complete', {'error': '训练数据不存在'})
                return
            
            df = pd.read_csv(csv_path, encoding='utf-8-sig')
            yield send_event('log', '数据加载完成，共 %d 条记录' % len(df))
            
            if len(df) == 0:
                yield send_event('error', '训练数据为空')
                yield send_event('complete', {'error': '训练数据为空'})
                return
            
            texts = df['text'].tolist()
            labels = df['label_id'].tolist()
            
            yield send_event('log', '标签分布:')
            for label_id in sorted(set(labels)):
                count = labels.count(label_id)
                yield send_event('log', '  %s: %d' % (LABEL_MAP.get(label_id, '未知'), count))
            
            seed = 42
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            train_texts, val_texts, train_labels, val_labels = train_test_split(
                texts, labels, test_size=0.15, random_state=seed, stratify=labels
            )
            
            yield send_event('log', '训练集: %d 条, 验证集: %d 条' % (len(train_texts), len(val_texts)))
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            yield send_event('info', '使用设备: ' + str(device))
            if torch.cuda.is_available():
                yield send_event('info', 'GPU: ' + torch.cuda.get_device_name(0))
            
            yield send_event('log', '加载BERT分词器...')
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', local_files_only=True)
            
            yield send_event('log', '加载BERT模型...')
            
            class MedicalDataset(Dataset):
                def __init__(self, texts, labels, tokenizer, max_length):
                    self.texts = texts
                    self.labels = labels
                    self.tokenizer = tokenizer
                    self.max_length = max_length
                
                def __len__(self):
                    return len(self.texts)
                
                def __getitem__(self, idx):
                    text = str(self.texts[idx])
                    label = self.labels[idx]
                    encoding = self.tokenizer(
                        text, add_special_tokens=True, max_length=self.max_length,
                        padding='max_length', truncation=True, return_tensors='pt'
                    )
                    return {
                        'input_ids': encoding['input_ids'].squeeze(0),
                        'attention_mask': encoding['attention_mask'].squeeze(0),
                        'label': torch.tensor(label, dtype=torch.long)
                    }
            
            train_dataset = MedicalDataset(train_texts, train_labels, tokenizer, max_length)
            val_dataset = MedicalDataset(val_texts, val_labels, tokenizer, max_length)
            
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
            
            class BertClassifier(nn.Module):
                def __init__(self, num_classes=15, dropout=0.3):
                    super(BertClassifier, self).__init__()
                    self.bert = BertModel.from_pretrained('bert-base-chinese', local_files_only=True)
                    self.dropout = nn.Dropout(dropout)
                    self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
                
                def forward(self, input_ids, attention_mask):
                    outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
                    pooled_output = outputs.pooler_output
                    pooled_output = self.dropout(pooled_output)
                    return self.classifier(pooled_output)
            
            yield send_event('log', '初始化模型...')
            model = BertClassifier(num_classes=15, dropout=0.3)
            model = model.to(device)
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            yield send_event('info', '模型参数: 总数=%d, 可训练=%d' % (total_params, trainable_params))
            
            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            
            total_steps = len(train_loader) * num_epochs
            warmup_steps = int(total_steps * 0.1)
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
            
            best_val_acc = 0.0
            training_history = {'epochs': [], 'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
            
            yield send_event('log', '=' * 60)
            yield send_event('log', '开始训练')
            yield send_event('log', '=' * 60)
            
            for epoch in range(num_epochs):
                model.train()
                total_loss = 0
                correct = 0
                total = 0
                
                epoch_start = time.time()
                yield send_event('log', '--- Epoch %d/%d ---' % (epoch+1, num_epochs))
                
                for i, batch in enumerate(train_loader):
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['label'].to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(input_ids, attention_mask)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    
                    total_loss += loss.item()
                    _, predicted = torch.max(outputs, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
                    
                    if (i + 1) % 10 == 0 or i == 0:
                        batch_loss = loss.item()
                        batch_acc = correct / total
                        progress = (i + 1) / len(train_loader) * 100
                        overall_progress = (epoch * 100 + progress) / num_epochs
                        yield send_event('progress', {
                            'epoch': epoch + 1,
                            'total_epochs': num_epochs,
                            'step': i + 1,
                            'total_steps': len(train_loader),
                            'progress': round(overall_progress, 2),
                            'loss': round(batch_loss, 4),
                            'acc': round(batch_acc, 4)
                        })
                        yield send_event('log', '  Step %d/%d [%.1f%%] - Loss: %.4f - Acc: %.4f' % (i+1, len(train_loader), progress, batch_loss, batch_acc))
                
                train_loss = total_loss / len(train_loader)
                train_acc = correct / total
                
                yield send_event('log', '验证中...')
                model.eval()
                val_loss = 0
                correct_val = 0
                total_val = 0
                
                with torch.no_grad():
                    for batch in val_loader:
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        labels = batch['label'].to(device)
                        
                        outputs = model(input_ids, attention_mask)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item()
                        _, predicted = torch.max(outputs, 1)
                        total_val += labels.size(0)
                        correct_val += (predicted == labels).sum().item()
                
                val_loss = val_loss / len(val_loader)
                val_acc = correct_val / total_val
                epoch_time = time.time() - epoch_start
                
                yield send_event('log', 'Epoch %d 完成: Train Loss=%.4f, Train Acc=%.4f, Val Loss=%.4f, Val Acc=%.4f, 耗时: %.1fs' % (epoch+1, train_loss, train_acc, val_loss, val_acc, epoch_time))
                
                training_history['epochs'].append(epoch + 1)
                training_history['train_loss'].append(round(train_loss, 4))
                training_history['train_acc'].append(round(train_acc, 4))
                training_history['val_loss'].append(round(val_loss, 4))
                training_history['val_acc'].append(round(val_acc, 4))
                
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    checkpoint_dir = os.path.join(BASE_DIR, 'checkpoints')
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_acc': val_acc,
                        'config': {
                            'num_epochs': num_epochs,
                            'batch_size': batch_size,
                            'learning_rate': learning_rate,
                            'max_length': max_length
                        }
                    }, os.path.join(checkpoint_dir, 'best_model.pt'))
                    yield send_event('log', '*** 保存最佳模型, Val Acc=%.4f ***' % val_acc)
                
                if os.path.exists(training_stop_flag):
                    os.remove(training_stop_flag)
                    yield send_event('log', '训练已被用户停止')
                    yield send_event('complete', {'error': '训练已停止', 'stopped': True})
                    break
            else:
                yield send_event('log', '=' * 60)
                yield send_event('log', '训练完成! 最佳验证准确率: %.4f' % best_val_acc)
                yield send_event('log', '=' * 60)
                
                yield send_event('complete', {
                    'best_val_acc': round(best_val_acc, 4),
                    'history': training_history,
                    'train_samples': len(train_texts),
                    'val_samples': len(val_texts),
                    'device': str(device)
                })
            
            if os.path.exists(lock_file):
                os.remove(lock_file)
        
        except Exception as e:
            import traceback
            yield send_event('error', '训练出错: ' + str(e))
            yield send_event('error', traceback.format_exc())
        finally:
            if os.path.exists(lock_file):
                os.remove(lock_file)
    
    response = StreamingHttpResponse(generate(), content_type='text/event-stream')
    response['Cache-Control'] = 'no-cache'
    response['X-Accel-Buffering'] = 'no'
    return response


@require_http_methods(["POST"])
def stop_training(request):
    """停止训练"""
    import os
    
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lock_file = os.path.join(BASE_DIR, 'checkpoints', 'training.lock')
    stop_flag = os.path.join(BASE_DIR, 'checkpoints', 'training_stop.flag')
    
    try:
        # 如果锁文件存在，说明训练正在进行
        if os.path.exists(lock_file):
            # 创建停止标志文件
            with open(stop_flag, 'w') as f:
                f.write(str(time.time()))
            return JsonResponse({'status': 'success', 'message': '已发送停止信号'})
        else:
            return JsonResponse({'status': 'info', 'message': '没有正在进行的训练'})
    except Exception as e:
        return JsonResponse({'error': str(e)}, status=500)
