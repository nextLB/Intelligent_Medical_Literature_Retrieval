"""
中文医疗文献爬虫 - 基于PubMed API
用于爬取医学文献的标题、摘要、关键词、发表信息等
适用于毕业设计的BERT模型微调数据集构建
"""

import requests
import xml.etree.ElementTree as ET
import pandas as pd
import time
from typing import List, Dict
import os

# PubMed E-utilities API 基础URL
BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"


def safe_request(url, params, retries=3, delay=2):
    """
    带重试机制的请求函数
    """
    for attempt in range(retries):
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response
        except requests.exceptions.RequestException as e:
            print(f"请求失败 (尝试 {attempt+1}/{retries}): {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise
    return None


def search_pubmed(query: str, max_results: int = 100, api_key: str = None) -> List[str]:
    """
    在PubMed中搜索文献，返回PMID列表

    Args:
        query: 检索式
        max_results: 最大返回数量
        api_key: NCBI API密钥（可选，提高速率限制）

    Returns:
        pmid列表
    """
    search_url = f"{BASE_URL}esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json",
        "sort": "relevance"
    }
    if api_key:
        params["api_key"] = api_key

    try:
        response = safe_request(search_url, params)
        if response is None:
            return []
        data = response.json()
        return data.get("esearchresult", {}).get("idlist", [])
    except Exception as e:
        print(f"搜索失败: {query}, 错误: {e}")
        return []


def fetch_article_details(pmids: List[str], api_key: str = None) -> List[Dict]:
    """
    获取文献详细信息（标题、摘要、作者、期刊、关键词等）

    Args:
        pmids: PMID列表

    Returns:
        文献信息字典列表
    """
    if not pmids:
        return []

    fetch_url = f"{BASE_URL}efetch.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(pmids),
        "retmode": "xml",
        "rettype": "abstract"
    }
    if api_key:
        params["api_key"] = api_key

    try:
        response = safe_request(fetch_url, params)
        if response is None:
            return []
        return parse_pubmed_xml(response.text)
    except Exception as e:
        print(f"获取详情失败: {e}")
        return []


def parse_pubmed_xml(xml_content: str) -> List[Dict]:
    """
    解析PubMed XML响应，提取关键字段
    """
    articles = []
    root = ET.fromstring(xml_content)

    for article in root.findall(".//PubmedArticle"):
        article_data = {}

        # 提取PMID
        pmid_elem = article.find(".//PMID")
        article_data["pmid"] = pmid_elem.text if pmid_elem is not None else ""

        # 提取标题
        title_elem = article.find(".//ArticleTitle")
        article_data["title"] = title_elem.text if title_elem is not None else ""

        # 提取摘要（可能包含多个AbstractText，合并）
        abstract_parts = []
        for abs_elem in article.findall(".//Abstract/AbstractText"):
            # 合并文本，忽略标签属性
            abstract_parts.append("".join(abs_elem.itertext()).strip())
        article_data["abstract"] = " ".join(abstract_parts) if abstract_parts else ""

        # 提取期刊
        journal_elem = article.find(".//Journal/Title")
        article_data["journal"] = journal_elem.text if journal_elem is not None else ""

        # 提取发表年份
        year_elem = article.find(".//PubDate/Year")
        if year_elem is None:
            year_elem = article.find(".//PubDate/MedlineDate")
        article_data["year"] = year_elem.text[:4] if year_elem is not None else ""

        # 提取作者列表
        authors = []
        for author in article.findall(".//Author"):
            last = author.find("LastName")
            fore = author.find("ForeName")
            if last is not None:
                author_name = last.text
                if fore is not None:
                    author_name = f"{fore.text} {last.text}"
                authors.append(author_name)
        article_data["authors"] = "; ".join(authors[:5])  # 只取前5位

        # 提取关键词（MeSH词）
        mesh_terms = []
        for mesh in article.findall(".//MeshHeading/DescriptorName"):
            mesh_terms.append(mesh.text)
        article_data["keywords"] = "; ".join(mesh_terms[:10])

        # 提取文献类型
        pub_types = []
        for pt in article.findall(".//PublicationType"):
            pub_types.append(pt.text)
        article_data["publication_type"] = "; ".join(pub_types)

        # 提取DOI
        doi_elem = article.find(".//ArticleId[@IdType='doi']")
        article_data["doi"] = doi_elem.text if doi_elem is not None else ""

        # 提取语言
        lang_elem = article.find(".//Language")
        article_data["language"] = lang_elem.text if lang_elem is not None else ""

        # 判断是否为中文相关（作者机构含中国 或 语言为中文）
        is_chinese = False
        # 检查机构
        for aff in article.findall(".//Affiliation"):
            if aff.text and ("China" in aff.text or "中国" in aff.text):
                is_chinese = True
                break
        # 检查语言
        if article_data["language"] and "chi" in article_data["language"].lower():
            is_chinese = True
        article_data["is_chinese_related"] = is_chinese

        articles.append(article_data)

    return articles


def build_dataset(keywords: List[str], max_per_keyword: int = 150, api_key: str = None) -> pd.DataFrame:
    """
    构建文献数据集主函数

    Args:
        keywords: 关键词列表
        max_per_keyword: 每个关键词最大采集数量
        api_key: NCBI API密钥

    Returns:
        包含文献数据的DataFrame
    """
    all_articles = []

    for keyword in keywords:
        print(f"正在搜索: {keyword}")

        # 构建检索式：关键词 + 有摘要 + (中国作者机构 OR 中文语言)
        # 对关键词加双引号，避免空格问题
        query = f'("{keyword}") AND (hasabstract[text]) AND (China[Affiliation] OR Chinese[Language])'

        # 获取PMID列表
        pmids = search_pubmed(query, max_results=max_per_keyword, api_key=api_key)
        print(f"  找到 {len(pmids)} 篇文献")

        # 分批获取详情（API限制：每次最多200个ID）
        batch_size = 200
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i+batch_size]
            articles = fetch_article_details(batch, api_key=api_key)
            all_articles.extend(articles)
            print(f"  已获取 {len(all_articles)} 篇")

            # 遵守NCBI速率限制（每秒3次，有API密钥可到10次）
            time.sleep(0.34)

    # 转换为DataFrame
    df = pd.DataFrame(all_articles)

    # 过滤掉空摘要和空标题的文献
    if len(df) > 0:
        # 摘要长度大于50
        df = df[df["abstract"].str.len() > 50]
        # 标题不为空
        df = df[df["title"].notna() & (df["title"].str.strip() != "")]

    # 添加主题分类标签
    if len(df) > 0:
        df["topic"] = df.apply(lambda x: classify_topic(x["title"], x["abstract"]), axis=1)

    return df


def classify_topic(title, abstract):
    """
    简单的主题分类函数
    处理缺失值（NaN）转为空字符串
    """
    # 处理NaN
    title = str(title) if pd.notna(title) else ""
    abstract = str(abstract) if pd.notna(abstract) else ""
    text = (title + " " + abstract).lower()

    if any(k in text for k in ["cancer", "tumor", "carcinoma", "肿瘤", "癌"]):
        return "肿瘤学"
    elif any(k in text for k in ["diabetes", "diabetic", "糖尿病"]):
        return "内分泌代谢"
    elif any(k in text for k in ["heart", "cardio", "cardiovascular", "心脏", "心血管"]):
        return "心血管"
    elif any(k in text for k in ["stroke", "cerebral", "脑卒中", "脑血管"]):
        return "神经内科"
    elif any(k in text for k in ["infection", "bacterial", "virus", "感染", "病毒"]):
        return "感染性疾病"
    elif any(k in text for k in ["traditional chinese medicine", "tcm", "herb", "中药", "中医"]):
        return "中医药"
    else:
        return "其他"


def save_dataset(df: pd.DataFrame, filename: str = "medical_literature_dataset.csv"):
    """
    保存数据集为CSV文件
    """
    os.makedirs("data", exist_ok=True)
    filepath = os.path.join("data", filename)
    df.to_csv(filepath, index=False, encoding="utf-8-sig")
    print(f"数据集已保存至: {filepath}")
    print(f"共 {len(df)} 条记录")
    print(f"字段: {list(df.columns)}")


# ========== 主程序入口 ==========
if __name__ == "__main__":
    # 可选：申请NCBI API密钥提高速率限制（免费）
    # 申请地址：https://account.ncbi.nlm.nih.gov/signup/
    NCBI_API_KEY = ""  # 留空即可，无需密钥

    # 定义检索关键词（覆盖多个医学领域）
    search_keywords = [
        "cancer", "diabetes", "hypertension",
        "stroke", "coronary heart disease",
        "chronic kidney disease", "COPD",
        "traditional Chinese medicine", "COVID-19",
        "liver cirrhosis", "gastric cancer"
    ]

    print("开始构建中文医疗文献数据集...")
    print(f"关键词列表: {search_keywords}")

    # 构建数据集（每个关键词最多采集150篇，可根据需要调整）
    dataset = build_dataset(search_keywords, max_per_keyword=150, api_key=NCBI_API_KEY)

    if len(dataset) == 0:
        print("未获取到任何文献，请检查网络或稍后重试。")
    else:
        # 统计信息
        print("\n=== 数据集统计 ===")
        print(f"总文献数: {len(dataset)}")
        print(f"中文相关文献数: {dataset['is_chinese_related'].sum()}")
        print(f"\n主题分布:\n{dataset['topic'].value_counts()}")

        # 保存数据集
        save_dataset(dataset, "medical_literature_dataset.csv")

        # 打印样例
        print("\n=== 数据样例 ===")
        print(dataset[["title", "journal", "year", "topic"]].head(10))
