from Bio import Entrez
import pandas as pd
import time

Entrez.email = "364574103@qq.com"

def fetch_chinese_articles(keyword, max_results=50):
    """检索并提取含中文内容的文献"""
    # 检索式：主题词 + 中国学者 + 中文语言
    query = f"({keyword}[Title/Abstract]) AND (China[Affiliation] OR chinese[Language])"

    print("正在检索...")
    handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
    record = Entrez.read(handle)
    pmid_list = record["IdList"]
    handle.close()
    print(f"找到 {len(pmid_list)} 篇文献")

    articles = []
    for idx, pmid in enumerate(pmid_list):
        print(f"正在处理 {idx+1}/{len(pmid_list)}: {pmid}")
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
            records = Entrez.read(handle)
            handle.close()

            article = records["PubmedArticle"][0]
            medline = article["MedlineCitation"]
            art = medline["Article"]

            # 标题
            title = art["ArticleTitle"]

            # 摘要：尝试提取中文部分
            abstract_texts = art.get("Abstract", {}).get("AbstractText", [])
            abstract_cn = ""
            abstract_en = ""
            for text in abstract_texts:
                label = text.attributes.get("Label", "")
                if "中文" in label or "chi" in label.lower():
                    abstract_cn = str(text)
                elif not label:  # 无标签的通常为英文
                    abstract_en = str(text)
            # 优先使用中文摘要
            abstract = abstract_cn if abstract_cn else abstract_en

            # 关键词（从KeywordList或MeSH中尝试找中文）
            keywords = []
            kw_list = medline.get("KeywordList", [])
            if kw_list:
                for kw in kw_list[0]:
                    keywords.append(str(kw))

            mesh_terms = []
            for mesh in art.get("MeshHeadingList", []):
                desc = mesh.get("DescriptorName", "")
                mesh_terms.append(str(desc))

            # 出版信息
            journal = art.get("Journal", {}).get("Title", "")
            pub_date = art.get("Journal", {}).get("JournalIssue", {}).get("PubDate", {})
            year = pub_date.get("Year", "")

            articles.append({
                "PMID": pmid,
                "标题": title,
                "摘要": abstract,
                "关键词": "; ".join(keywords),
                "主题词": "; ".join(mesh_terms),
                "期刊": journal,
                "年份": year
            })

            time.sleep(0.5)  # 遵守 NCBI 限制

        except Exception as e:
            print(f"处理 {pmid} 失败: {e}")

    return articles

def main():
    # 可改为你需要的医学关键词
    keyword = "diabetes"
    articles = fetch_chinese_articles(keyword, max_results=300)
    df = pd.DataFrame(articles)
    df.to_csv("chinese_medical_from_pubmed.csv", index=False, encoding="utf-8-sig")
    print("保存完成")

if __name__ == "__main__":
    main()







