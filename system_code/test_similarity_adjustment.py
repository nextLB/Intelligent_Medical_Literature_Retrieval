#!/usr/bin/env python3
"""
测试相似度调整逻辑
"""

def test_keyword_adjustment():
    """测试关键词检索相似度调整"""
    print("=== 关键词检索相似度调整测试 ===")
    test_cases = [
        # (原始相似度, 排名, 期望调整后范围)
        (0.8, 0, (0.1, 0.6)),   # 0.8会被限制到0.6
        (0.5, 1, (0.09, 0.59)),  # 0.5保持，减去0.01
        (None, 2, (0.27, 0.29)), # 默认值0.3-0.01*2=0.28
        (0.2, 3, (0.07, 0.17)),  # 0.2保持，减去0.03
        (0.9, 5, (0.1, 0.6)),    # 0.9被限制到0.6，再减0.05
    ]

    for raw_sim, idx, expected_range in test_cases:
        if raw_sim is not None:
            # 降低关键词检索的相似度：最大值不超过0.6，并随排名递减
            adj_sim = min(raw_sim, 0.6) - (idx * 0.01)
            adj_sim = max(adj_sim, 0.1)  # 确保最小值
        else:
            # 默认值：0.3递减
            adj_sim = 0.3 - (idx * 0.01)

        adj_sim = max(adj_sim, 0.1)  # 确保不低于0.1
        result = round(adj_sim, 4)

        if expected_range[0] <= result <= expected_range[1]:
            status = "✓"
        else:
            status = "✗"

        print(f"{status} 原始={raw_sim}, 排名={idx} -> 调整后={result} (期望范围: {expected_range})")

def test_bert_adjustment():
    """测试BERT检索相似度调整"""
    print("\n=== BERT检索相似度调整测试 ===")
    test_cases = [
        # (原始相似度, 期望调整后)
        (0.4, 0.75),   # <0.6 -> 0.75
        (0.5, 0.75),   # <0.6 -> 0.75
        (0.6, 0.75),   # 0.6 -> 0.75
        (0.7, 0.83),   # 0.7 -> 0.75 + (0.1*0.8) = 0.83
        (0.8, 0.91),   # 0.8 -> 0.75 + (0.2*0.8) = 0.91
        (0.85, 0.95),  # 0.85 -> 0.75 + (0.25*0.8) = 0.95
        (0.9, 0.99),   # 0.9 -> 0.99
        (0.95, 0.99),  # >0.9 -> 0.99
    ]

    for raw_sim, expected in test_cases:
        # 积极调整BERT相似度：映射到更高的范围 [0.75, 0.99]
        # 假设原始相似度在[0.6, 0.9]之间，映射到[0.75, 0.99]
        if raw_sim < 0.6:
            adj_sim = 0.75  # 最低值
        elif raw_sim > 0.9:
            adj_sim = 0.99  # 最高值
        else:
            # 线性映射：0.6->0.75, 0.9->0.99
            adj_sim = 0.75 + (raw_sim - 0.6) * (0.24 / 0.3)

        result = round(min(adj_sim, 0.99), 4)

        if abs(result - expected) < 0.01:
            status = "✓"
        else:
            status = "✗"

        print(f"{status} 原始={raw_sim} -> 调整后={result} (期望: {expected})")

def test_comparison():
    """测试两种方法的对比"""
    print("\n=== 方法对比测试 ===")

    # 模拟几个示例
    keyword_examples = [
        (0.8, 0),  # 调整后: 0.6
        (0.5, 1),  # 调整后: 0.49
        (None, 2), # 调整后: 0.28
    ]

    bert_examples = [
        0.4,  # 调整后: 0.75
        0.7,  # 调整后: 0.83
        0.9,  # 调整后: 0.99
    ]

    print("关键词检索示例:")
    for raw_sim, idx in keyword_examples:
        if raw_sim is not None:
            adj_sim = min(raw_sim, 0.6) - (idx * 0.01)
            adj_sim = max(adj_sim, 0.1)
        else:
            adj_sim = 0.3 - (idx * 0.01)
        adj_sim = max(adj_sim, 0.1)
        print(f"  原始={raw_sim}, 排名={idx} -> {round(adj_sim, 4)}")

    print("\nBERT检索示例:")
    for raw_sim in bert_examples:
        if raw_sim < 0.6:
            adj_sim = 0.75
        elif raw_sim > 0.9:
            adj_sim = 0.99
        else:
            adj_sim = 0.75 + (raw_sim - 0.6) * (0.24 / 0.3)
        print(f"  原始={raw_sim} -> {round(adj_sim, 4)}")

if __name__ == "__main__":
    test_keyword_adjustment()
    test_bert_adjustment()
    test_comparison()

    print("\n=== 结论 ===")
    print("1. 关键词检索相似度被限制在0.6以下，且随排名递减")
    print("2. BERT检索相似度被映射到0.75-0.99范围")
    print("3. BERT相似度将始终高于关键词相似度")