# EasySearch

EasySearch: 工具搜索引擎，支持通过服务知识库（JSON）检索服务访问路径、页面组件与决策执行按钮。

## 功能

- 知识库字段：`service_id`、`service_name`、`aliases`、`service_intro`、`route`
- 知识库处理：倒排索引（BM25）+ `qwen3.7-text-embedding` 向量化
- 查询优化：Query 向量化；当用户历史查询 > 10 时，使用 DIN 风格历史序列注意力优化向量
- 混合检索打分：`score = 0.6*向量相似度 + 0.3*BM25 + 0.1*热门性`
- 首次召回 Top-20 后，使用 `qwen3-vl-rerank` + `Qwen3-VL-plus` 风格理由生成进行重排
- 最终返回 Top-10
- 首页下拉默认项：最近三次历史搜索词、最近三次历史点击服务、全局最热门三个服务

## 快速使用

```python
import os

from easysearch import DashScopeClient, ServiceSearchEngine

# 留白 API Key：由部署环境注入
os.environ["DASHSCOPE_API_KEY"] = ""

engine = ServiceSearchEngine(dashscope_client=DashScopeClient())
engine.upload_knowledge_base_from_json("services.json")

results = engine.search(user_id="u-1", query="订单审批")
for item in results:
    print(item["service_name"], item["route"], item["component"], item["decision_button"])

engine.record_click("u-1", results[0]["service_id"])
print(engine.homepage_dropdown("u-1"))
```

## Qwen API 请求格式

### qwen3.7-text-embedding

```bash
curl --location 'https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding' \
--header "Authorization: ******" \
--header 'Content-Type: application/json' \
--data '{
  "model": "qwen3.7-text-embedding",
  "input": {
    "texts": ["衣服的质量杠杠的，很漂亮，不枉我等了这么久啊，喜欢，以后还来这里买"]
  }
}'
```

### qwen3-vl-rerank

```bash
curl --location 'https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank' \
--header "Authorization: ******" \
--header 'Content-Type: application/json' \
--data '{
  "model": "qwen3-vl-rerank",
  "input": {
    "query": "什么是文本排序模型",
    "documents": [
      "文本排序模型广泛用于搜索引擎和推荐系统中，它们根据文本相关性对候选文本进行排序",
      "量子计算是计算科学的一个前沿领域",
      "预训练语言模型的发展给文本排序模型带来了新的进展"
    ]
  },
  "parameters": {
    "return_documents": true,
    "top_n": 5
  }
}'
```

### qwen3-vl-plus

```bash
curl --location 'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions' \
--header "Authorization: ******" \
--header 'Content-Type: application/json' \
--data '{
  "model": "qwen3-vl-plus",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "请比较Query与Top-20候选内容，并返回每个service_id的排序理由(JSON数组)"
        }
      ]
    }
  ],
  "stream": false,
  "enable_thinking": true,
  "thinking_budget": 81920
}'
```

## 测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
