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
from easysearch import ServiceSearchEngine

engine = ServiceSearchEngine()
engine.upload_knowledge_base_from_json("services.json")

results = engine.search(user_id="u-1", query="订单审批")
for item in results:
    print(item["service_name"], item["route"], item["component"], item["decision_button"])

engine.record_click("u-1", results[0]["service_id"])
print(engine.homepage_dropdown("u-1"))
```

## 测试

```bash
python -m unittest discover -s tests -p 'test_*.py'
```
