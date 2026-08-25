# pytest 根配置：确保项目根目录在 sys.path，使 `easysearch` / `api` 可被导入

import os

# M12：测试隔离——确保鉴权/限流中间件在测试套件中默认关闭，
# 避免开发机 .env 中的 EASYSEARCH_API_KEY/EASYSEARCH_RATE_LIMIT 干扰测试。
# 个别测试用例可在自身 setUp 中显式设置 env var 来验证鉴权/限流行为。
# 注意：用「置空」而非 pop，因为 config.py 的 load_dotenv(override=False)
# 仅对 os.environ 中不存在的键才从 .env 加载——pop 后 .env 中的值会被重新
# 加载导致鉴权意外开启；置空字符串后 load_dotenv 视其为已存在不覆盖，
# 且 ApiKeyMiddleware 的 `if not api_key` 对空串判真→鉴权关闭。
os.environ["EASYSEARCH_API_KEY"] = ""
# 限流默认禁用（=0），避免测试套件累计请求触发 429；
# 需测试限流时在 setUp 显式设 EASYSEARCH_RATE_LIMIT=2 等
os.environ["EASYSEARCH_RATE_LIMIT"] = "0"
# 缓存隔离：确保测试不连真实 Redis（配则降级内存 LRU；个别缓存测试在 setUp 显式 mock）
os.environ["REDIS_URL"] = ""
os.environ["EASYSEARCH_REDIS_URL"] = ""
# API Key 隔离：.env 中的真实 DashScope/DeepSeek Key 会通过 config.py 的
# load_dotenv(override=False) 加载到 DASHSCOPE_API_KEY/DEEPSEEK_API_KEY，
# 导致 DashScopeClient(api_key=None) 的 enabled=True（构造器回退到 env）。
# 用「置空」而非 pop：load_dotenv(override=False) 对已存在的键不覆盖，
# 空→bool("")=False→enabled=False，恢复离线模式。需测试 Key 的用例显式传入。
os.environ["DASHSCOPE_API_KEY"] = ""
os.environ["DEEPSEEK_API_KEY"] = ""
