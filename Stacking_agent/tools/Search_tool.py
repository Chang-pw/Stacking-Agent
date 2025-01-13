import requests
from googleapiclient.discovery import build



# 配置你的 API 密钥和搜索引擎 ID
API_KEY = 'AIzaSyAfgwkQH_1JmKqVGtuIu--S7KkCTNTaDtg'
CX = 'b0e0c6e1be75a4e52'

# 初始化搜索服务
service = build("customsearch", "v1", developerKey=API_KEY)

# 查询搜索
query = "Python 教程"
res = service.cse().list(q=query, cx=CX).execute()

# 打印结果
for item in res.get('items', []):
    print(f"标题: {item['title']}")
    print(f"链接: {item['link']}")
    print(f"摘要: {item['snippet']}\n")
