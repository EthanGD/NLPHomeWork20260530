# test_api.py
import requests

BASE_URL = "http://ethanchenyansong:5000/api"
BASE_URL = "http://127.0.0.1:5000/api"

# 添加
resp = requests.post(f"{BASE_URL}/add", json={
    "question": "什么是机器学习？？555",
    "answer": "机器学习是人工智能的一个分支..."
})
print("添加:", resp.status_code, resp.json())

# 搜索
resp = requests.post(f"{BASE_URL}/search", json={
    "question": "机器学习相关"
})
print("搜索:", resp.status_code, resp.json())


# list
resp = requests.post(f"{BASE_URL}/search", json={
    "question": "机器学习相关"
})
print("搜索:", resp.status_code, resp.json())