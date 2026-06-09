#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 /api/list 分页接口
用法: python test_list_api.py
"""
import requests
import json
import sys

# ==================== 配置 ====================
BASE_URL = "http://127.0.0.1:5000/api"
TIMEOUT = 30

def print_section(title: str):
    """打印分隔标题"""
    print(f"\n{'='*20} {title} {'='*20}")

def print_response(resp: requests.Response, show_items: bool = True):
    """格式化打印响应"""
    print(f"状态码: {resp.status_code}")
    try:
        data = resp.json()
        # pretty print JSON
        if show_items and "items" in data:
            # 只显示简要信息，避免输出太长
            items_summary = [
                {"id": item["id"], "question": item["question"][:30]} 
                for item in data.get("items", [])
            ]
            data["items"] = items_summary
        print(json.dumps(data, ensure_ascii=False, indent=2))
    except:
        print(f"响应体: {resp.text[:500]}")

# ==================== 测试用例 ====================

def test_list_default():
    """测试 1: 默认参数 (page=1, limit=20)"""
    print_section("测试 1: 默认参数")
    resp = requests.get(f"{BASE_URL}/list", timeout=TIMEOUT)
    print_response(resp)
    assert resp.status_code == 200, "状态码应为 200"
    data = resp.json()
    assert "items" in data, "响应应包含 items 字段"
    assert "page" in data and "limit" in data, "响应应包含分页元数据"
    print("✅ 默认参数测试通过")

def test_list_custom_page():
    """测试 2: 自定义页码"""
    print_section("测试 2: 自定义页码 (page=2, limit=5)")
    resp = requests.get(
        f"{BASE_URL}/list",
        params={"page": 2, "limit": 5},
        timeout=TIMEOUT
    )
    print_response(resp)
    assert resp.status_code == 200
    data = resp.json()
    assert data["page"] == 2, "返回页码应与请求一致"
    assert data["limit"] == 5, "返回 limit 应与请求一致"
    assert len(data["items"]) <= 5, "返回条目数不应超过 limit"
    print("✅ 自定义页码测试通过")

def test_list_limit_boundary():
    """测试 3: limit 边界值 (最大 100)"""
    print_section("测试 3: limit 边界值")
    
    # 请求 limit=200，服务端应限制为 100
    resp = requests.get(
        f"{BASE_URL}/list",
        params={"page": 1, "limit": 200},
        timeout=TIMEOUT
    )
    print_response(resp, show_items=False)
    data = resp.json()
    assert data["limit"] == 100, "服务端应限制 limit 最大为 100"
    assert len(data["items"]) <= 100, "返回条目数不应超过 100"
    print("✅ limit 边界测试通过")

def test_list_empty_db():
    """测试 4: 空数据库"""
    print_section("测试 4: 空数据库响应")
    # 先清空测试（可选，谨慎使用）
    # requests.delete(f"{BASE_URL}/clear_all")  # 如果你有这个接口
    
    resp = requests.get(f"{BASE_URL}/list", timeout=TIMEOUT)
    print_response(resp)
    data = resp.json()
    assert data["items"] == [], "空数据库应返回空列表"
    assert data["total"] == 0, "总数应为 0"
    assert data["pages"] == 0, "总页数应为 0"
    print("✅ 空数据库测试通过")

def test_list_invalid_params():
    """测试 5: 非法参数处理"""
    print_section("测试 5: 非法参数")
    
    test_cases = [
        {"page": 0, "desc": "page=0 (应从 1 开始)"},
        {"page": -1, "desc": "page=-1 (负数)"},
        {"limit": 0, "desc": "limit=0 (无意义)"},
        {"page": "abc", "desc": "page=abc (非数字)"},
    ]
    
    for params, desc in test_cases:
        print(f"\n  测试: {desc}")
        resp = requests.get(
            f"{BASE_URL}/list",
            params=params,
            timeout=TIMEOUT
        )
        # 服务端应优雅处理，返回 200 + 空/默认结果，或 400 错误
        print(f"    状态码: {resp.status_code}")
        if resp.status_code >= 400:
            print(f"    错误: {resp.json().get('error')}")

def test_list_pagination_logic():
    """测试 6: 分页逻辑验证（需要预置数据）"""
    print_section("测试 6: 分页连续性验证")
    
    # 先获取总数
    resp = requests.get(f"{BASE_URL}/list", params={"limit": 1}, timeout=TIMEOUT)
    data = resp.json()
    total = data.get("total", 0)
    
    if total == 0:
        print("⚠️  数据库为空，跳过分页连续性测试")
        return
    
    print(f"数据库总记录数: {total}")
    
    # 获取第 1 页最后一条的 id
    resp1 = requests.get(f"{BASE_URL}/list", params={"page": 1, "limit": 5}, timeout=TIMEOUT)
    items_p1 = resp1.json()["items"]
    
    # 获取第 2 页第一条的 id
    resp2 = requests.get(f"{BASE_URL}/list", params={"page": 2, "limit": 5}, timeout=TIMEOUT)
    items_p2 = resp2.json()["items"]
    
    if items_p1 and items_p2:
        last_id_p1 = items_p1[-1]["id"]
        first_id_p2 = items_p2[0]["id"]
        print(f"第 1 页最后 id: {last_id_p1}, 第 2 页第一条 id: {first_id_p2}")
        # 由于按 id DESC 排序，第 1 页的 id 应 > 第 2 页的 id
        assert last_id_p1 > first_id_p2, "分页顺序应连续（按 id DESC）"
        print("✅ 分页连续性验证通过")
    else:
        print("⚠️  数据不足，跳过连续性验证")

# ==================== 主函数 ====================

def main():
    print(f"🚀 开始测试 /api/list 接口 (BASE_URL: {BASE_URL})")
    
    # 先检查服务是否可达
    try:
        health = requests.get(f"{BASE_URL}/health", timeout=5)
        if health.status_code != 200:
            print(f"❌ 服务健康检查失败: {health.status_code}")
            sys.exit(1)
        print(f"✅ 服务正常: {health.json()}")
    except requests.exceptions.ConnectionError:
        print(f"❌ 无法连接到 {BASE_URL}，请确认 Flask 服务正在运行")
        sys.exit(1)
    
    # 执行测试
    tests = [
        test_list_default,
        test_list_custom_page,
        test_list_limit_boundary,
        test_list_empty_db,
        test_list_invalid_params,
        test_list_pagination_logic,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"❌ {test_func.__name__} 失败: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__} 异常: {e}")
            failed += 1
    
    # 总结
    print_section("测试总结")
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"📊 总计: {passed + failed}")
    
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())