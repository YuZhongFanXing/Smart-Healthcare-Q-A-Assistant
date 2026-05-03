"""
RAGFlow 服务层
封装与 RAGFlow API 的交互：聊天、知识库管理、文档解析
"""

import json
import requests
import logging

logger = logging.getLogger(__name__)

# 延迟导入避免循环依赖
_config_cache = {}


def _get_config():
    """延迟加载配置，避免循环导入"""
    if not _config_cache:
        from config import (
            RAGFLOW_API_URL, RAGFLOW_AUTHORIZATION, RAGFLOW_DATASET_NAME,
            RAGFLOW_CHAT_ID, RAGFLOW_ASSIST_NAME
        )
        _config_cache.update({
            'API_URL': RAGFLOW_API_URL,
            'AUTHORIZATION': RAGFLOW_AUTHORIZATION,
            'DATASET_NAME': RAGFLOW_DATASET_NAME,
            'CHAT_ID': RAGFLOW_CHAT_ID,
            'ASSIST_NAME': RAGFLOW_ASSIST_NAME,
        })
    return _config_cache


def _make_headers(json_content=True):
    """构建请求头"""
    cfg = _get_config()
    headers = {"Authorization": f"Bearer {cfg['AUTHORIZATION']}"}
    if json_content:
        headers["Content-Type"] = "application/json"
    return headers


def rq(method, url, allow_code_102=False, **kw):
    """通用 RAGFlow API 请求封装"""
    cfg = _get_config()
    kw.setdefault("headers", _make_headers())
    try:
        r = requests.request(method, f"{cfg['API_URL']}{url}", timeout=30, **kw)
        r.raise_for_status()
        j = r.json()
        code = j.get("code")
        if code == 102 and allow_code_102:
            return None
        if code != 0:
            raise RuntimeError(f"[API {code}] {j.get('message')}")
        return j.get("data")
    except Exception as e:
        logger.error(f"API 请求失败: {method} {url} - {e}")
        raise


def stream_chat(question, session_id=None):
    """流式聊天请求，返回完整回复和新会话ID"""
    cfg = _get_config()
    payload = {"question": question, "stream": True}
    if session_id:
        payload["session_id"] = session_id

    try:
        logger.info(f"发送聊天请求: {question[:50]}...")
        with requests.post(
            f"{cfg['API_URL']}/api/v1/chats/{cfg['CHAT_ID']}/completions",
            headers=_make_headers(),
            json=payload,
            stream=True,
            timeout=120
        ) as r:
            r.raise_for_status()
            full_response = ""
            new_session = session_id

            for line_num, raw in enumerate(r.iter_lines(decode_unicode=True)):
                if not raw:
                    continue
                if raw.startswith(":"):
                    continue
                if raw.startswith("data: "):
                    raw = raw[6:]

                json_start = raw.find("{")
                if json_start == -1:
                    continue

                try:
                    chunk = json.loads(raw[json_start:])
                except json.JSONDecodeError:
                    logger.warning(f"JSON 解析失败 (行 {line_num})")
                    continue

                code = chunk.get("code")
                if code == 102:
                    continue
                elif code != 0:
                    raise RuntimeError(f"API 错误: {chunk.get('message', '未知错误')}")

                data = chunk.get("data")
                if data is True:
                    break
                if isinstance(data, dict):
                    answer = data.get("answer", "")
                    if data.get("session_id"):
                        new_session = data["session_id"]
                    if answer:
                        full_response = answer

            logger.info(f"流式响应完成，长度: {len(full_response)}")
            if not full_response.strip():
                full_response = "抱歉，我暂时无法回答您的问题。请稍后再试或重新表述您的问题。"
            return full_response.strip(), new_session

    except requests.exceptions.Timeout:
        logger.error("请求超时")
        return "请求超时，请稍后再试", session_id
    except requests.exceptions.RequestException as e:
        logger.error(f"网络请求失败: {e}")
        return f"网络连接失败: {e}", session_id
    except Exception as e:
        logger.error(f"聊天流处理失败: {e}")
        return f"处理失败: {e}", session_id


# ========== 知识库管理 ==========

_dataset_id_cache = None


def get_dataset_id():
    """获取目标知识库的 ID（带缓存）"""
    global _dataset_id_cache
    cfg = _get_config()

    if _dataset_id_cache:
        return _dataset_id_cache

    try:
        params = {"page": 1, "page_size": 100, "name": cfg['DATASET_NAME']}
        data = rq('GET', '/api/v1/datasets', params=params)

        for dataset in data:
            if dataset.get('name') == cfg['DATASET_NAME']:
                _dataset_id_cache = dataset.get('id')
                logger.info(f"找到知识库 '{cfg['DATASET_NAME']}' ID: {_dataset_id_cache}")
                return _dataset_id_cache

        logger.warning(f"未找到名为 '{cfg['DATASET_NAME']}' 的知识库")
        return None
    except Exception as e:
        logger.error(f"获取知识库 ID 失败: {e}")
        return None


def upload_document(dataset_id, filename, file_content):
    """上传文档到指定知识库"""
    cfg = _get_config()
    upload_url = f"{cfg['API_URL']}/api/v1/datasets/{dataset_id}/documents"
    headers = {"Authorization": f"Bearer {cfg['AUTHORIZATION']}"}
    files = {'file': (filename, file_content)}

    logger.info(f"正在上传文件 '{filename}' 到知识库 {dataset_id}...")
    response = requests.post(upload_url, headers=headers, files=files, timeout=60)
    response.raise_for_status()

    resp_json = response.json()
    if resp_json.get("code") != 0:
        raise RuntimeError(f"上传失败: {resp_json.get('message')}")

    return resp_json.get('data', [])


def parse_documents(dataset_id, doc_ids):
    """触发文档解析"""
    cfg = _get_config()
    parse_url = f"{cfg['API_URL']}/api/v1/datasets/{dataset_id}/chunks"
    payload = {"document_ids": doc_ids}

    response = requests.post(parse_url, headers=_make_headers(), json=payload, timeout=30)
    if response.status_code == 200:
        logger.info(f"解析任务已提交，文档 IDs: {doc_ids}")
    else:
        logger.warning(f"解析任务提交可能失败: {response.text}")
