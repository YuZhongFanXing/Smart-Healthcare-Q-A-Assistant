"""
聊天路由 - 会话创建 & 消息处理
"""

from flask import Blueprint, request, jsonify
import uuid
import logging

from services.ragflow_service import stream_chat

chat_bp = Blueprint('chat', __name__)
logger = logging.getLogger(__name__)


@chat_bp.route('/create_session', methods=['GET'])
def create_session():
    """创建新聊天会话"""
    try:
        response, session_id = stream_chat("你好", None)
        if not session_id:
            session_id = str(uuid.uuid4())
            logger.warning(f"使用回退会话 ID: {session_id}")
        else:
            logger.info(f"新会话 ID: {session_id}")

        return jsonify({"status": "success", "session_id": session_id})
    except Exception as e:
        logger.error(f"创建会话失败: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500


@chat_bp.route('/chat', methods=['POST'])
def chat():
    """处理聊天消息"""
    data = request.json
    message = data.get('message', '')
    session_id = data.get('session_id')

    if not message:
        return jsonify({"error": "No message provided", "status": "error"}), 400

    try:
        logger.info(f"聊天请求: {message[:50]}... (会话: {session_id})")
        response, new_session = stream_chat(message, session_id)
        logger.info(f"聊天响应完成，长度: {len(response)}")
        return jsonify({
            "status": "success",
            "response": response,
            "session_id": session_id
        })
    except Exception as e:
        logger.error(f"聊天处理失败: {e}")
        return jsonify({"status": "error", "error": str(e)}), 500
