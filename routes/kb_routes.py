"""
知识库路由 - 文档上传 & 解析
"""

from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
import logging

from services.ragflow_service import get_dataset_id, upload_document, parse_documents

kb_bp = Blueprint('kb', __name__)
logger = logging.getLogger(__name__)


@kb_bp.route('/upload_kb', methods=['POST'])
def upload_kb():
    """上传文件到 RAGFlow 知识库"""
    if 'file' not in request.files:
        return jsonify({"error": "未找到文件", "status": "error"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "文件名为空", "status": "error"}), 400

    filename = secure_filename(file.filename)
    file_content = file.read()

    try:
        dataset_id = get_dataset_id()
        if not dataset_id:
            return jsonify({
                "error": "找不到知识库 '药物说明书'，请先在 RAGFlow 控制台创建",
                "status": "error"
            }), 404

        uploaded_docs = upload_document(dataset_id, filename, file_content)

        # 提取文档 ID 并触发解析
        doc_ids = []
        if isinstance(uploaded_docs, list):
            doc_ids = [doc['id'] for doc in uploaded_docs if 'id' in doc]
        elif isinstance(uploaded_docs, dict) and 'id' in uploaded_docs:
            doc_ids = [uploaded_docs['id']]

        if doc_ids:
            parse_documents(dataset_id, doc_ids)
        else:
            logger.warning("上传成功但未获取到文档 ID，无法自动触发解析")

        return jsonify({
            "status": "success",
            "message": "文件上传并解析成功",
            "filename": filename
        })

    except Exception as e:
        logger.error(f"知识库上传异常: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500
