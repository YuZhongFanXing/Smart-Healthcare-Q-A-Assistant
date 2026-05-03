"""
智能医疗助理 — 主应用入口
Smart Healthcare Q&A Assistant

基于 ConvNeXt-Tiny + 跨模态注意力池化的多模态皮肤疾病诊断系统，
集成 RAGFlow 知识库检索增强问答。

启动方式:
    python app.py
    然后访问 http://localhost:5000
"""

import os
import logging
from logging.handlers import RotatingFileHandler

from flask import Flask, render_template_string

from config import (
    DEVICE, MODEL_PATH, UPLOAD_FOLDER, LOG_DIR,
    LOG_MAX_BYTES, LOG_BACKUP_COUNT,
    RAGFLOW_CHAT_ID, FLASK_HOST, FLASK_PORT, DISEASE_NAMES
)
from models.skin_model import SkinDiseasePredictor
from routes.chat_routes import chat_bp
from routes.skin_routes import skin_bp, init_predictor
from routes.kb_routes import kb_bp


def setup_logging(app):
    """配置日志系统（UTF-8 编码，自动轮转）"""

    class SafeRotatingFileHandler(RotatingFileHandler):
        def __init__(self, filename, **kwargs):
            kwargs['encoding'] = 'utf-8'
            super().__init__(filename, **kwargs)

        def emit(self, record):
            try:
                super().emit(record)
            except UnicodeEncodeError:
                record.msg = record.msg.encode('utf-8', errors='replace').decode('utf-8')
                super().emit(record)

    os.makedirs(LOG_DIR, exist_ok=True)
    file_handler = SafeRotatingFileHandler(
        os.path.join(LOG_DIR, 'app.log'),
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
    app.logger.setLevel(logging.INFO)
    app.logger.info('Application startup')


def load_model():
    """加载多模态皮肤分类模型"""
    from services.ragflow_service import get_dataset_id

    app.logger.info("正在加载多模态皮肤分类模型...")
    predictor = SkinDiseasePredictor(
        MODEL_PATH, device=str(DEVICE), disease_names=DISEASE_NAMES
    )
    app.logger.info(f"模型加载完成 | 设备: {DEVICE}")

    # 注入到路由
    init_predictor(predictor)

    return predictor


def create_app():
    """创建并配置 Flask 应用"""
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

    setup_logging(app)

    # 注册蓝图
    app.register_blueprint(chat_bp)
    app.register_blueprint(skin_bp)
    app.register_blueprint(kb_bp)

    # 加载 HTML 模板
    template_path = os.path.join(os.path.dirname(__file__), 'templates', 'index.html')
    with open(template_path, 'r', encoding='utf-8') as f:
        INDEX_HTML = f.read()

    @app.route('/')
    def index():
        return render_template_string(INDEX_HTML)

    return app


# ========== 启动入口 ==========
if __name__ == '__main__':
    app = create_app()

    try:
        predictor = load_model()
    except Exception as e:
        app.logger.error(f"模型加载失败: {e}")
        raise

    app.logger.info(f"聊天助手 ID: {RAGFLOW_CHAT_ID}")

    # 预热知识库缓存
    from services.ragflow_service import get_dataset_id
    try:
        get_dataset_id()
    except Exception:
        pass

    app.run(host=FLASK_HOST, port=FLASK_PORT, threaded=True)
