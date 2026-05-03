"""
皮肤诊断路由 - 图像分析 & 疾病预测
"""

from flask import Blueprint, request, jsonify
import logging

skin_bp = Blueprint('skin', __name__)
logger = logging.getLogger(__name__)

# 由 app.py 在启动时注入
_predictor = None


def init_predictor(predictor):
    """注入全局预测器实例"""
    global _predictor
    _predictor = predictor


@skin_bp.route('/predict_skin', methods=['POST'])
def predict_skin():
    """皮肤疾病预测"""
    if 'image' not in request.files:
        return jsonify({"error": "No image provided", "status": "error"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Empty filename", "status": "error"}), 400

    try:
        age = request.form.get('age')
        sex = request.form.get('sex', 'unknown')
        anatom_site = request.form.get('anatom_site', 'unknown')
        age = int(age) if age and age.strip() else None

        logger.info(f"皮肤分析 - 年龄: {age}, 性别: {sex}, 部位: {anatom_site}")

        result = _predictor.predict(file, age, sex, anatom_site)

        if result['status'] == 'success':
            logger.info(f"分析成功: {result['disease_name']} ({result['prediction']}) - {result['confidence']:.2f}")
        else:
            logger.error(f"分析失败: {result.get('error')}")

        return jsonify(result)
    except Exception as e:
        logger.error(f"皮肤分析异常: {e}")
        return jsonify({"error": str(e), "status": "error"}), 500
