"""
智能医疗助理 - 全局配置
Smart Healthcare Q&A Assistant - Configuration
"""

import os
import torch

# ========== 设备配置 ==========
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ========== 多模态皮肤分类模型配置 ==========
NUM_CLASSES = 7
IMAGE_SIZE = 224

# 模型权重路径（相对于项目根目录）
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'weights', 'best_model.pth')

# ========== 文件上传配置 ==========
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'temp_uploads')
ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}

# ========== 日志配置 ==========
LOG_DIR = os.path.join(BASE_DIR, 'logs')
LOG_MAX_BYTES = 1024 * 1024 * 10  # 10MB
LOG_BACKUP_COUNT = 10

# ========== RAGFlow API 配置 ==========
RAGFLOW_API_URL = "http://localhost:80"
RAGFLOW_AUTHORIZATION = "ragflow-I1YmQ2OWIyNTg4NDExZjA5MjQxNmVmNz"
RAGFLOW_DATASET_NAME = "药物说明书"
RAGFLOW_ASSIST_NAME = "aa-bot"
RAGFLOW_CHAT_ID = "be09b0a858df11f0adea8efcd719a948"

# ========== 疾病中英文映射 ==========
DISEASE_NAMES = {
    'MEL': '黑色素瘤',
    'NV': '色素痣',
    'BCC': '基底细胞癌',
    'AK': '光化性角化病',
    'BKL': '良性角化病',
    'DF': '皮肤纤维瘤',
    'VASC': '血管病变',
    'SCC': '鳞状细胞癌',
}

# ========== 元数据统计值（用于归一化） ==========
AGE_MEAN = 54
AGE_STD = 323

# ========== Flask 配置 ==========
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
