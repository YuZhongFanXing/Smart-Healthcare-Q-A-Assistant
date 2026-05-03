# Smart Healthcare Q&A Assistant

智能医疗助理 — 基于多模态深度学习与 RAG 知识库检索增强的医疗辅助诊断系统。

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-3.0+-lightgrey.svg)](https://flask.palletsprojects.com/)
[![RAGFlow](https://img.shields.io/badge/RAGFlow-0.x-green.svg)](https://github.com/infiniflow/ragflow)
[![License](https://img.shields.io/badge/License-Apache%202.0-orange.svg)](LICENSE)

---

## 项目简介

本项目是一个集**多模态皮肤病分类**和 **RAG 知识库问答**于一体的智能医疗辅助系统。核心功能包括：

1. **皮肤疾病智能分类**：支持黑色素瘤、基底细胞癌等 7 种常见皮肤病的自动识别
2. **医疗智能问答**：基于 RAGFlow 的专业医学知识检索增强生成，支持药品查询、疾病咨询等
3. **知识库管理**：支持上传 PDF、Word、TXT 等格式的医学文档，自动解析入库

系统底层技术栈基于开源项目 [RAGFlow](https://github.com/infiniflow/ragflow)（79k+ Stars），实现了检索增强生成（RAG）与 AI Agent 的深度融合。

> **重要提示**：本系统仅供辅助参考，诊断结果不能替代专业医疗意见。

---

## 功能模块

### 1. 皮肤疾病诊断

采用 **ConvNeXt-Tiny + 跨模态注意力池化** 架构，融合皮肤图像和患者元数据（年龄、性别、病灶部位）进行综合分类。

| 英文代码 | 中文名称     |
|---------|------------|
| MEL     | 黑色素瘤    |
| NV      | 色素痣      |
| BCC     | 基底细胞癌  |
| AK      | 光化性角化病 |
| BKL     | 良性角化病  |
| DF      | 皮肤纤维瘤  |
| VASC    | 血管病变    |

**技术亮点：**
- 多模态注意力机制：融合图像与元数据信息
- 空间金字塔投影：将 ConvNeXt 特征图映射为区域特征序列
- 跨模态注意力池化：自适应学习图像区域与元数据的交互权重

### 2. 医疗智能问答

基于 RAGFlow 的检索增强生成系统，功能包括：
- 医学知识问答
- 药物信息查询
- 诊断结果解读
- 多轮对话支持

### 3. 知识库管理

- 文档上传：支持 PDF、DOCX、TXT、MD 等格式
- 自动解析与向量化
- 知识库索引管理

---

## 系统架构

```
┌──────────────────────────────────────────────────┐
│                   Flask Web UI                     │
│         (templates/ + static/css + static/js)      │
├─────────────┬──────────────┬──────────────────────┤
│  Chat 路由   │  Skin 路由    │  KB 路由             │
│  (会话/问答) │  (图像诊断)   │  (文档上传/解析)      │
├─────────────┼──────────────┼──────────────────────┤
│ RAGFlow     │ SkinModel    │  RAGFlow             │
│ stream_chat │ Predictor    │  upload_document     │
└─────────────┴──────────────┴──────────────────────┘
```

**项目目录结构：**

```
Smart-Healthcare-Q-A-Assistant/
├── app.py                       # Flask 应用入口
├── config.py                    # 全局配置
├── requirements.txt             # Python 依赖
├── .gitignore
│
├── models/
│   ├── __init__.py
│   ├── skin_model.py            # 多模态皮肤分类模型
│   └── weights/
│       └── best_model.pth       # 训练好的模型权重 (Git LFS)
│
├── services/
│   ├── __init__.py
│   └── ragflow_service.py       # RAGFlow API 服务封装
│
├── routes/
│   ├── __init__.py
│   ├── chat_routes.py           # 聊天/会话路由
│   ├── skin_routes.py           # 皮肤诊断路由
│   └── kb_routes.py             # 知识库路由
│
├── templates/
│   └── index.html               # 前端页面模板
│
└── static/
    ├── css/
    │   └── style.css            # 样式表
    └── js/
        └── main.js              # 前端交互逻辑
```

---

## 从零开始部署

### 第一步：环境准备

#### 硬件要求

| 组件 | 最低配置                | 推荐配置                     |
|------|-----------------------|-----------------------------|
| CPU  | Intel Core i5 或同等    | Intel Core i7+              |
| 内存 | 8 GB                   | 16 GB+                      |
| 磁盘 | 10 GB 可用空间          | 50 GB SSD                   |
| GPU  | 无（CPU 模式可用）       | NVIDIA GPU (支持 CUDA 加速)  |

#### 软件要求

- **操作系统**：Windows 10/11 或 Linux (Ubuntu 20.04+)
- **Python**：3.8 及以上
- **Git**：用于克隆仓库
- **Docker & Docker Compose**：用于运行 RAGFlow

---

### 第二步：安装 RAGFlow（知识库引擎）

本项目依赖 RAGFlow 提供知识库检索和聊天服务。首先需要安装并启动 RAGFlow。

```bash
# 1. 克隆 RAGFlow 仓库
git clone https://github.com/infiniflow/ragflow.git
cd ragflow

# 2. 启动 RAGFlow 服务（使用 Docker Compose）
docker compose -f docker/docker-compose.yml up -d
```

启动后，RAGFlow 服务默认运行在 `http://localhost:80`。

> **RAGFlow 启动依赖**：Elasticsearch、MySQL、Redis、MinIO，均由 Docker Compose 自动管理。

#### RAGFlow 初始化配置

1. 浏览器访问 `http://localhost:80` 进入 RAGFlow 控制台
2. 注册管理员账号并登录
3. 创建知识库，名称为：**药物说明书**（与 `config.py` 中 `RAGFLOW_DATASET_NAME` 保持一致）
4. 创建聊天助手，名称为：**aa-bot**（与 `config.py` 中 `RAGFLOW_ASSIST_NAME` 保持一致）
5. 在 RAGFlow 控制台 → API 管理中获取 API Key，更新 `config.py` 中的 `RAGFLOW_AUTHORIZATION`
6. 获取聊天助手 ID，更新 `config.py` 中的 `RAGFLOW_CHAT_ID`

---

### 第三步：克隆并配置本项目

```bash
# 1. 克隆仓库
git clone https://github.com/YuZhongFanXing/Smart-Healthcare-Q-A-Assistant.git
cd Smart-Healthcare-Q-A-Assistant

# 2. 创建 Python 虚拟环境（推荐）
python -m venv venv

# Windows 激活虚拟环境
venv\Scripts\activate
# Linux/Mac 激活虚拟环境
source venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt
```

> **注意**：如果使用 GPU 加速，请安装 CUDA 版本的 PyTorch：
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

### 第四步：获取模型权重

模型权重文件使用 Git LFS 管理。如果你已安装 Git LFS，克隆仓库时会自动下载。

```bash
# 安装 Git LFS（如果尚未安装）
# Windows: 下载安装 https://git-lfs.com
# Linux: sudo apt install git-lfs
# Mac: brew install git-lfs

git lfs install
git lfs pull
```

如果无法使用 Git LFS，可以从以下备选方式获取：
- 模型文件位于 `models/weights/best_model.pth`（约 341 MB）
- 请确保该文件存在后再启动应用

---

### 第五步：启动应用

```bash
python app.py
```

启动成功后，浏览器访问 `http://localhost:5000`。

预期输出：
```
 * Running on http://0.0.0.0:5000
[INFO] Application startup
[INFO] 模型加载成功 | 设备: cpu | 类别数: 7 | 元数据维度: 13
[INFO] 聊天助手 ID: be09b0a858df11f0adea8efcd719a948
```

---

## 使用指南

### 皮肤诊断操作

1. 点击页面右侧「皮肤诊断」标签
2. 填写患者信息：
   - **年龄**：输入患者年龄（可选，留空默认为归一化中值）
   - **性别**：选择男性/女性/未知
   - **病灶部位**：选择皮肤病变所在的身体部位
3. 上传皮肤病变图片（点击或拖拽）
4. 点击「开始分析」
5. 查看诊断结果，包括：
   - 预测疾病名称（中文）
   - 疾病英文代码
   - 置信度百分比
6. 可点击「将此结果发送给助理」让 AI 解读诊断结果

### 聊天问答操作

1. 页面加载后自动创建聊天会话
2. 在左侧聊天框输入问题
3. 按「Enter」键或点击发送按钮
4. AI 助理基于医学知识库进行回答
5. 支持多轮对话，上下文连贯

### 知识库上传

1. 点击右侧「知识库上传」标签
2. 点击上传区域选择文件，或拖拽文件到上传区域
3. 确认文件信息，点击「确认上传并解析」
4. 等待解析完成（可在 RAGFlow 控制台查看解析状态）
5. 解析完成后，AI 助理即可基于新文档回答问题

---

## 技术详解

### 多模态皮肤分类模型

**模型架构：**

```
输入 (Image + Metadata)
  │
  ├─ ConvNeXt-Tiny ──→ Spatial Pyramid Projection
  │                            │
  │                    图像区域特征 (B, 16, 256)
  │                            │
  └─ MLP Encoder ──→ 元数据特征 (B, 128)
                              │
                    Cross-Modal Attention Pooling
                              │
                      融合特征 (B, 384)
                              │
                          Classifier
                              │
                    7 类疾病概率分布
```

**关键组件说明：**

| 组件 | 功能 |
|------|------|
| ConvNeXt-Tiny | 图像骨干网络，提取皮肤病变视觉特征 |
| Spatial Pyramid Projection | 将 2D 特征图映射为空间区域序列（4x4 网格） |
| MLP Encoder | 编码患者元数据（年龄/性别/部位） |
| Cross-Modal Attention | 学习图像区域与元数据的交互权重 |
| Classifier | 两层 MLP 输出 7 类概率分布 |

### RAG 问答流程

```
用户提问
  │
  ├─→ RAGFlow API /completions
  │       │
  │       ├─ 检索相关文档块 (Elasticsearch)
  │       ├─ 拼接检索上下文 + 用户问题
  │       ├─ 调用 LLM 生成回答
  │       └─ 流式返回
  │
  └─→ 前端展示（打字机效果）
```

### 图像处理流程

1. 接收用户上传的图片文件
2. 验证图片格式和大小
3. 图像预处理：缩放到 224x224 → 归一化（ImageNet 均值/标准差）
4. 提取患者元数据并编码
5. 多模态特征融合与推理
6. Softmax 概率计算
7. 返回分类结果和置信度

---

## 配置说明

所有可配置项位于 [config.py](config.py)：

| 配置项 | 说明 | 默认值 |
|--------|------|--------|
| `RAGFLOW_API_URL` | RAGFlow 服务地址 | `http://localhost:80` |
| `RAGFLOW_AUTHORIZATION` | RAGFlow API 认证 Token | 需自行获取 |
| `RAGFLOW_CHAT_ID` | RAGFlow 聊天助手 ID | 需自行获取 |
| `RAGFLOW_DATASET_NAME` | 知识库名称 | `药物说明书` |
| `MODEL_PATH` | 模型权重文件路径 | `models/weights/best_model.pth` |
| `FLASK_HOST` | Flask 监听地址 | `0.0.0.0` |
| `FLASK_PORT` | Flask 监听端口 | `5000` |
| `DEVICE` | 推理设备 | 自动检测 cuda/cpu |

---

## 常见问题

### Q1: 启动时提示模型文件找不到？
确保 `models/weights/best_model.pth` 文件存在。如果使用 Git LFS 克隆，需执行 `git lfs pull` 拉取大文件。

### Q2: RAGFlow 连接失败怎么办？
1. 确认 RAGFlow Docker 容器正在运行：`docker ps`
2. 确认 RAGFlow 服务端口（默认 80）未被占用
3. 检查 `config.py` 中的认证 Token 和 Chat ID 是否正确

### Q3: 皮肤诊断结果不准确？
- 确保上传的图片清晰、光线充足
- 填写准确的患者元数据（年龄、性别、病灶部位）
- 本模型为辅助工具，诊断结果仅供参考

### Q4: 如何使用 GPU 加速？
安装 CUDA 版本的 PyTorch 后，系统会自动检测并使用 GPU。启动日志中会显示 `设备: cuda`。

### Q5: 知识库上传后 AI 无法引用新文档？
文档上传后需要等待 RAGFlow 完成解析和向量化。可在 RAGFlow 控制台 → 知识库 → 文档管理中查看解析状态。解析完成后需等待索引构建完毕（通常数秒到数分钟）。

---

## 安全说明

- 文件上传严格验证格式和大小
- 患者元数据仅用于推理，不持久化存储
- 上传的图片在临时目录中自动清理
- 系统日志不记录敏感用户信息

---

## 许可证

本项目基于 [Apache 2.0 License](LICENSE) 开源。

底层 RAGFlow 引擎同样使用 Apache 2.0 许可证。

---

## 致谢

- [RAGFlow](https://github.com/infiniflow/ragflow) — 开源 RAG 引擎
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt) — 图像骨干网络
- [timm](https://github.com/huggingface/pytorch-image-models) — 预训练模型库
