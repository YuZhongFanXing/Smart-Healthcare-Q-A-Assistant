# 智能医疗问答助手

基于 Flask、多模态深度学习、Supervisor Agent 和医学知识检索的医疗辅助问答系统。系统将文本、医学图像、患者元数据和多轮对话放入同一个任务上下文，由 Agent 根据问题选择合适的医学工具。

> 本系统用于医疗信息辅助和科研验证，输出不能替代医生面诊、检查和正式诊断。

## 一、主要能力

- 统一的文本、图片和结构化信息输入；
- 皮肤病变多模态分类：图片结合年龄、性别和病灶部位；
- 口腔病变多模态检测：图片结合年龄、性别和口腔部位；
- 症状分析、医学知识检索、风险评估；
- 药物相互作用和患者历史查询工具接口；
- 内部医学知识库优先，外部检索补充最新信息；
- 多轮对话和主动追问；
- 医学文档上传与知识库管理。

## 二、多 Agent 工作流

当前工作台截图：

![多模态多 Agent 工作台](docs/images/current_interface.png)

### 2.1 统一会话

用户可以在同一个会话中输入文字、上传一张或多张图片，并补充年龄、性别、病灶位置、症状、用药和患者编号。图片不再单独进入某个固定诊断页面，Agent 会结合对话内容判断图片类型。

### 2.2 Supervisor Agent

Supervisor Agent 负责理解用户意图、维护任务上下文、收集必要元数据、主动追问、选择工具、汇总结果、评估风险并生成多轮回复。

皮肤识别所需元数据：年龄、性别、皮肤病灶部位。  
口腔识别所需元数据：年龄、性别、口腔病变部位。

### 2.3 工具清单

| 工具 | 用途 |
| --- | --- |
| `skin_lesion_classifier` | 皮肤图片与患者元数据联合分类 |
| `oral_lesion_detector` | 口腔图片与患者元数据联合检测 |
| `symptom_analyzer` | 文本症状分析 |
| `medical_knowledge_search` | 内部指南、教材和药品知识检索 |
| `web_search` | 最新研究、药物审批和时效性信息检索 |
| `risk_assessor` | 根据诊断结果和症状进行风险评估 |
| `drug_interaction_checker` | 药物相互作用检查 |
| `patient_history_query` | 患者历史查询 |

### 2.4 RAG 与 Web Search

医学知识检索遵循内部知识优先的决策规则：先调用 `medical_knowledge_search`；命中且置信度充分时直接采用；内容不完整、需要更新或未命中时补充 `web_search`。外部结果保留来源、标题和日期，并优先采用权威医学机构信息。

### 2.5 标准流程

```text
统一文本和图片输入
    -> 输入理解与任务拆解
    -> 识别图片类型并收集必要元数据
    -> 调用图像、症状、历史或用药工具
    -> 优先检索内部医学知识
    -> 必要时补充 Web Search
    -> 汇总证据并评估风险
    -> 返回带来源和置信度的多轮回复
```

## 三、接口

### 统一 Agent 接口

```http
POST /api/agent/chat
Content-Type: multipart/form-data
```

字段包括 `message`、`image` 或 `images[]`、`modality`、`age`、`sex`、`anatom_site`、`oral_site`、`patient_id` 和 `session_id`。

### 兼容接口

- `POST /chat`：文本聊天；
- `POST /predict_skin`：皮肤预测；
- `POST /upload_kb`：知识库文档上传。

## 四、安装与启动

```bash
git clone https://github.com/YuZhongFanXing/Smart-Healthcare-Q-A-Assistant.git
cd Smart-Healthcare-Q-A-Assistant
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

浏览器访问 `http://localhost:5000`。模型权重使用 Git LFS 管理，首次使用时执行 `git lfs pull`。

## 五、配置

敏感配置通过环境变量提供：

```bash
RAGFLOW_API_URL=http://localhost:80
RAGFLOW_AUTHORIZATION=你的API密钥
RAGFLOW_CHAT_ID=你的聊天助手ID
RAGFLOW_DATASET_NAME=药物说明书
RAGFLOW_ASSIST_NAME=aa-bot
FLASK_HOST=0.0.0.0
FLASK_PORT=5000
```

## 六、项目结构

```text
.
├── agents/                 # Supervisor Agent 与任务上下文
├── tools/                  # 医学工具和工具注册表
├── models/                 # 多模态模型
├── routes/                 # Agent、聊天、诊断和知识库接口
├── services/               # RAGFlow 服务封装
├── templates/              # Web 页面
├── static/                 # 前端脚本和样式
├── app.py                  # Flask 应用入口
├── config.py               # 全局配置
└── requirements.txt        # Python 依赖
```

## 七、处理与安全

- 图片进行格式校验和模型预处理；
- 工具结果包含状态、置信度和来源；
- 低置信度结果进入风险评估流程；
- 上传文件使用白名单和临时目录；
- API 密钥通过环境变量管理。

## 八、致谢

- [RAGFlow](https://github.com/infiniflow/ragflow)
- [ConvNeXt](https://github.com/facebookresearch/ConvNeXt)
- [timm](https://github.com/huggingface/pytorch-image-models)
