/**
 * 智能医疗助理 - 前端交互逻辑
 * Smart Healthcare Q&A Assistant
 */

let currentSession = null;
let skinResult = null;
let currentImageFile = null;
let currentKbFile = null;

// ===== DOM 引用 =====
const chatContainer   = document.getElementById('chatContainer');
const chatInput       = document.getElementById('chatInput');
const sendButton      = document.getElementById('sendButton');
const assistantStatus = document.getElementById('assistantStatus');

// 皮肤诊断
const dropArea      = document.getElementById('dropArea');
const fileInput     = document.getElementById('fileInput');
const previewImage  = document.getElementById('previewImage');
const previewInfo   = document.getElementById('previewInfo');
const actionButtons = document.getElementById('actionButtons');
const analyzeButton = document.getElementById('analyzeButton');
const clearButton   = document.getElementById('clearButton');
const skinLoading   = document.getElementById('skinLoading');
const skinResultDiv = document.getElementById('skinResult');
const patientAge    = document.getElementById('patientAge');
const patientSex    = document.getElementById('patientSex');
const anatomSite    = document.getElementById('anatomSite');
const diseaseName   = document.getElementById('diseaseName');
const diseaseCode   = document.getElementById('diseaseCode');
const confidenceVal = document.getElementById('confidenceValue');
const useResultBtn  = document.getElementById('useResultButton');

// 知识库
const kbDropArea     = document.getElementById('kbDropArea');
const kbFileInput    = document.getElementById('kbFileInput');
const kbFileList     = document.getElementById('kbFileList');
const kbActionBtns   = document.getElementById('kbActionButtons');
const kbUploadButton = document.getElementById('kbUploadButton');
const kbLoading      = document.getElementById('kbLoading');

// ===== 面板切换 =====
window.switchTab = function(tabName) {
    document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
    event.target.classList.add('active');

    if (tabName === 'skin') {
        document.getElementById('skinPanel').style.display = 'block';
        document.getElementById('kbPanel').style.display = 'none';
    } else {
        document.getElementById('skinPanel').style.display = 'none';
        document.getElementById('kbPanel').style.display = 'block';
    }
};

// ===== 自动调整输入框高度 =====
chatInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = Math.min(this.scrollHeight, 120) + 'px';
});

// ===== 会话管理 =====
async function createSession() {
    try {
        assistantStatus.className = "assistant-status status-loading";
        assistantStatus.innerHTML = '<i class="fas fa-spinner fa-spin"></i> 正在创建聊天会话...';

        const response = await fetch('/create_session');
        const data = await response.json();

        if (data.status === 'success') {
            currentSession = data.session_id;
            sessionStorage.setItem('medical_session_id', currentSession);
            assistantStatus.className = "assistant-status status-ready";
            assistantStatus.innerHTML = `<i class="fas fa-check-circle"></i> 聊天助手已就绪 (ID: ${currentSession.substring(0, 8)}...)`;

            chatContainer.innerHTML = '';
            addMessage('您好！我是医疗助理，请问有什么可以帮您？您可以描述症状，上传皮肤图片进行分析，或上传相关医学文档丰富我的知识库。', 'assistant');
            chatInput.disabled = false;
            chatInput.placeholder = "输入您的问题...";
            sendButton.disabled = false;
            return true;
        } else {
            throw new Error(data.error || '未知错误');
        }
    } catch (error) {
        console.error('创建会话失败:', error);
        assistantStatus.className = "assistant-status status-error";
        assistantStatus.innerHTML = `<i class="fas fa-exclamation-circle"></i> 聊天助手初始化失败: ${error.message}`;
        return false;
    }
}

function restoreSession(sessionId) {
    currentSession = sessionId;
    assistantStatus.className = "assistant-status status-ready";
    assistantStatus.innerHTML = `<i class="fas fa-check-circle"></i> 聊天助手已恢复会话 (ID: ${sessionId.substring(0, 8)}...)`;

    chatContainer.innerHTML = '';
    addMessage('已恢复之前的会话，您可以继续提问。', 'assistant');
    chatInput.disabled = false;
    chatInput.placeholder = "输入您的问题...";
    sendButton.disabled = false;
}

// ===== 消息处理 =====
function sendMessage() {
    const message = chatInput.value.trim();
    if (!message) return;

    chatInput.value = '';
    chatInput.style.height = 'auto';
    addMessage(message, 'user');

    if (!currentSession) {
        addMessage('聊天会话尚未初始化，请稍后再试', 'assistant');
        return;
    }

    sendButton.disabled = true;
    chatInput.disabled = true;

    let messageWithContext = message;
    if (skinResult) {
        messageWithContext = `（根据皮肤分析：${skinResult.disease_name}，置信度${(skinResult.confidence * 100).toFixed(1)}%）${message}`;
        skinResult = null;
    }

    const thinkingMessage = addMessage('正在思考中...', 'assistant');

    fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: messageWithContext, session_id: currentSession })
    })
    .then(response => response.json())
    .then(data => {
        thinkingMessage.remove();
        if (data.status === 'success') {
            addMessage(data.response || '抱歉，我暂时无法回答您的问题。', 'assistant');
        } else {
            addMessage('出错: ' + (data.error || '未知错误'), 'assistant');
        }
    })
    .catch(() => {
        thinkingMessage.remove();
        addMessage('网络错误，请重试', 'assistant');
    })
    .finally(() => {
        sendButton.disabled = false;
        chatInput.disabled = false;
        chatInput.focus();
    });
}

function addMessage(text, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}-message`;

    const avatar = document.createElement('div');
    avatar.className = `message-avatar ${sender}-avatar`;
    avatar.innerHTML = sender === 'user'
        ? '<i class="fas fa-user"></i>'
        : '<i class="fas fa-robot"></i>';

    const content = document.createElement('div');
    content.className = 'message-content';
    content.textContent = text;

    if (sender === 'user') {
        messageDiv.appendChild(content);
        messageDiv.appendChild(avatar);
    } else {
        messageDiv.appendChild(avatar);
        messageDiv.appendChild(content);
    }

    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
    return messageDiv;
}

// ===== 皮肤图片处理 =====
function handleImageSelect(file) {
    if (!file.type.match('image.*')) {
        alert('请选择图片文件！');
        return;
    }

    currentImageFile = file;
    const reader = new FileReader();
    reader.onload = function(e) {
        previewImage.src = e.target.result;
        previewImage.style.display = 'block';
        previewInfo.style.display = 'block';
        actionButtons.style.display = 'flex';
    };
    reader.readAsDataURL(file);
    skinResultDiv.style.display = 'none';
    skinResult = null;
}

function startAnalysis() {
    if (!currentImageFile) return alert('请先选择图片！');

    skinLoading.style.display = 'block';
    skinResultDiv.style.display = 'none';
    actionButtons.style.display = 'none';

    const formData = new FormData();
    formData.append('image', currentImageFile);
    formData.append('age', patientAge.value || '');
    formData.append('sex', patientSex.value);
    formData.append('anatom_site', anatomSite.value);

    fetch('/predict_skin', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        skinLoading.style.display = 'none';
        if (data.status === 'success') {
            skinResult = data;
            diseaseName.textContent = data.disease_name;
            diseaseCode.textContent = data.prediction;
            confidenceVal.textContent = (data.confidence * 100).toFixed(2) + '%';
            skinResultDiv.style.display = 'block';
        } else {
            alert('分析失败: ' + (data.error || '未知错误'));
            actionButtons.style.display = 'flex';
        }
    })
    .catch(error => {
        skinLoading.style.display = 'none';
        alert('请求失败: ' + error.message);
        actionButtons.style.display = 'flex';
    });
}

// ===== 知识库文件处理 =====
function handleKbFileSelect(file) {
    currentKbFile = file;
    kbFileList.innerHTML = `
        <div style="padding: 0.5rem; background: #eee; border-radius: 5px; display: flex; align-items: center; gap: 0.5rem;">
            <i class="fas fa-file-alt"></i> ${file.name} (${(file.size/1024).toFixed(1)} KB)
        </div>
    `;
    kbActionBtns.style.display = 'flex';
}

function startKbUpload() {
    if (!currentKbFile) return alert('请选择文件！');

    kbLoading.style.display = 'block';
    kbActionBtns.style.display = 'none';

    const formData = new FormData();
    formData.append('file', currentKbFile);

    fetch('/upload_kb', { method: 'POST', body: formData })
    .then(response => response.json())
    .then(data => {
        kbLoading.style.display = 'none';
        if (data.status === 'success') {
            alert('文件上传并开始解析成功！');
            kbFileList.innerHTML = '';
            currentKbFile = null;
            addMessage(`我已接收到新文档 "${data.filename}"，正在进行解析学习。完成后我将能回答相关问题。`, 'assistant');
        } else {
            alert('上传失败: ' + (data.error || '未知错误'));
            kbActionBtns.style.display = 'flex';
        }
    })
    .catch(error => {
        kbLoading.style.display = 'none';
        alert('请求失败: ' + error.message);
        kbActionBtns.style.display = 'flex';
    });
}

// ===== 拖拽支持 =====
function setupDragDrop(element, callback) {
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, e => { e.preventDefault(); e.stopPropagation(); }, false);
    });
    ['dragenter', 'dragover'].forEach(eventName => {
        element.addEventListener(eventName, () => element.style.backgroundColor = 'rgba(52, 152, 219, 0.15)', false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        element.addEventListener(eventName, () => element.style.backgroundColor = '', false);
    });
    element.addEventListener('drop', e => {
        const files = e.dataTransfer.files;
        if (files.length) callback(files[0]);
    });
}

// ===== 事件绑定 =====
sendButton.addEventListener('click', sendMessage);
chatInput.addEventListener('keypress', e => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage(); }
});

// 皮肤图片上传
dropArea.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', e => { if (e.target.files.length) handleImageSelect(e.target.files[0]); });
analyzeButton.addEventListener('click', startAnalysis);
clearButton.addEventListener('click', () => {
    currentImageFile = null;
    previewImage.style.display = 'none';
    previewInfo.style.display = 'none';
    actionButtons.style.display = 'none';
    skinResultDiv.style.display = 'none';
    fileInput.value = '';
});

// 结果发送到聊天
useResultBtn.addEventListener('click', () => {
    if (skinResult) {
        const message = `我的皮肤分析结果是：${skinResult.disease_name}（${skinResult.prediction}），置信度${(skinResult.confidence * 100).toFixed(1)}%，请问这意味着什么？`;
        chatInput.value = message;
        sendMessage();
        skinResultDiv.style.display = 'none';
        skinResult = null;
    }
});

// 知识库上传
kbDropArea.addEventListener('click', () => kbFileInput.click());
kbFileInput.addEventListener('change', e => { if (e.target.files.length) handleKbFileSelect(e.target.files[0]); });
kbUploadButton.addEventListener('click', startKbUpload);

// 拖拽
setupDragDrop(dropArea, handleImageSelect);
setupDragDrop(kbDropArea, handleKbFileSelect);

// ===== 初始化 =====
document.addEventListener('DOMContentLoaded', () => {
    const savedSession = sessionStorage.getItem('medical_session_id');
    if (savedSession) restoreSession(savedSession);
    else createSession();
});
