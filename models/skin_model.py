"""
多模态皮肤疾病分类模型
基于 ConvNeXt-Tiny + 跨模态注意力池化（Cross-Modal Attention Pooling）

架构：
  Image → ConvNeXt-Tiny → Spatial Pyramid Projection ─┐
                                                        ├→ Cross-Modal Attention → Classifier
  Metadata (age/sex/site) → MLP Encoder ───────────────┘

支持 7 类皮肤病分类：MEL, NV, BCC, AK, BKL, DF, VASC
"""

import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class SpatialPyramidProjection(nn.Module):
    """空间金字塔投影：将 ConvNeXt 特征图映射为区域特征序列"""

    def __init__(self, in_channels, out_channels, grid_size=4):
        super().__init__()
        self.grid_size = grid_size
        self.num_regions = grid_size * grid_size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((grid_size, grid_size))
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

    def forward(self, x):
        x = self.adaptive_pool(x)
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        return x


class CrossModalAttentionPooling(nn.Module):
    """跨模态注意力池化：融合图像区域特征和元数据特征"""

    def __init__(self, img_feat_dim, meta_feat_dim, hidden_dim=256):
        super().__init__()
        self.attention_net = nn.Sequential(
            nn.Linear(img_feat_dim + meta_feat_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )
        self.fc = nn.Linear(img_feat_dim + meta_feat_dim, hidden_dim)
        self.norm = nn.BatchNorm1d(hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, img_regions, meta_feat):
        B, num_regions, _ = img_regions.shape
        meta_expanded = meta_feat.unsqueeze(1).expand(-1, num_regions, -1)
        combined = torch.cat([img_regions, meta_expanded], dim=-1)
        attn_scores = self.attention_net(combined).squeeze(-1)
        attn_weights = torch.softmax(attn_scores, dim=-1)
        weighted_features = torch.einsum('brd,br->bd', combined, attn_weights)
        fused_feat = self.fc(weighted_features)
        fused_feat = self.norm(fused_feat)
        fused_feat = self.activation(fused_feat)
        fused_feat = self.dropout(fused_feat)
        return fused_feat, attn_weights


class MultiModalConvNeXt(nn.Module):
    """多模态 ConvNeXt：融合皮肤图像和患者元数据进行分类"""

    def __init__(self, num_classes, meta_dim, region_size=4, region_feat_dim=256):
        super().__init__()
        self.img_backbone = timm.create_model(
            'convnext_tiny', pretrained=False, num_classes=0,
            features_only=True, out_indices=[3]
        )
        backbone_out_dim = 768
        self.spatial_pyramid = SpatialPyramidProjection(
            backbone_out_dim, region_feat_dim, grid_size=region_size
        )
        self.meta_fc = nn.Sequential(
            nn.Linear(meta_dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.2)
        )
        self.cross_attn_pooling = CrossModalAttentionPooling(
            region_feat_dim, 128, hidden_dim=384
        )
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img, meta):
        img_features = self.img_backbone(img)[0]
        img_regions = self.spatial_pyramid(img_features)
        meta_features = self.meta_fc(meta)
        fused_features, attn_weights = self.cross_attn_pooling(img_regions, meta_features)
        logits = self.classifier(fused_features)
        return logits, attn_weights


class SkinDiseasePredictor:
    """皮肤病预测器：封装模型加载、预处理和推理"""

    def __init__(self, model_path, device='cpu', disease_names=None):
        self.device = torch.device(device)
        self.model_path = model_path
        self.model = None
        self.label_encoder = None
        self.meta_dim = None
        self.disease_names = disease_names or {}
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self._load_model()

    def _load_model(self):
        """加载训练好的模型权重"""
        try:
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint['model_state_dict']['classifier.4.weight'].shape[0]
            self.meta_dim = checkpoint['model_state_dict']['meta_fc.0.weight'].shape[1]
            self.model = MultiModalConvNeXt(num_classes, self.meta_dim).to(self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            if 'label_encoder' in checkpoint:
                self.label_encoder = checkpoint['label_encoder']
            logger.info(f"模型加载成功 | 设备: {self.device} | 类别数: {num_classes} | 元数据维度: {self.meta_dim}")
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise

    def preprocess_metadata(self, age=None, sex=None, anatom_site=None):
        """预处理患者元数据（年龄归一化 + 性别/部位 One-Hot 编码）"""
        from config import AGE_MEAN, AGE_STD

        age_norm = (age - AGE_MEAN) / AGE_STD if age is not None else 0.0
        sex = sex if sex is not None else "unknown"
        anatom_site = anatom_site if anatom_site is not None else "unknown"

        if self.meta_dim == 13:
            meta_features = {
                'age_norm': age_norm,
                'sex_female': 1.0 if sex == 'female' else 0.0,
                'sex_male': 1.0 if sex == 'male' else 0.0,
                'sex_unknown': 1.0 if sex == 'unknown' else 0.0,
                'site_anterior torso': 1.0 if anatom_site == 'anterior torso' else 0.0,
                'site_head/neck': 1.0 if anatom_site == 'head/neck' else 0.0,
                'site_lower extremity': 1.0 if anatom_site == 'lower extremity' else 0.0,
                'site_oral/genital': 1.0 if anatom_site == 'oral/genital' else 0.0,
                'site_palms/soles': 1.0 if anatom_site == 'palms/soles' else 0.0,
                'site_posterior torso': 1.0 if anatom_site == 'posterior torso' else 0.0,
                'site_torso': 1.0 if anatom_site == 'torso' else 0.0,
                'site_upper extremity': 1.0 if anatom_site == 'upper extremity' else 0.0,
                'site_unknown': 1.0 if anatom_site == 'unknown' else 0.0
            }
        else:
            # 回退：构建简单向量
            meta_vector = [0.0] * self.meta_dim
            meta_vector[0] = age_norm
            if sex == 'female':
                meta_vector[1] = 1.0
            elif sex == 'male':
                meta_vector[2] = 1.0
            else:
                meta_vector[3] = 1.0
            return torch.tensor(meta_vector, dtype=torch.float32).unsqueeze(0)

        sorted_features = dict(sorted(meta_features.items()))
        meta_tensor = torch.tensor(list(sorted_features.values()), dtype=torch.float32)
        return meta_tensor.unsqueeze(0)

    def preprocess_image(self, image):
        """预处理输入图像（支持路径、文件对象、PIL Image）"""
        try:
            if isinstance(image, str):
                image = Image.open(image).convert('RGB')
            elif hasattr(image, 'read'):
                image = Image.open(image.stream).convert('RGB')
            elif isinstance(image, Image.Image):
                image = image.convert('RGB')
            else:
                raise ValueError(f"不支持的图像格式: {type(image)}")
            return self.transform(image).unsqueeze(0)
        except Exception as e:
            logger.error(f"图像预处理失败: {e}")
            raise

    def predict(self, image, age=None, sex=None, anatom_site=None):
        """预测皮肤疾病类型，返回疾病名和置信度"""
        try:
            image_tensor = self.preprocess_image(image).to(self.device)
            meta_tensor = self.preprocess_metadata(age, sex, anatom_site).to(self.device)

            with torch.no_grad():
                logits, _ = self.model(image_tensor, meta_tensor)
                probabilities = torch.softmax(logits, dim=1)

            max_prob, max_idx = torch.max(probabilities, dim=1)
            class_idx = max_idx.item()
            prob = max_prob.item()

            if self.label_encoder is not None:
                class_name = self.label_encoder.inverse_transform([class_idx])[0]
            else:
                class_name = f"Class_{class_idx}"

            disease_name = self.disease_names.get(class_name, class_name)

            return {
                "prediction": class_name,
                "disease_name": disease_name,
                "confidence": float(prob),
                "status": "success"
            }
        except Exception as e:
            logger.error(f"预测出错: {e}")
            return {"error": str(e), "status": "error"}
