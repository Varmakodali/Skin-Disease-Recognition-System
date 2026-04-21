import torch
import torch.nn as nn
import timm
import torch.nn.functional as F

class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for channel attention.
    """
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SkinLesionModel(nn.Module):
    """
    Skin Lesion Classification Model using EfficientNetV2-L and SE blocks.
    """
    def __init__(self, num_classes=10, pretrained=True):
        super(SkinLesionModel, self).__init__()
        # Load EfficientNet-B0 backbone (Smaller and faster for CPU)
        self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=0, global_pool='')
        
        # Determine the number of output channels from the backbone dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, 224, 224)
            dummy_features = self.backbone(dummy_input)
            self.num_features = dummy_features.shape[1]
        
        # Channel Attention Module
        self.attention = SEBlock(self.num_features)
        
        # Classifier Head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(self.num_features, num_classes)
        )

    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Apply Channel Attention
        attended_features = self.attention(features)
        
        # Global Pooling and Classification
        pooled = self.global_pool(attended_features)
        output = self.classifier(pooled)
        
        return output

class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance.
    L(p_t) = -(1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

if __name__ == "__main__":
    # Test model instantiation
    model = SkinLesionModel(num_classes=10, pretrained=False)
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}") # Expect [1, 10]
