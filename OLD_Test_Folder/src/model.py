import torch
import torch.nn as nn
import timm

class MultimodalMILKNet(nn.Module):
    def __init__(self, backbone_name="efficientnet_b0", tabular_input_dim=16, num_classes=11, pretrained=True):
        super(MultimodalMILKNet, self).__init__()
        
        self.clin_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        self.derm_backbone = timm.create_model(backbone_name, pretrained=pretrained, num_classes=0)
        
        feature_dim = self.clin_backbone.num_features

        self.tab_mlp = nn.Sequential(
            nn.Linear(tabular_input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        total_features = (feature_dim * 2) + 64
        self.head = nn.Sequential(
            nn.Linear(total_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )

    def forward(self, img_clin, img_derm, tabular):
        feat_clin = self.clin_backbone(img_clin)
        feat_derm = self.derm_backbone(img_derm)
        feat_tab = self.tab_mlp(tabular)

        fused = torch.cat([feat_clin, feat_derm, feat_tab], dim=1)
        return self.head(fused)