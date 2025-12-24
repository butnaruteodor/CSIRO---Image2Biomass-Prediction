import torch, torch.nn as nn
import timm,os
from configs.cfg import CFG
import open_clip
from peft import LoraConfig, get_peft_model

class BiomassModelMLP(nn.Module):
    def __init__(self, model_name, freeze_backbone=False, checkpoint_path=None, model_state_dict=None):
        super().__init__()
        
        # 1. Image Backbone
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0, global_pool='avg')
        
        # self.load_pretrained()

        if checkpoint_path:
            # SIMPLE LOADING
            weights = torch.load(checkpoint_path, map_location='cpu')
            # strict=False allows ignoring the 'head' layers if dimensions differ
            self.backbone.load_state_dict(weights, strict=True)
        if model_state_dict:
            print("Loading pretrained clip model")
            self.backbone.load_state_dict(model_state_dict, strict=True)

        # self.backbone.avg_pool=GeM()
        nf = self.backbone.num_features
        
        # We have TWO image feature streams (left + right)
        image_feature_dim = nf * 2

        # 3. Main Head
        self.head = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim//2), 
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(image_feature_dim//2, image_feature_dim//4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3)
        )
        self.regressor = nn.Linear(image_feature_dim//4, 3)

        if freeze_backbone:
            self.freeze_backbone()

    def load_pretrained(self):
        try:
            # Note: Ensure CFG is accessible or pass model_name here
            state_dict = timm.create_model(self.backbone.default_cfg['architecture'], pretrained=True, num_classes=0).state_dict()
            self.backbone.load_state_dict(state_dict, strict=True)
            print("Pretrained weights loaded (CPU)")
        except Exception as e:
            print(f"Warning: Pretrained load failed: {e}")
    def load_pretrained(self, state_dict):
        try:
            self.backbone.load_state_dict(state_dict, strict=True)
        except Exception as e:
            print(f"Warning: Pretrained load failed: {e}")

    def freeze_backbone(self):
        print("Freezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = False
            
    def unfreeze_backbone(self):
        print("Unfreezing backbone parameters.")
        for param in self.backbone.parameters():
            param.requires_grad = True

    def forward(self, left, right):
        # 1. Extract Raw Image Features
        fl = self.backbone(left)
        fr = self.backbone(right)
        image_features = torch.cat([fl, fr], dim=1) # [B, 1536] (if ConvNeXt-Tiny)

        safe_features = image_features
        fused = self.head(safe_features)
        predictions = self.regressor(fused)
        
        p_total, p_gdm, p_green = predictions.split(1, dim=1)
        
        return (p_total, p_gdm, p_green)
    

def get_lora_model():
    print(f"Loading OpenCLIP model: {CFG.CLIP_NAME}...")
    
    # 1. Load Model via OpenCLIP
    model, _, preprocess = open_clip.create_model_and_transforms(
        CFG.CLIP_NAME, 
        pretrained=CFG.CLIP_FT_NAME,
        device=CFG.DEVICE
    )
    tokenizer = open_clip.get_tokenizer(CFG.CLIP_NAME)

    # 3. Freeze Everything
    for param in model.parameters():
        param.requires_grad = False
        
    # 4. Apply LoRA to Visual Encoder ONLY
    # ConvNeXt uses 'fc1', 'fc2' in its MLP blocks
    config = LoraConfig(
        r=4, 
        lora_alpha=16,
        target_modules=["fc1", "fc2"], 
        lora_dropout=0.1,
        bias="none"
    )
    
    # Wrap the visual tower specifically
    model.visual = get_peft_model(model.visual, config)
    
    model.visual.print_trainable_parameters()
    return model, preprocess, tokenizer