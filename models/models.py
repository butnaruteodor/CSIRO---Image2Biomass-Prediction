import torch, torch.nn as nn
import timm,os
from configs.cfg import CFG
import open_clip
from peft import LoraConfig, get_peft_model
import torch.nn.functional as F

class BiomassModelMLP(nn.Module):
    def __init__(self, model_name, freeze_backbone=False, checkpoint_path=None, model_state_dict=None, is_linear=False):
        super().__init__()
        self.is_linear=is_linear
        # 1. Image Backbone
        self.backbone = timm.create_model(
            model_name, pretrained=False, num_classes=0)
        print(f"{CFG.MODEL_NAME} parameters: {sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)}")
        self.load_pretrained()
        torch.save(self.backbone.state_dict(),f"{CFG.MODEL_NAME}.pth")

        if checkpoint_path:
            # SIMPLE LOADING
            weights = torch.load(checkpoint_path, map_location='cpu')
            # strict=False allows ignoring the 'head' layers if dimensions differ
            self.backbone.load_state_dict(weights, strict=True)
        if model_state_dict:
            print("Loading pretrained clip model")
            self.backbone.load_state_dict(model_state_dict, strict=True)

        nf = self.backbone.num_features
        # We have TWO image feature streams (left + right)
        image_feature_dim = nf * 2

        self.head_total = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim//2), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(image_feature_dim//2, image_feature_dim//4), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(image_feature_dim//4, 1)
        )
        self.head_ratios = nn.Sequential(
            nn.Linear(image_feature_dim, image_feature_dim//2), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(image_feature_dim//2, image_feature_dim//4), 
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(image_feature_dim//4, 2),
            nn.Sigmoid() # Forces output between 0 and 1
        )

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
    # def load_pretrained(self, state_dict):
    #     try:
    #         self.backbone.load_state_dict(state_dict, strict=True)
    #     except Exception as e:
    #         print(f"Warning: Pretrained load failed: {e}")

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

        image_features = torch.cat([fl, fr], dim=1)

        p_total = F.softplus(self.head_total(image_features))

        preds = self.head_ratios(image_features)
        r_dead, r_clover = preds.split(1, dim=1)

        p_dead  = p_total * r_dead
        p_clover  = p_total * r_clover

        p_gdm = p_total - p_dead
        p_green = p_gdm - p_clover
        
        return (p_total, p_gdm, p_green)

class BiomassSimpleMLP(nn.Module):
    def __init__(self, image_feature_dim):
        super().__init__()

        self.head_total = self._create_head(image_feature_dim)
        self.head_gdm = self._create_head(image_feature_dim)
        self.head_clover = self._create_head(image_feature_dim)
        self.head_green = self._create_head(image_feature_dim)

    def _create_head(self, feature_dim):
        return nn.Sequential(
        nn.Linear(feature_dim, feature_dim//2), 
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(feature_dim//2, feature_dim//4),
        nn.GELU(),
        nn.Dropout(0.3),
        nn.Linear(feature_dim//4, 1)
        )
    def forward(self, feats):
        p_total = F.softplus(self.head_total(feats))
        p_gdm = F.softplus(self.head_gdm(feats))
        p_clover = F.softplus(self.head_clover(feats))
        p_green = F.softplus(self.head_green(feats))
        p_dead = p_total - p_gdm

        return (p_total, p_gdm, p_green, p_clover, p_dead)

class BiomassOneLayer(nn.Module):
    def __init__(self, image_feature_dim):
        super().__init__()

        self.head_total = self._create_head(image_feature_dim)
        self.head_gdm = self._create_head(image_feature_dim)
        self.head_clover = self._create_head(image_feature_dim)
        self.head_green = self._create_head(image_feature_dim)

    def _create_head(self, feature_dim):
        return nn.Linear(feature_dim, 1)

    def forward(self, feats):
        p_total = F.softplus(self.head_total(feats))
        p_gdm = F.softplus(self.head_gdm(feats))
        p_clover = F.softplus(self.head_clover(feats))
        p_green = F.softplus(self.head_green(feats))

        p_dead = p_total - p_gdm

        return (p_total, p_gdm, p_green, p_clover, p_dead)

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
    
class BiomassCLIP(nn.Module):
    def __init__(self, clip_model, embed_dim):
        super().__init__()
        self.clip = clip_model
        
        # The Learnable Attention Layer
        # It takes a feature vector and outputs a single "importance score"
        self.attention_pool = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 1)
        )

    def encode_image(self, tiles):
        """
        Input: [Num_Tiles, 3, H, W]
        Output: [1, Embed_Dim] (One vector for the whole bag of tiles)
        """
        # 1. Get features for all tiles using the LoRA-CLIP visual tower
        # Shape: [Num_Tiles, Embed_Dim]
        tile_features = self.clip.encode_image(tiles)
        
        # 2. Calculate Attention Scores
        # Shape: [Num_Tiles, 1]
        scores = self.attention_pool(tile_features)
        
        # 3. Softmax over the "Bag" dimension so weights sum to 1
        weights = torch.softmax(scores, dim=0)
        
        # 4. Weighted Sum: Sum(Weight * Feature)
        # Shape: [1, Embed_Dim]
        weighted_avg = torch.sum(weights * tile_features, dim=0, keepdim=True)
        
        return weighted_avg
    
    # Helper to expose internal CLIP methods/attributes
    @property
    def logit_scale(self):
        return self.clip.logit_scale
    
    def encode_text(self, text):
        return self.clip.encode_text(text)
    
def get_lora_model_with_attention():
    print(f"Loading OpenCLIP model: {CFG.CLIP_NAME}...")
    
    # 1. Load Base Model
    base_model, _, preprocess = open_clip.create_model_and_transforms(
        CFG.CLIP_NAME, 
        pretrained=CFG.CLIP_FT_NAME,
        device=CFG.DEVICE
    )
    tokenizer = open_clip.get_tokenizer(CFG.CLIP_NAME)

    # 2. Freeze Base Model Completely
    for param in base_model.parameters():
        param.requires_grad = False
        
    # 3. Apply LoRA to Visual Encoder
    config = LoraConfig(
        r=4, 
        lora_alpha=16,
        target_modules=["fc1", "fc2"], 
        lora_dropout=0.1,
        bias="none"
    )
    base_model.visual = get_peft_model(base_model.visual, config)
    
    # 4. Wrap with Attention Mechanism
    # We fetch the output dim dynamically so it works with any CLIP model
    if hasattr(base_model, 'embed_dim'):
        embed_dim = base_model.embed_dim
    else:
        # Fallback: Run a dummy text to check output size
        # (Safer than running an image which might need resizing)
        print("Inferring embed_dim via dummy forward pass...")
        dummy_text = tokenizer(["test"]).to(CFG.DEVICE)
        with torch.no_grad():
            embed_dim = base_model.encode_text(dummy_text).shape[-1]
    final_model = BiomassCLIP(base_model, embed_dim).to(CFG.DEVICE)
    
    # Print trainable params to verify (LoRA + Attention should be True)
    trainable_params = sum(p.numel() for p in final_model.parameters() if p.requires_grad)
    print(f"Model Ready. Total Trainable Parameters (LoRA + Attention): {trainable_params:,}")
    
    return final_model, preprocess, tokenizer


class LocalMambaBlock(nn.Module):
    """Lightweight Mamba-like block"""

    def __init__(self, dim: int, kernel_size: int = 5, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel_size,
                                 padding=kernel_size // 2, groups=dim)
        self.gate = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm(x)
        g = torch.sigmoid(self.gate(x))
        x = x * g
        x = x.transpose(1, 2)
        x = self.dwconv(x)
        x = x.transpose(1, 2)
        x = self.proj(x)
        x = self.drop(x)
        return shortcut + x
    
class LatentResampler(nn.Module):
    """
    Perceiver-style Resampler.
    Uses 'num_latents' learnable queries to scan the input sequence 
    and extract relevant information via Cross-Attention.
    """
    def __init__(self, input_dim, num_latents=64, num_heads=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # 1. The Latent Queries (The "Scanners")
        # Instead of 1 global token, we have 'num_latents' (e.g., 64)
        # This preserves diversity (one token for dead, one for clover, etc.)
        self.latents = nn.Parameter(torch.randn(1, num_latents, input_dim) * 0.02)
        
        # 2. Cross Attention Layers
        self.q = nn.Linear(input_dim, input_dim)
        self.kv = nn.Linear(input_dim, input_dim * 2)
        self.proj = nn.Linear(input_dim, input_dim)
        
        self.norm_latents = nn.LayerNorm(input_dim)
        self.norm_input   = nn.LayerNorm(input_dim)
        
        # 3. Internal processing for the latents (Self-computation)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.GELU(),
            nn.Linear(input_dim * 4, input_dim)
        )
        self.norm_post = nn.LayerNorm(input_dim)

    def forward(self, x):
        # x: [Batch, N_Input_Tokens (2048), Dim]
        B, N, C = x.shape
        
        # Expand latents to batch size
        # latents: [Batch, 64, Dim]
        latents = self.latents.expand(B, -1, -1)
        
        # --- CROSS ATTENTION ---
        # Query = Latents (What we want to find)
        # Key/Value = Input Image Patches (Where we look)
        
        # Apply Norms before attention (Pre-Norm architecture)
        q_in = self.norm_latents(latents)
        kv_in = self.norm_input(x)
        
        q = self.q(q_in).reshape(B, self.latents.shape[1], self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        kv = self.kv(kv_in).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        
        # Attention Score: [Batch, Heads, Num_Latents, N_Inputs]
        # This is where the magic happens: deciding which pixels matter for which latent
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Gather information
        latents_out = (attn @ v).transpose(1, 2).reshape(B, self.latents.shape[1], C)
        latents_out = self.proj(latents_out)
        
        # Residual connection
        latents = latents + latents_out
        
        # Feed Forward (MLP) to process the extracted info
        latents = latents + self.mlp(self.norm_post(latents))
        
        # Output: [Batch, Num_Latents (64), Dim]
        return latents