import torch
import timm

class TimmCNNEncoder(torch.nn.Module):
    def __init__(self, model_name: str = 'resnet50', 
                 kwargs: dict = {'features_only': True, 'out_indices': (3,), 'pretrained': True, 'num_classes': 0}, 
                 pool: bool = True):
        super().__init__()
        assert kwargs.get('pretrained', False), 'only pretrained models are supported'
        
        # 🔍 Debugging: Print the original model name
        print(f"🚨 Debug: Original model_name = {model_name}")
        
        # ✅ Force valid model name
        if model_name in ["resnet50_trunc", "resnet50.tv_in1k"]:
            model_name = "resnet50"  # ✅ Change to a valid timm model
        
        print(f"✅ Debug: Using timm model_name = {model_name}")  # Debugging
        
        # 🔥 Load the correct model from timm
        self.model = timm.create_model(model_name, **kwargs)
        self.model_name = model_name

        # 🔍 Check if pooling is required
        if pool:
            self.pool = torch.nn.AdaptiveAvgPool2d(1)
        else:
            self.pool = None
    
    def forward(self, x):
        out = self.model(x)
        if isinstance(out, list):
            assert len(out) == 1
            out = out[0]
        if self.pool:
            out = self.pool(out).squeeze(-1).squeeze(-1)
        return out
