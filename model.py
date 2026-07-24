import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_revgrad import RevGrad
import math

class DANNAlphaScheduler:
    def __init__(self, total_steps, gamma=10.0, max_alpha=1.0):
        self.total_steps = total_steps
        self.gamma = gamma
        self.max_alpha = max_alpha
        self.current_step = 0

    def step(self):
        """
        Calculate the current alpha based on training progress and increment the step.
        Should be called once per training batch.
        """
        # Calculate progress 'p' as a float between 0.0 and 1.0
        p = float(self.current_step) / self.total_steps
        p = min(max(p, 0.0), 1.0) 
        
        # Calculate alpha using the DANN formula
        alpha = (2.0 / (1.0 + math.exp(-self.gamma * p)) - 1.0) * self.max_alpha
        
        self.current_step += 1
        return float(alpha)


class SVModel(nn.Module):
    def __init__(self, model_name: str = "redimnet-b6"):
        super().__init__()
        self.model_name = model_name
        self.encoder = self._load_model()

    def _load_model(self):
        if self.model_name == "redimnet-b6":
            model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name="b6", 
                       train_type="ft_lm", 
                       dataset="vox2")
            self.in_channels = 192
        elif self.model_name == "redimnet-b4":
            model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name="b4", 
                       train_type="ft_lm", 
                       dataset="vox2")
            self.in_channels = 192
        elif self.model_name == "redimnet-b2":
            model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name="b2", 
                       train_type="ft_lm", 
                       dataset="vox2")
            self.in_channels = 192
        
        return model

    def forward(self, x):
        return self.encoder(x)
    
class PostProcessor(nn.Module):
    def __init__(self, sv_model: str = "redimnet-b6",  hidden_dim: int = 128, bottleneck_dim: int = 64, dropout: float = 0.3):
        super().__init__()

        self.sv_model = SVModel(sv_model)
        self.in_channels = self.sv_model.in_channels
        self.encoder = nn.Sequential(
            nn.Linear(self.in_channels, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, bottleneck_dim),
            nn.Dropout(dropout),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, hidden_dim),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.in_channels)
        )
    
    def encode(self, x):
        emb = self.sv_model(x)
        return self.encoder(emb), emb
    
    def decode(self, x, residual):
        return self.decoder(x) + residual

    def forward(self, x):
        return self.decode(self.encode(x), x)


class GRLStyleClassifier(nn.Module):
    def __init__(self, bottleneck_dim):
        super().__init__()

        self.grl = RevGrad()

        self.clf_head = nn.Sequential(
            nn.Linear(bottleneck_dim, bottleneck_dim//2),
            nn.ReLU(),
            nn.Linear(bottleneck_dim//2, 2)
        )

    def forward(self, x):
        return self.grl(self.clf_head(x))
    

######### SPEAKER HEADS ###########

class AAMSoftmax(nn.Module):
    def __init__(self, emb_dim: int, n_speakers: int, scale: float, margin: float):
        super().__init__()

        self.emb_dim = emb_dim
        self.n_speakers = n_speakers
        self.scale = scale
        self.margin = margin
        
        self.weight = nn.Parameter(torch.empty(n_speakers, emb_dim, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight)


        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)

        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(self, x, labels):

        x_norm = F.normalize(x, dim = 1)
        w_norm = F.normalize(self.weight, dim = 1)

        cosine = F.linear(x_norm, w_norm)
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))

        phi = cosine * self.cos_m - sine * self.sin_m

        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=x.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        logits = output * self.scale
        
        return logits
    

class CosineSoftmax(nn.Module):
    def __init__(self, emb_dim, n_speakers=1000, scale=30.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.weight = nn.Parameter(torch.empty(n_speakers, emb_dim, dtype=torch.float32))
        nn.init.xavier_uniform_(self.weight)
        self.scale = scale

    def forward(self, x, labels): # labels - to match AAMSoftmax forward structure
        w = F.normalize(self.weight, dim=1)
        x = F.normalize(x, dim = 1)
        logits = self.scale * (x @ w.t())
        return logits



if __name__ == "__main__":
    cs = CosineSoftmax(emb_dim=192, n_speakers=5, scale=30.0)

    pp = PostProcessor()



    x = torch.rand((32,500))

    pp(x)
    # print(out.shape)
