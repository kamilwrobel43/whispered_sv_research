import math
import torch
import torch.nn as nn
import torch.nn.functional as F



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
        elif self.model_name == "redimnet-b4":
            model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name="b4", 
                       train_type="ft_lm", 
                       dataset="vox2")
        elif self.model_name == "redimnet-b2":
            model = torch.hub.load('IDRnD/ReDimNet', 'ReDimNet', 
                       model_name="b2", 
                       train_type="ft_lm", 
                       dataset="vox2")
        
        return model

    def forward(self, x):
        return self.encoder(x)

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

    x = torch.rand((32,192))
    out = cs(x, x)
    print(out.shape)
