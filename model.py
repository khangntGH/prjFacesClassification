import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerEncoder(nn.Module):
    def __init__(self, num_heads, embed_size):
        super(TransformerEncoder, self).__init__()
        self.transformer = nn.Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=1)
        self.fc = nn.Linear(embed_size, embed_size)

    def forward(self, x):
        batch = x.shape[0]
        x = x.view(2, batch, 64) 
        x = self.transformer(x, tgt=x)
        x = x.mean(dim=0)
        return self.fc(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, inputsize, embed_size):
        super(ResidualBlock, self).__init__()
        self.FC1 = nn.Linear(inputsize, 64)
        self.BN = nn.BatchNorm1d(64)
        self.GLU = nn.ReLU()
        self.FC2 = nn.Linear(64, embed_size)
        self.BN2 = nn.BatchNorm1d(embed_size)

    def forward(self, x):
        x = self.FC1(x)
        x = self.BN(x)
        x = self.GLU(x)
        residual_brand = x
        x  = self.FC2(x)
        x = self.BN2(x)
        x = self.GLU(x)

        output = x + residual_brand
        return output


class FaciesClassification(nn.Module):
    def __init__(self, num_categories, num_numerical_features, n_class):
        super(FaciesClassification, self).__init__()
        
        self.category_embedding = nn.Embedding(20, 64)
        self.transformer_encoder = TransformerEncoder(num_heads=4, embed_size=64)
        self.fc_numerical = ResidualBlock(num_numerical_features, 64)

        # MLP
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, n_class)

    def forward(self, category_data, numerical_data):
        category_embedded = self.category_embedding(category_data.long()).unsqueeze(1) 
        category_output = self.transformer_encoder(category_embedded)

        numerical_output = F.relu(self.fc_numerical(numerical_data))

        combined = torch.cat((category_output, numerical_output), dim=1)
        x = F.relu(self.fc1(combined))
        output = self.fc2(x)
        return output
