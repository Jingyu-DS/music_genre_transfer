import torch, torch.nn as nn, torch.nn.functional as F
import torchaudio
from torchaudio.pipelines import WAV2VEC2_BASE
import math

class GenreConditionedW2VEncoder(nn.Module):
    def __init__(self, latent_dim=128, freeze_base=True, num_genres=10, genre_embed_dim=32):
        super().__init__()

        bundle = WAV2VEC2_BASE
        self.w2v = bundle.get_model()

        if freeze_base:
            for p in self.w2v.parameters():
                p.requires_grad = False

        hid = getattr(self.w2v, "encoder_embed_dim", 768)
        
        self.genre_embedding = nn.Embedding(num_genres, genre_embed_dim)
        self.proj = nn.Linear(hid + genre_embed_dim, latent_dim * 2)

    def forward(self, wav16k, genre_labels):
        feats, _ = self.w2v.extract_features(wav16k.squeeze(1))
        pooled = feats[-1].mean(1)
        genre_embed = self.genre_embedding(genre_labels)
        combined = torch.cat([pooled, genre_embed], dim=-1)
        mu, logvar = self.proj(combined).chunk(2, -1)
        std = (0.5 * logvar).exp()
        z = mu + torch.randn_like(std) * std
        return mu, logvar, z

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model))
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos * div), torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, x):
        return x + self.alpha * self.pe[:x.size(1)]

class GenreConditionedFiLM(nn.Module):
    def __init__(self, ch, latent_dim, num_genres=10, genre_embed_dim=32):
        super().__init__()
        self.scale = nn.Linear(latent_dim + genre_embed_dim, ch)
        self.shift = nn.Linear(latent_dim + genre_embed_dim, ch)
        self.genre_embedding = nn.Embedding(num_genres, genre_embed_dim)

    def forward(self, x, z, genre_labels):
        genre_embed = self.genre_embedding(genre_labels)
        z_genre = torch.cat([z, genre_embed], dim=-1)
        s = self.scale(z_genre).unsqueeze(-1)
        b = self.shift(z_genre).unsqueeze(-1)
        return x * (1 + torch.tanh(s)) + b

class GenreConditionedGLUBlock(nn.Module):
    def __init__(self, in_ch, out_ch, up_factor, latent_dim, num_genres=10, genre_embed_dim=32):
        super().__init__()
        self.up_factor = up_factor
        self.pre = nn.utils.weight_norm(nn.Conv1d(in_ch, out_ch*2, 1))
        self.conv_dw = nn.utils.weight_norm(nn.Conv1d(out_ch, out_ch, 3, padding=1, groups=out_ch))
        self.norm = nn.InstanceNorm1d(out_ch)
        self.film = GenreConditionedFiLM(out_ch, latent_dim, num_genres, genre_embed_dim)

    def forward(self, x, z, genre_labels):
        x, gate = self.pre(x).chunk(2, dim=1)
        x = x * torch.sigmoid(gate)
        x = F.interpolate(x, scale_factor=self.up_factor, mode="linear", align_corners=False)
        x = self.norm(F.gelu(self.conv_dw(x)))
        return self.film(x, z, genre_labels)

class MRF(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(ResBlock1D(ch, 3, d), ResBlock1D(ch, 3, d)) for d in (1, 3)
        ])

    def forward(self, x):
        return sum(b(x) for b in self.blocks) / len(self.blocks)

class ResBlock1D(nn.Module):
    def __init__(self, ch, kernel=3, dilation=1):
        super().__init__()
        pad = (kernel - 1) * dilation // 2
        self.conv = nn.utils.weight_norm(nn.Conv1d(ch, ch, kernel, padding=pad, dilation=dilation))
        nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        return x + 0.1 * F.gelu(self.conv(x))

class GenreConditionedDecoder(nn.Module):
    def __init__(self, latent_dim=128, d_model=256, init_len=200, target_len=80000, num_genres=10, genre_embed_dim=32):
        super().__init__()
        self.fc = nn.Linear(latent_dim + genre_embed_dim, d_model * init_len)
        self.pos = PositionalEncoding(d_model, init_len)
        layer = nn.TransformerEncoderLayer(d_model, 4, d_model*2, 0.3, batch_first=True)
        self.xf = nn.TransformerEncoder(layer, num_layers=4)
        chans = [256, 128, 64, 32]
        factors = [10, 10, 8, 10]
        self.ups = nn.ModuleList([GenreConditionedGLUBlock(chans[i], chans[i+1], factors[i], latent_dim, num_genres, genre_embed_dim) for i in range(len(chans)-1)])
        self.mrf = MRF(chans[-1])
        self.head = nn.Conv1d(chans[-1], 1, 3, padding=1)
        self.genre_embedding = nn.Embedding(num_genres, genre_embed_dim)
        self.TARGET_LEN = target_len

    def forward(self, z, genre_labels):
        genre_embed = self.genre_embedding(genre_labels)
        z_genre = torch.cat([z, genre_embed], dim=-1)
        x = self.fc(z_genre).view(z_genre.size(0), -1, self.fc.out_features // 200)
        x = self.xf(self.pos(x)).permute(0, 2, 1)
        for up in self.ups:
            x = up(x, z, genre_labels)
        x = self.mrf(x)
        x = torch.tanh(self.head(x))
        return F.pad(x, (0, max(0, self.TARGET_LEN - x.size(-1))))[:, :, :self.TARGET_LEN]

encoder = GenreConditionedW2VEncoder(latent_dim=256, num_genres=10)
wav16k_dummy = torch.randn(1, 1, 80000)  # (batch, channels, length)
genre_labels_dummy = torch.randint(0, 10, (1,))
    
mean, logvar, latent_encoding = encoder(wav16k_dummy, genre_labels_dummy)
print("Mean Shape:", mean.shape)
print("Log Variance Shape:", logvar.shape)
print("Latent Encoding Shape:", latent_encoding.shape)

decoder = GenreConditionedDecoder(latent_dim=256, num_genres=10)
output_audio = decoder(latent_encoding, genre_labels_dummy)
print(f"Decoder output shape: {output_audio.shape}")

# note: this is a conditional vae, prepare data and trainer has to be modifeied to include genre