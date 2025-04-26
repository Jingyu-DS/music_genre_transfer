import torch
import torch.nn as nn
import torch
import torch.nn as nn

class HierarchicalEncoder(nn.Module):
    def __init__(self, n_mels=128, segment_length=16, segment_hidden=256, latent_dim=128):
        super().__init__()
        self.segment_length = segment_length
        self.segment_hidden = segment_hidden

        # First-level encoder (segment-wise GRU)
        self.segment_encoder = nn.GRU(n_mels, segment_hidden, batch_first=True)

        # Second-level encoder (over segments)
        self.high_encoder = nn.GRU(segment_hidden, segment_hidden, batch_first=True)

        # Latent projection
        self.fc_mean = nn.Linear(segment_hidden, latent_dim)
        self.fc_logvar = nn.Linear(segment_hidden, latent_dim)

    def forward(self, x):
        batch, seq_len, n_mels = x.size()

        # Reshape input into segments
        num_segments = seq_len // self.segment_length
        x = x[:, :num_segments * self.segment_length, :]
        x = x.view(batch, num_segments, self.segment_length, n_mels)

        # Encode segments individually
        segments_encoded = []
        for i in range(num_segments):
            segment = x[:, i, :, :]  # [batch, segment_length, n_mels]
            _, h = self.segment_encoder(segment)
            segments_encoded.append(h.squeeze(0))

        segments_encoded = torch.stack(segments_encoded, dim=1)  # [batch, num_segments, segment_hidden]

        # Encode the segment representations
        _, h_high = self.high_encoder(segments_encoded)
        h_high = h_high.squeeze(0)

        mean = self.fc_mean(h_high)
        logvar = self.fc_logvar(h_high)
        z = mean + torch.exp(0.5 * logvar) * torch.randn_like(logvar)

        return mean, logvar, z

class HierarchicalDecoder(nn.Module):
    def __init__(self, latent_dim=128, n_mels=128, segment_length=16, segment_hidden=256, seq_len=128):
        super().__init__()
        self.segment_length = segment_length
        self.num_segments = seq_len // segment_length
        self.segment_hidden = segment_hidden

        # Project latent vector to segment embeddings
        self.fc = nn.Linear(latent_dim, self.num_segments * segment_hidden)

        # Segment-wise decoder GRU
        self.segment_decoder = nn.GRU(segment_hidden, segment_hidden, batch_first=True)

        # Output projection to spectrogram segments
        self.fc_out = nn.Linear(segment_hidden, n_mels)

    def forward(self, z):
        batch = z.size(0)

        # Project latent vector to segment embeddings
        segments_emb = self.fc(z).view(batch, self.num_segments, self.segment_hidden)

        outputs = []
        for i in range(self.num_segments):
            segment_emb = segments_emb[:, i, :].unsqueeze(1).repeat(1, self.segment_length, 1)
            decoded_segment, _ = self.segment_decoder(segment_emb)
            decoded_segment = self.fc_out(decoded_segment)  # [batch, segment_length, n_mels]
            outputs.append(decoded_segment)

        reconstructed = torch.cat(outputs, dim=1)  # [batch, seq_len, n_mels]

        return torch.sigmoid(reconstructed)