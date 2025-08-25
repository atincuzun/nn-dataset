import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


def supported_hyperparameters():
    return {'lr', 'momentum'}

class BottleneckWithBottleneckConvs(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BottleneckWithBottleneckConvs, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64)) * groups
        self.conv1 = nn.Conv2d(inplanes, width, kernel_size=1, bias=False)
        self.bn1 = norm_layer(width)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride, padding=1, groups=groups, bias=False)
        self.bn2 = norm_layer(width)
        self.conv3 = nn.Conv2d(width, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = norm_layer(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.bn3(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class CustomResNet50(torchvision.models.resnet.ResNet):
    def __init__(self, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(CustomResNet50, self).__init__(
            block=BottleneckWithBottleneckConvs,
            layers=[3, 4, 6, 3],
            num_classes=num_classes,
            zero_init_residual=zero_init_residual,
            groups=groups,
            width_per_group=width_per_group,
            replace_stride_with_dilation=replace_stride_with_dilation,
            norm_layer=norm_layer)
        self.avgpool = nn.Identity()
        self.fc = nn.Identity()

class ResNetSpatialEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super().__init__()
        backbone = CustomResNet50()
        modules = list(backbone.children())
        self.cnn = nn.Sequential(*modules)
        self.fc = nn.Linear(2048, output_dim)

    def forward(self, x):
        x = self.cnn(x)
        B, C, H, W = x.size()
        x = x.view(B, C, H*W)
        x = x.permute(0, 2, 1)
        x = self.fc(x)
        return x

class SpatialAttentionLSTMDecoder(nn.Module):
    def __init__(self, vocab_size, feature_dim=768, hidden_size=768, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.attn_linear = nn.Linear(feature_dim + hidden_size, hidden_size)
        self.attn_v = nn.Linear(hidden_size, 1)
        self.lstm = nn.LSTMCell(hidden_size + feature_dim, hidden_size)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self.hidden_size = hidden_size

    def init_zero_hidden(self, batch, device):
        h0 = torch.zeros(batch, self.hidden_size, device=device)
        c0 = torch.zeros(batch, self.hidden_size, device=device)
        return (h0, c0)

    def forward(self, features, captions, hidden_state=None):
        B, num_regions, feature_dim = features.size()
        seq_len = captions.size(1)

        if hidden_state is None:
            h = features.mean(dim=1)
            h = torch.tanh(h)
            c = torch.zeros(B, self.hidden_size, device=features.device)
        else:
            h, c = hidden_state

        embeddings = self.embedding(captions)
        outputs = []
        for t in range(seq_len):
            emb_t = embeddings[:, t, :]
            h_exp = h.unsqueeze(1).expand(-1, num_regions, -1)
            attn_input = torch.cat([features, h_exp], dim=2)
            attn_hidden = torch.tanh(self.attn_linear(attn_input))
            attn_scores = self.attn_v(attn_hidden).squeeze(2)
            alpha = torch.softmax(attn_scores, dim=1)
            context = (features * alpha.unsqueeze(2)).sum(dim=1)

            lstm_input = torch.cat([emb_t, context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            out_t = self.fc(self.dropout(h))
            outputs.append(out_t.unsqueeze(1))

        outputs = torch.cat(outputs, dim=1)
        return outputs, (h, c)

    def step(self, input_token, features, h, c):
        # Ensure shape [B], not [B,1]
        input_token = input_token.squeeze(-1).contiguous().view(-1).long()
        emb = self.embedding(input_token)        # [B, hidden]
        B, num_regions, feature_dim = features.size()
        h_exp = h.unsqueeze(1).expand(-1, num_regions, -1)
        attn_input = torch.cat([features, h_exp], dim=2)
        attn_hidden = torch.tanh(self.attn_linear(attn_input))
        attn_scores = self.attn_v(attn_hidden).squeeze(2)
        alpha = torch.softmax(attn_scores, dim=1)
        context = (features * alpha.unsqueeze(2)).sum(dim=1)   # [B, feature_dim]
        lstm_input = torch.cat([emb, context], dim=1)          # [B, hidden + feature_dim]
        h, c = self.lstm(lstm_input, (h, c))
        logits = self.fc(self.dropout(h))                      # [B, vocab]
        return logits, (h, c)

class Net(nn.Module):
    def __init__(self, in_shape: tuple, out_shape: tuple, prm: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.hidden_size = 768
        self.vocab_size = out_shape[0]
        self.cnn = ResNetSpatialEncoder(self.hidden_size)
        self.rnn = SpatialAttentionLSTMDecoder(
            self.vocab_size,
            feature_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=1,
            dropout=0.3
        )

    def forward(self, images, captions=None, hidden_state=None):
        features = self.cnn(images)
        batch_size = features.size(0)

        if captions is not None:
            captions = captions[:, 0, :] if captions.dim() == 3 else captions
            inputs = captions[:, :-1]
            targets = captions[:, 1:]
            outputs, _ = self.rnn(features, inputs, hidden_state)
            assert outputs.shape[1] == targets.shape[1]
            return outputs, targets
        else:
            max_len = 20
            start_idx = 1                           
            inputs = torch.full((batch_size,), start_idx, dtype=torch.long, device=self.device)
            captions_gen = []
            h, c = hidden_state if hidden_state is not None else (None, None)
            if h is None:
                h = features.mean(dim=1)
                h = torch.tanh(h)
                c = torch.zeros_like(h)
            for _ in range(max_len):
                logits, (h, c) = self.rnn.step(inputs, features, h, c)  # inputs: [B]
                inputs = logits.argmax(dim=1)                           # [B]
                captions_gen.append(inputs.unsqueeze(1))                   # collect as [B,1]
            return torch.cat(captions_gen, dim=1) 

    def train_setup(self, prm):
        self.to(self.device)
        self.criteria = (nn.CrossEntropyLoss().to(self.device),)
        self.optimizer = torch.optim.SGD(self.parameters(), lr=prm['lr'], momentum=prm['momentum'])
        self.scaler = GradScaler()
        self.device_type = self.device.type 

    def learn(self, train_data):
        self.train()
        for images, captions in train_data:
            images, captions = images.to(self.device), captions.to(self.device)
            with autocast(device_type=self.device_type, enabled=(self.device_type == "cuda")):
                outputs, targets = self.forward(images, captions)
                loss = self.criteria[0](outputs.reshape(-1, outputs.size(-1)), targets.reshape(-1))
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            nn.utils.clip_grad_norm_(self.parameters(), 3)
            self.scaler.step(self.optimizer)
            self.scaler.update()

    @staticmethod
    def init_zero_hidden(batch, device):
        h0 = torch.zeros(batch, 768, device=device)
        c0 = torch.zeros(batch, 768, device=device)
        return h0, c0