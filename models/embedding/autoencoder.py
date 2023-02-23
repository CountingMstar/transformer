import torch
from torch import nn


class LinearLayer(nn.Module):
    def __init__(self, embedding, dropout=0.1):
        super(LinearLayer, self).__init__()

        x_embedding, y_embedding = embedding.shape
        self.w_1 = nn.Linear(y_embedding, 512)
        self.w_2 = nn.Linear(512, int(y_embedding/2))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.w_1(x)
        x = self.w_2(x)
        x = self.dropout(x)
        return x

# 오토인코더 모듈 정의
class AutoEncoder(nn.Module):
    def __init__(self, embedding, dropout=0.1):
        super(AutoEncoder, self).__init__()

        x_embedding, y_embedding = embedding.shape
        #인코더는 간단한 신경망으로 분류모델처럼 생겼습니다.
        self.encoder = nn.Sequential( # nn.Sequential을 사용해 encoder와 decoder 두 모듈로 묶어줍니다.
            nn.Linear(y_embedding, 512), #차원을 28*28에서 점차 줄여나갑니다.
            nn.ReLU(),
            nn.Linear(512, int(y_embedding/2)),   # 입력의 특징을 3차원으로 압축합니다 (출력값이 바로 잠재변수가 됩니다.)
        )
        self.decoder = nn.Sequential(
            nn.Linear(int(y_embedding/2), 512), #디코더는 차원을 점차 28*28로 복원합니다.
            nn.ReLU(),
            nn.Linear(512, y_embedding)
        )

    def forward(self, x):
        encoded = self.encoder(x) # encoder는 encoded라는 잠재변수를 만들고
        decoded = self.decoder(encoded) # decoder를 통해 decoded라는 복원이미지를 만듭니다.
        return encoded, decoded