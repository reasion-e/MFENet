import torch
from torch import nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, seq_len, pred_len):
        super(Net, self).__init__()
        # Parameters
        self.pred_len = pred_len

        # MSP-FEM
        #1
        self.fc5 = nn.Linear(seq_len, pred_len * 4)
        self.avgpool1 = nn.AvgPool1d(kernel_size=2)
        self.ln1 = nn.LayerNorm(pred_len * 2)
        self.fc6 = nn.Linear(pred_len * 2, pred_len)
        self.avgpool2 = nn.AvgPool1d(kernel_size=2)
        self.ln2 = nn.LayerNorm(pred_len // 2)
        self.fc7 = nn.Linear(pred_len // 2, pred_len)

        #2
        self.fc10 = nn.Linear(seq_len, pred_len * 2)
        self.avgpool3 = nn.AvgPool1d(kernel_size=2)
        self.ln3 = nn.LayerNorm(pred_len)
        self.fc11 = nn.Linear(pred_len, pred_len // 2)
        self.avgpool4 = nn.AvgPool1d(kernel_size=2)
        self.ln4 = nn.LayerNorm(pred_len // 4)
        self.fc12 = nn.Linear(pred_len // 4, pred_len)

        # UFE-M
        #1
        self.fc13 = nn.Linear(seq_len, pred_len // 4)
        self.ln5 = nn.LayerNorm(pred_len // 2)
        self.fc14 = nn.Linear(pred_len // 2, pred_len)
        self.ln6 = nn.LayerNorm(pred_len * 2)
        self.fc15 = nn.Linear(pred_len * 2, pred_len)

        #2

        self.fc16 = nn.Linear(seq_len,pred_len//2)
        self.ln7 = nn.LayerNorm(pred_len)
        self.fc17 = nn.Linear(pred_len,pred_len)
        self.ln8 = nn.LayerNorm(pred_len*2)
        self.fc18 = nn.Linear(pred_len*2,pred_len)
        # Streams Concatination
        self.fc8 = nn.Linear(pred_len * 4, pred_len)

    def forward(self, t):
        # x: [Batch,  Channel ,Input]

        # Channel split for channel independence
        B = t.shape[0]  # Batch size
        C = t.shape[1]  # Channel size
        I = t.shape[2]  # Input size

        t = torch.reshape(t, (B * C, I))  # [Batch and Channel, Input]
        t1 = t
        t2 = t
        t4 = t

        # MSP-FEM
        t = self.fc5(t)
        t = self.avgpool1(t)
        t = self.ln1(t)
        t = self.fc6(t)
        t = self.avgpool2(t)
        t = self.ln2(t)
        t = self.fc7(t)

        # MLP
        t1 = self.fc10(t1)
        t1 = self.avgpool3(t1)
        t1 = self.ln3(t1)
        t1 = self.fc11(t1)
        t1 = self.avgpool4(t1)
        t1 = self.ln4(t1)
        t1 = self.fc12(t1)

        # UFE-M
        t2 = self.fc13(t2).unsqueeze(1)
        t2 = F.interpolate(t2, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        t2 = self.ln5(t2)
        t2 = self.fc14(t2).unsqueeze(1)
        t2 = F.interpolate(t2, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        t2 = self.ln6(t2)
        t2 = self.fc15(t2)

        t4 = self.fc16(t4).unsqueeze(1)
        t4 = F.interpolate(t4, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        t4 = self.ln7(t4)
        t4 = self.fc17(t4).unsqueeze(1)
        t4 = F.interpolate(t4, scale_factor=2, mode="linear", align_corners=True).squeeze(1)
        t4 = self.ln8(t4)
        t4 = self.fc18(t4)

        t = torch.cat((t,t2,t1,t4), dim=1)

        # Streams Concatination
        x = self.fc8(t)

        # Channel concatination
        x = torch.reshape(x, (B, C, self.pred_len))  # [Batch, Channel, Output]

        x = x.permute(0, 2, 1)  # to [Batch, Output, Channel]

        return x