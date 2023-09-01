import torch
import torch.nn as nn
import torch.nn.functional as F
from .tools import Linear_fw
class additionFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, t,  v):
        return t + v

class MultiHeadAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_heads=3):
        super(MultiHeadAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        self.query = Linear_fw(input_size, hidden_size * num_heads)
        self.key = Linear_fw(input_size, hidden_size * num_heads)
        self.value = Linear_fw(input_size, hidden_size * num_heads)
        self.output = Linear_fw(hidden_size * num_heads, input_size)
        
    def forward(self, t, v):
        x = torch.cat((t, v), 1)
        x = torch.unsqueeze(x, dim=1)
        batch_size, seq_len, input_size = x.size()
        assert input_size == self.input_size, "Input size doesn't match."
        
        q = self.query(x).view(batch_size, seq_len, self.hidden_size, self.num_heads).transpose(1, 2)  # [batch_size, num_heads, seq_len, hidden_size]
        k = self.key(x).view(batch_size, seq_len, self.hidden_size, self.num_heads).transpose(1, 2)  # [batch_size, num_heads, seq_len, hidden_size]
        v = self.value(x).view(batch_size, seq_len, self.hidden_size, self.num_heads).transpose(1, 2)  # [batch_size, num_heads, seq_len, hidden_size]
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.hidden_size, dtype=torch.float32, device=x.device))
        # scores: [batch_size, num_heads, seq_len, seq_len]
        attention = F.softmax(scores, dim=-1)
        # attention: [batch_size, num_heads, seq_len, seq_len]
        
        context = torch.matmul(attention, v)  # [batch_size, num_heads, seq_len, hidden_size]
        
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size * self.num_heads)
        # context: [batch_size, seq_len, hidden_size * num_heads]

        output = self.output(context)
        # output: [batch_size, seq_len, input_size]
        output = torch.squeeze(output, dim=1)
        return output

class multipleFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, t,  v):
        return t *  v

class concatFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, t, v):
        return torch.cat((t, v), 1)

class tensorFusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

    def forward(self, t, v):
        batch_size = t.data.shape[0]
        # next we perform "tensor fusion", which is essentially appending 1s to the tensors and take Kronecker product
        add_one = torch.ones(size=[batch_size, 1], requires_grad=False).type_as(t).to(t.device)
        _text_h = torch.cat((add_one, t), dim=1)
        _image_h = torch.cat((add_one, v), dim=1)

        fusion_tensor = torch.bmm(_text_h.unsqueeze(2), _image_h.unsqueeze(1))
        
        fusion_tensor = fusion_tensor.view(-1, (self.args.embd_size + 1) ** 2)
        return fusion_tensor
