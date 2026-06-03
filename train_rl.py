import torch, torch.nn as nn, numpy as np
class PPOAgent(nn.Module):
    def __init__(self, s=64, a=4, h=256):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(s,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,a), nn.Softmax(-1))
        self.critic = nn.Sequential(nn.Linear(s,h), nn.ReLU(), nn.Linear(h,h), nn.ReLU(), nn.Linear(h,1))
    def forward(self, x): return self.actor(x), self.critic(x)
if __name__ == "__main__":
    d = "cuda" if torch.cuda.is_available() else "cpu"
    m = PPOAgent().to(d)
    print(f"PPO on {torch.cuda.get_device_name(0)}, {sum(p.numel() for p in m.parameters())/1e6:.1f}M params")