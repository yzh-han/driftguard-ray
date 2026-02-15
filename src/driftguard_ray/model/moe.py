from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

@dataclass
class GateArgs:
    in_dim: int
    num_exp: int
    topk: int
    hidden_dim: int = 16
    eps: float = 1e-12

    def __post_init__(self):
        self.out_dim = self.num_exp

class Gate(nn.Module):
    """
    Gating mechanism for Mixture of Experts.
    """
    
    def __init__(self, args: GateArgs):
        super().__init__()
        self.num_exp = args.num_exp
        self.topk = args.topk
        self.eps = args.eps

        self.fc1 = nn.Linear(args.in_dim, args.hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(args.hidden_dim, args.out_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the gating mechanism.
        """
        # x: [B, C, H, W] -> [B, C]
        # x = x[:, 0]
        # pooled = F.adaptive_avg_pool2d(x, 1).squeeze(-1).squeeze(-1)
        assert x.dim() == 2, "Gate Input should be [B, T]"

        x = self.fc2(self.relu(self.fc1(x)))
        x = torch.softmax(x, dim=-1)

        vars, idxs = x.topk(self.topk, dim=-1)
        x = torch.zeros_like(x).scatter_(-1, idxs, vars)
        x = x / (x.sum(dim=-1, keepdim=True) + self.eps)

        return x  # [B, num_exp]

# def balance_loss(gate_list: List[torch.Tensor]) -> torch.Tensor:
#     """
#     Compute balance loss to encourage uniform expert utilization.
    
#     This loss function encourages the gating mechanism to use all experts
#     equally by minimizing the KL divergence between the average expert
#     usage and a uniform distribution.
    
#     Args:
#         gate_list: List of gate weight tensors from different layers
        
#     Returns:
#         Average balance loss across all gate layers
#     """
#     loss = 0
#     for gate_weights in gate_list:
#         # Compute average expert utilization across batch
#         expert_prob = gate_weights.mean(dim=0)
#         # Target uniform distribution
#         target = torch.full_like(expert_prob, 1.0 / expert_prob.size(0))
#         # KL divergence loss
#         loss += F.kl_div((expert_prob + 1e-8).log(), target, reduction='batchmean')
#     return loss / len(gate_list)