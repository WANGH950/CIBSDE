import torch
import torch.nn as nn

def euclid_corr(x):
    x_cos = torch.cos(x)
    x_sin = torch.sin(x)
    return torch.cat([x_sin[:,:1]*x_cos[:,1:],x_sin[:,:1]*x_sin[:,1:],x_cos[:,:1]],dim=1)

def polar_corr(x):
    theta = torch.pi/2 - torch.arctan(x[:,2:3]/x[:,:2].norm(dim=1,keepdim=True))
    phi = torch.pi/2 - torch.arctan(x[:,:1]/x[:,1:2]) + (x[:,1:2]<0)*torch.pi
    phi[torch.isnan(phi)] = 0.
    return torch.cat([theta,phi],dim=1)

class FNN(nn.Module):
    def __init__(self, in_dim, out_dim, hid_dim=32, num_layers=5) -> None:
        super(FNN, self).__init__()
        if num_layers == 0:
            self.fnn = nn.Linear(in_dim, out_dim)
        else:
            fnn_layers = [nn.Sequential(
                    nn.Linear(in_dim,hid_dim+out_dim+in_dim),
                    nn.Tanh()
                )] + [nn.Sequential(
                    nn.Linear(hid_dim+out_dim+in_dim,hid_dim+out_dim+in_dim),
                    nn.Tanh()
                ) for _ in range(num_layers)] + [
                    nn.Linear(hid_dim+out_dim+in_dim,out_dim)
                ]
            self.fnn = nn.Sequential(*fnn_layers)

    def forward(self, x):
        y = self.fnn(x)
        return y
    
class Parameter(nn.Module):
    def __init__(self, out_dim):
        super().__init__()
        self.out_dim = out_dim
        self.res = nn.Parameter(torch.randn(out_dim),requires_grad=True)
    
    def forward(self, x):
        B = x.shape[0]
        return self.res.expand([B,self.out_dim])