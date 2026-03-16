import torch
import torch.nn as nn
import cibsde.utils as utils
from tqdm import tqdm

class CIBSDE(nn.Module):
    def __init__(self, d, t, f, mu, sigma, h, pc, data_gen, N, refb, hitb, param=False):
        super().__init__()
        self.d = d
        self.t = t
        self.f = f
        self.mu = mu
        self.sigma = sigma
        self.h = h
        self.pc = pc
        self.data_gen = data_gen
        self.N = N
        self.refb = refb
        self.hitb = hitb
        self.param = param
        self.dt = t / N
        self.ones = nn.Parameter(torch.ones([1]),requires_grad=False)
        self.zeros = nn.Parameter(torch.zeros([1]),requires_grad=False)
    
    def T_inv(self, xt):
        B = xt.shape[0]
        ones = self.ones.expand([B,1])
        zeros = self.zeros.expand([B,1])
        x_polar = utils.polar_corr(xt)
        x_polar[:,0] -= torch.pi/2
        trans1 = torch.cat([torch.cat([torch.cos(x_polar[:,:1]),zeros,torch.sin(x_polar[:,:1])],dim=1).unsqueeze(1),torch.cat([zeros,ones,zeros],dim=1).unsqueeze(1),torch.cat([-torch.sin(x_polar[:,:1]),zeros,torch.cos(x_polar[:,:1])],dim=1).unsqueeze(1)],dim=1)
        trans2 = torch.cat([torch.cat([torch.cos(x_polar[:,1:]),-torch.sin(x_polar[:,1:]),zeros],dim=1).unsqueeze(1),torch.cat([torch.sin(x_polar[:,1:]),torch.cos(x_polar[:,1:]),zeros],dim=1).unsqueeze(1),torch.cat([zeros,zeros,ones],dim=1).unsqueeze(1)],dim=1)
        return torch.bmm(trans2,trans1)


class BoundaryIBSDE(CIBSDE):
    def __init__(self, d, t, f, mu, sigma, h, pc, data_gen, N, refb, hitb, param=False):
        super().__init__(d, t, f, mu, sigma, h, pc, data_gen, N, refb, hitb, param)
        if param:
            self.p = utils.Parameter(1)
        else:
            self.p = utils.FNN(self.d,1)
        self.grad_p = nn.ModuleList([utils.FNN(self.d,self.d) for _ in range(self.N)])
    
    def forward(self, batch):
        xt = self.data_gen(batch)
        dBt = torch.randn([self.N,batch,self.d],device=xt.device)*torch.sqrt(self.dt)
        p_pre = self.p(xt)
        run = torch.ones(batch,device=xt.device).bool()
        p_rel = torch.zeros([batch,1],device=xt.device)
        for i in range(self.N):
            grad_p = self.grad_p[i](xt)
            f = self.f(i*self.dt,xt,p_pre,grad_p)
            mu = self.mu(i*self.dt,xt)
            sigma = self.sigma(i*self.dt,xt)
            xt, xb, dRt, ref, hit = self.const_sde(xt,mu,sigma,dBt[i],run)
            h = self.h(i*self.dt,xb)
            p_pre = self.const_bsde(p_pre,grad_p,f,h,sigma,dBt[i],dRt,ref,run)
            p_rel[run*hit] = self.pc(i,xt)[run*hit]
            run = run * (~hit)
        p_rel[run] = self.pc(self.N,xt)[run]
        return p_pre, p_rel
    
    def const_sde(self, xt, mu, sigma, dBt, run):
        dxt = mu*self.dt + torch.bmm(sigma,dBt.unsqueeze(-1)).squeeze(-1)
        xt = xt + dxt*run.unsqueeze(1)
        xb, nb, ref = self.refb(xt)
        dRt = 2*((xb-xt)*(-nb)).sum(dim=1,keepdim=True)
        xt[ref] = xt[ref] + dRt[ref]*(-nb[ref])
        hit = self.hitb(xt)
        return xt, xb, dRt, ref, hit
    
    def const_bsde(self, p, grad_p, f, h, sigma, dBt, dRt, ref, run):
        grad_p = torch.bmm(grad_p.unsqueeze(1),sigma).squeeze(1)
        dp = - f*self.dt + (grad_p*dBt).sum(dim=1,keepdim=True) - h*dRt*ref[:,None]
        p = p + dp*run[:,None]
        return p


class SphereIBSDE(CIBSDE):
    def __init__(self, dx, t, f, D, h, pc, data_gen, N, refb, hitb, param=False):
        super().__init__(3*dx, t, f, None, None, h, pc, data_gen, N, refb, hitb, param)
        self.dx = dx
        self.D = D
        self.fix_x = nn.Parameter(torch.tensor([1.,0,0]),requires_grad=False)
        self.grad_eucl_polar = nn.Parameter(torch.tensor([[0.,0.],[0.,1.],[-1.,0.]]),requires_grad=False)
        if param:
            self.p = utils.Parameter(1)
        else:
            self.p = utils.FNN(self.d,1)
        self.grad_p = nn.ModuleList([utils.FNN(self.d,self.d) for _ in range(self.N)])
    
    def forward(self, batch):
        xt = self.data_gen(batch)
        dBt = torch.randn([self.N,batch,self.dx,2],device=xt.device) * torch.sqrt(self.dt)
        p_pre = self.p(xt)
        run = torch.ones(batch,device=xt.device).bool()
        p_rel = torch.zeros([batch,1],device=xt.device)
        for i in range(self.N):
            grad_p = self.grad_p[i](xt)
            f = self.f(i*self.dt,xt,p_pre)
            xt, xb, dRt, ref, hit, T_inv = self.const_sde(xt,dBt[i],run)
            h = self.h(i*self.dt,xb)
            p_pre = self.const_bsde(p_pre,grad_p,f,h,T_inv,dBt[i],dRt,ref,run)
            p_rel[run*hit] = self.pc(i,xt)[run*hit]
            run = run * (~hit)
        p_rel[run] = self.pc(self.N,xt)[run]
        return p_pre, p_rel

    def const_sde(self, xt, dBt, run):
        B = xt.shape[0]
        xt = xt.reshape([B*self.dx,3])
        T_inv = self.T_inv(xt)
        dxt = torch.sqrt(2*self.D).unsqueeze(1) * dBt
        dxt[:,:,0] += torch.pi/2
        dxt = utils.euclid_corr(dxt.reshape([B*self.dx,2])) - self.fix_x
        dxt = torch.bmm(T_inv,dxt.unsqueeze(-1)).squeeze(-1)
        xt = xt + dxt*run.unsqueeze(1).expand([B,self.dx]).reshape([B*self.dx,1])
        xb, nb, ref = self.refb(xt)
        dRt = 2*((xb-xt)*(-nb)).sum(dim=1,keepdim=True)
        xt[ref] = xt[ref] + dRt[ref]*(-nb[ref])
        hit = self.hitb(xt)
        return xt.reshape([B,self.d]), xb.reshape([B,self.d]), dRt, ref, hit, T_inv
    
    def const_bsde(self, p, grad_p, f, h, T_inv, dBt, dRt, ref, run):
        B = p.shape[0]
        grad_p = torch.bmm(grad_p.reshape([B*self.dx,1,3]),T_inv)
        grad_p = torch.bmm(grad_p,self.grad_eucl_polar.expand([B*self.dx,3,2])).reshape([B,self.dx*2])
        dp = - f*self.dt + (grad_p*(torch.sqrt(2*self.D).unsqueeze(1)*dBt).reshape([B,self.dx*2])).sum(dim=1,keepdim=True) - h*(dRt*ref[:,None]).view(B,-1).sum(dim=-1,keepdim=True)
        p = p + dp*run.unsqueeze(1)
        return p


class ConstaintIBSDE(CIBSDE):
    def __init__(self, dx, dy, t, f, mu, sigma, Dx, h, pc, data_gen, N, refb, hitb, params=False):
        super().__init__(3*dx+dy, t, f, mu, sigma, h, pc, data_gen, N, refb, hitb)
        self.dx = dx
        self.dy = dy
        self.Dx = Dx
        self.params = params
        self.fix_x = nn.Parameter(torch.tensor([1.,0,0]),requires_grad=False)
        self.grad_eucl_polar = nn.Parameter(torch.tensor([[0.,0.],[0.,1.],[-1.,0.]]),requires_grad=False)
        if params:
            self.p = utils.Parameter(1)
        else:
            self.p = utils.FNN(self.d,1)
        self.grad_p = nn.ModuleList([utils.FNN(self.d,self.d) for _ in range(self.N)])
    
    def forward(self, batch):
        xt, yt = self.data_gen(batch)
        dBxt = torch.randn([self.N,batch,self.dx,2],device=xt.device) * torch.sqrt(self.dt)
        dByt = torch.randn([self.N,batch,self.dy],device=xt.device) * torch.sqrt(self.dt)
        p_pre = self.p(torch.cat([xt,yt],dim=1))
        run = torch.ones(batch,device=xt.device).bool()
        p_rel = torch.zeros([batch,1],device=xt.device)
        for i in range(self.N):
            grad_p = self.grad_p[i](torch.cat([xt,yt],dim=1))
            f = self.f(i*self.dt,xt,yt,p_pre,grad_p[:,self.dx*3:])
            mu = self.mu(i*self.dt,yt)
            sigma = self.sigma(i*self.dt,yt)
            xt, yt, xb, yb, dRxt, dRyt, refx, refy, hit, T_inv = self.const_sde(xt,yt,mu,sigma,dBxt[i],dByt[i],run)
            h = self.h(i*self.dt,xb,yb)
            p_pre = self.const_bsde(p_pre,grad_p,f,h,sigma,T_inv,dBxt[i],dByt[i],dRxt,dRyt,refx,refy,run)
            p_rel[run*hit] = self.pc(i,xt,yt)[run*hit]
            run = run * (~hit)
        p_rel[run] = self.pc(self.N,xt,yt)[run]
        return p_pre, p_rel

    def const_sde(self, xt, yt, mu, sigma, dBxt, dByt, run):
        B = xt.shape[0]
        xt = xt.reshape([B*self.dx,3])
        T_inv = self.T_inv(xt)
        dxt = torch.sqrt(2*self.Dx).unsqueeze(1) * dBxt
        dxt[:,:,0] += torch.pi/2
        dxt = utils.euclid_corr(dxt.reshape([B*self.dx,2])) - self.fix_x
        dxt = torch.bmm(T_inv,dxt.unsqueeze(-1)).squeeze(-1)
        xt = xt + dxt*run.unsqueeze(1).expand([B,self.dx]).reshape([B*self.dx,1])
        dyt = mu*self.dt + torch.bmm(sigma,dByt.unsqueeze(-1)).squeeze(-1)
        yt = yt + dyt*run.unsqueeze(1)
        xb, nxb, refx, yb, nyb, refy = self.refb(xt,yt)
        dRxt = 2*((xb-xt)*(-nxb)).sum(dim=1,keepdim=True)
        dRyt = 2*((yb-yt)*(-nyb)).sum(dim=1,keepdim=True)
        xt[refx] = xt[refx] + (-nxb[refx])*dRxt[refx]
        yt[refy] = yt[refy] + (-nyb[refy])*dRyt[refy]
        hit = self.hitb(xt,yt)
        return xt.reshape([B,self.dx*3]), yt, xb.reshape([B,self.dx*3]), yb, dRxt, dRyt, refx, refy, hit, T_inv
    
    def const_bsde(self, p, grad_p, f, h, sigma, T_inv, dBxt, dByt, dRxt, dRyt, refx, refy, run):
        B = p.shape[0]
        gradx_p = grad_p[:,:self.dx*3]
        grady_p = grad_p[:,self.dx*3:]
        gradx_p = torch.bmm(gradx_p.reshape([B*self.dx,1,3]),T_inv)
        gradx_p = torch.bmm(gradx_p,self.grad_eucl_polar.expand([B*self.dx,3,2])).reshape([B,self.dx*2])
        grady_p = torch.bmm(grady_p.unsqueeze(1),sigma).squeeze(1)
        dp = - f*self.dt + (gradx_p*(torch.sqrt(2*self.Dx).unsqueeze(1)*dBxt).reshape([B,self.dx*2])).sum(dim=1,keepdim=True) + (grady_p*dByt).sum(dim=1,keepdim=True) - h*((dRxt*refx[:,None]).view(B,-1).sum(dim=-1,keepdim=True) + dRyt*refy[:,None])
        p = p + dp*run.unsqueeze(1)
        return p


def train(model, params:dict):
    epoch = params['epoch']
    batch = params['batch']
    lr = params['lr']

    optim = torch.optim.Adam(model.parameters(),lr=lr)
    loss_fun = nn.MSELoss()
    
    loss_values = torch.zeros(epoch)
    pbar = tqdm(range(epoch))
    for i in pbar:
        model.train()
        optim.zero_grad()
        u_pre, u_rel = model(batch)
        loss = loss_fun(u_pre,u_rel)
        loss.backward()
        optim.step()

        model.eval()
        loss_values[i] = loss.item()
        pbar.set_postfix(loss='%e' % loss_values[i])
    print("\nTraining has been completed.")
    return loss_values