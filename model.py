
import torch
import torch.nn.functional as F
import torch.nn as nn

class CombineNet(nn.Module):
    def __init__(self, n,  nfeats,n_clusters, args, device):
        super(CombineNet, self).__init__()
        self.n = n
        self.n_clusters = n_clusters
        self.blocks = args.block
        self.nfeats = nfeats
        self.delta = torch.FloatTensor([args.delta]).to(device)
        self.ZZ_init = nn.Linear(nfeats, n_clusters)
        self.theta1 = nn.Parameter(torch.FloatTensor([args.thre1]), requires_grad=True).to(device)
        self.theta2 = nn.Parameter(torch.FloatTensor([args.thre2]), requires_grad=True).to(device)
        self.bn_input_01 = nn.BatchNorm1d(n_clusters, momentum=0.6).to(device)
        self.bn_input_02 = nn.BatchNorm1d(nfeats, momentum=0.6).to(device)
        self.U = nn.Linear(n_clusters, n_clusters).to(device)
        self.device = device
    def  self_active_l1(self, u, theta):
        return F.selu(u - theta) - F.selu(-1.0 * u - theta)
    def self_active_l21(self, x, theta):
        nw = torch.norm(x)
        if nw > theta:
            x = (nw - 1 / theta) * x / nw
        else:
            x = torch.zeros_like(x)
        return x
    def forward(self, features, lap, active_O):
        Z = list()
        Z.append(self.ZZ_init(features / 1.0))
        O=list()
        E=list()
        # O.append(torch.zeros(self.n,self.nfeats).to(self.device))
        O.append(torch.rand_like(features).to(self.device)*0.05)

        for i in range(self.blocks):

            input1 = torch.pinverse(torch.mm(Z[-1].t(), Z[-1]) + 0.0001)
            E.append(torch.mm(input1,torch.mm( Z[-1].t(), features-O[-1])))

            input2 = self.U(Z[-1])
            L1 = torch.norm(E[-1].t().matmul(E[-1]))
            input3 = torch.mm( lap,Z[-1]) /L1
            input1 = torch.mm(features-O[-1],E[-1].t()) /L1
            z=input2-self.delta*input3+ input1

            Z.append(self.self_active_l1(self.bn_input_01(z), self.theta1))
            if active_O=='l1':
                O.append(self.self_active_l1( self.bn_input_02(features- torch.mm(Z[-1], E[-1])), self.theta2))
            else:
                O.append(self.self_active_l21( self.bn_input_02(features - torch.mm(Z[-1], E[-1])), self.theta2))
        return Z[-1]


class CombineNet_abO(nn.Module):
    def __init__(self, n,  nfeats,n_clusters, args, device):
        super(CombineNet_abO, self).__init__()
        self.n = n
        self.n_clusters = n_clusters
        self.blocks = args.block
        self.nfeats = nfeats
        self.delta = torch.FloatTensor([args.delta]).to(device)
        self.ZZ_init = nn.Linear(nfeats, n_clusters)
        self.theta1 = nn.Parameter(torch.FloatTensor([args.thre1]), requires_grad=True).to(device)
        self.theta2 = nn.Parameter(torch.FloatTensor([args.thre2]), requires_grad=True).to(device)
        self.bn_input_01 = nn.BatchNorm1d(n_clusters, momentum=0.6).to(device)
        self.U = nn.Linear(n_clusters, n_clusters).to(device)
        self.device = device
    def  self_active_l1(self, u, theta):
        return F.selu(u - theta) - F.selu(-1.0 * u - theta)
    def self_active_l21(self, x, theta):
        nw = torch.norm(x)
        if nw > theta:
            x = (nw - 1 / theta) * x / nw
        else:
            x = torch.zeros_like(x)
        return x
    def forward(self, features, lap, active_O):
        Z = list()
        Z.append(self.ZZ_init(features / 1.0))
        E=list()

        for i in range(self.blocks):

            input1 = torch.pinverse(torch.mm(Z[-1].t(), Z[-1]) +  0.0001)
            E.append(torch.mm(input1,torch.mm( Z[-1].t(), features)))

            input2 = self.U(Z[-1])
            L1 = torch.norm(E[-1].t().matmul(E[-1]))
            input1 = torch.mm(features,E[-1].t()) /L1
            z=input2+ input1

            Z.append(self.self_active_l1(self.bn_input_01(z), self.theta1))

        return Z[-1]

