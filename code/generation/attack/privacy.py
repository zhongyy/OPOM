import torch
import torch.nn as nn
import sklearn
import sklearn.preprocessing
import numpy as np
import cvxpy as cp

def cos_sim(fea1,fea2):
    assert fea1.shape[0] == fea2.shape[0]
    fea1 = sklearn.preprocessing.normalize(fea1)
    fea2 = sklearn.preprocessing.normalize(fea2)
    similarity = []
    for i in range(fea1.shape[0]):
        similarity.append(np.sqrt(np.sum((fea1[i]-fea2[i])*(fea1[i]-fea2[i]))))
    return similarity

class inverse_mse(nn.Module):
    def __init__(self):
        super(inverse_mse, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fea1, fea2):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim = 1).view(fea1.shape[0],1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim = 1).view(fea2.shape[0],1)
        dis = - self.mse(nfea1, nfea2)
        return dis

class eachother_dot(nn.Module):
    def __init__(self):
        super(eachother_dot, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fea1, fea2):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
        dis = torch.mean(torch.mm(nfea1, torch.transpose(nfea2, 0, 1)))
        return dis

class affine_hull_cvx(nn.Module):
    def __init__(self):
        super(affine_hull_cvx, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fea1, fea2):
        nfea1 = torch.transpose(fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1),0,1)
        nfea2 = torch.transpose(fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1),0,1)
        # nfea2 --> A, nfea1 --> y, caculate x.
        # Using cvx to calculate variable x

        A = nfea2.detach().cpu().numpy()
        XX = torch.tensor(np.zeros((nfea1.shape[1],nfea1.shape[1])), dtype=torch.float32, device=torch.device("cuda:0"))
        for i in range(nfea1.shape[1]):
            y = nfea1[:,i].detach().cpu().numpy()

            x = cp.Variable(nfea1.shape[1])
            objective = cp.Minimize(cp.sum_squares(A @ x - y))
            constraints = [sum(x)==1]
            prob = cp.Problem(objective, constraints)
            print(i, "loss", prob.solve(), sum(x.value))
            print(i, "x:", x.value)
            x_tensor = torch.tensor(x.value, dtype=torch.float32, device=torch.device("cuda:0"))
            XX[:,i]= x_tensor
        #embed()
        DIS = - self.mse(torch.mm(nfea2, XX.detach()), nfea1)
        return DIS

class convex_hull_cvx_dyn(nn.Module):
    def __init__(self):
        super(convex_hull_cvx_dyn, self).__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, fea1, fea2, lower = 0.0, upper = 1.0):
        nfea1 = fea1 / torch.linalg.norm(fea1, dim=1).view(fea1.shape[0], 1)
        nfea2 = fea2 / torch.linalg.norm(fea2, dim=1).view(fea2.shape[0], 1)
        # nfea2 --> A, nfea1 --> y, caculate x.
        # Using cvx to calculate variable x
        lowerbound = lower
        upperbound = upper
        A = nfea2.detach().cpu().numpy()
        XX = torch.tensor(np.zeros((nfea1.shape[0],nfea1.shape[0])), dtype=torch.float32, device=torch.device("cuda:0"))
        for i in range(nfea1.shape[0]):
            y = nfea1[i].detach().cpu().numpy()

            x = cp.Variable(nfea1.shape[0])
            #embed()
            objective = cp.Minimize(cp.sum_squares(x @ A - y))
            constraints = [sum(x)==1, lowerbound <= x, x <= upperbound]
            prob = cp.Problem(objective, constraints)
            print(i, "loss", prob.solve(), sum(x.value))
            #embed()
            print(i, "x:", x.value)
            x_tensor = torch.tensor(x.value, dtype=torch.float32, device=torch.device("cuda:0"))
            XX[i]= x_tensor
        #embed()
        DIS = - self.mse(torch.mm(XX.detach().to(fea1.device), nfea2), nfea1)
        #embed()
        return DIS

class FIM():
    def __init__(self, step = 10, epsilon = 10, alpha = 1, random_start = True, loss_type = 0, nter = 5000, upper = 1.0, lower = 0.0):

        self.step = step
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_start = random_start
        self.loss_type = loss_type
        self.lower = lower
        self.upper = upper
        self.nter = nter
        if loss_type == 0: # FI-UAP
            self.LossFunction = inverse_mse()
        elif loss_type == 2: # FI-UAP+
            self.LossFunction = eachother_dot()
        elif loss_type == 7: # OPOM-ClassCenter
            self.LossFunction = convex_hull_cvx_dyn()
        elif loss_type == 8: # OPOM-AffineHull
            self.LossFunction = affine_hull_cvx()
        elif loss_type == 9: # OPOM-ConvexHull
            self.LossFunction = convex_hull_cvx_dyn()

    def process(self, model, pdata):
        model.eval()
        data = pdata.detach().clone()
        nFeature = model.forward(data)

        if self.random_start:
            torch.manual_seed(0)
            data_adv = data + torch.zeros_like(data).uniform_(-self.epsilon, self.epsilon)
        else:
            data_adv = data
        data_adv = data_adv.detach()

        for i in range(self.step):

            data_adv.requires_grad_()
            advFeature = model.forward(data_adv)
            dis = cos_sim(advFeature.cpu().detach().numpy(), nFeature.cpu().detach().numpy())
            print("step", i, dis)
            #if i == self.step - 1:
            #    print("step", i, dis)
            if self.loss_type == 9:
                if i < self.nter: # init several steps to push adv to the outside of the convexhull
                    Loss = -self.LossFunction(advFeature, nFeature, 1/pdata.shape[0], 1/pdata.shape[0])
                else:
                    Loss = -self.LossFunction(advFeature, nFeature, self.lower, self.upper)
            elif self.loss_type == 7: # center: use 1/pdata.shape[0] as the upper and lower bound
                Loss = -self.LossFunction(advFeature, nFeature, 1 / pdata.shape[0], 1 / pdata.shape[0])
            else:
                Loss = -self.LossFunction(advFeature, nFeature)

            model.zero_grad()
            Loss.backward(retain_graph=True)
            grad_step_mean = torch.mean(data_adv.grad, 0, keepdim=True)
            data_adv = data_adv.detach() + self.alpha * grad_step_mean.sign()

            deta = torch.mean(data_adv - data, 0, keepdim=True)

            eta = torch.clamp(deta, min=-self.epsilon, max=self.epsilon)
            data_adv = torch.clamp(data + eta, 0, 255).detach()

        return eta

class DFANet_MFIM():
    def __init__(self, step = 10, epsilon = 10, alpha = 1, random_start = True, loss_type=3, nter = 5000, upper = 1.0, lower = 0.0):

        self.loss_type = loss_type
        self.step = step
        self.epsilon = epsilon
        self.alpha = alpha
        self.random_start = random_start
        self.lower = lower
        self.upper = upper
        self.nter = nter
        if loss_type == 0: # FI-UAP
            self.LossFunction = inverse_mse()
        elif loss_type == 2: # FI-UAP+
            self.LossFunction = eachother_dot()
        elif loss_type == 7:  # OPOM-ClassCenter
            self.LossFunction = convex_hull_cvx_dyn()
        elif loss_type == 8: # OPOM-AffineHull
            self.LossFunction = affine_hull_cvx()
        elif loss_type == 9:  # OPOM-ConvexHull
            self.LossFunction = convex_hull_cvx_dyn()


    def process(self, model, model_ori, pdata):
        #prepare the models
        model.eval()
        model_ori.eval()
        idx = 0
        for m in model.modules():
            if m.__class__.__name__.startswith('Dropout'):
                if idx == 1:
                    idx += 1
                else:
                    print("drop", m)
                    idx += 1
                    m.train()

        # data prepare
        data = pdata.detach().clone()
        if self.random_start:
            data_adv = data + torch.zeros_like(data).uniform_(-self.epsilon, self.epsilon)
        else:
            data_adv = data
        data_adv = data_adv.detach()

        Grad = 0
        for i in range(self.step):
            nFeature = model.forward(data)
            data_adv.requires_grad_()
            advFeature = model.forward(data_adv)

            dis = cos_sim(advFeature.cpu().detach().numpy(), nFeature.cpu().detach().numpy())

            print("step", i, "dis", dis)

            if self.loss_type == 9:
                if i < self.nter: # init several steps to push adv to the outside of the convexhull
                    Loss = -self.LossFunction(advFeature, nFeature, 1 / pdata.shape[0], 1 / pdata.shape[0])
                else:
                    Loss = -self.LossFunction(advFeature, nFeature, self.lower, self.upper)
            elif self.loss_type == 7:  # center: use 1/pdata.shape[0] as the upper and lower bound
                Loss = -self.LossFunction(advFeature, nFeature, 1 / pdata.shape[0], 1 / pdata.shape[0])
            else:
                Loss = -self.LossFunction(advFeature, nFeature)

            model.zero_grad()
            Loss.backward(retain_graph=True)
            grad_step_mean = torch.abs(data_adv.grad)
            for dim in range(3):
                grad_step_mean = torch.mean(grad_step_mean, dim, keepdim=True)

            Grad += data_adv.grad / grad_step_mean
            data_adv = data_adv.detach() + self.alpha * Grad.sign()

            deta = torch.mean(data_adv - data, 0, keepdim=True)
            eta = torch.clamp(deta, min=-self.epsilon, max=self.epsilon)
            data_adv = torch.clamp(data + eta, 0, 255).detach()

        return eta

