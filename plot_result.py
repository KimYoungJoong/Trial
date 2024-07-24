import sys, os
import torch
from collections import OrderedDict
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pandas as pd
from torch import Tensor
import shutil
os.chdir(os.path.dirname(__file__))
# from google.colab import drive
# drive.mount('drive')


# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

warnings.filterwarnings('ignore')

np.random.seed(1234)
torch.manual_seed(8)


class DNN(torch.nn.Module):
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1

        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))
            layer_list.append(('activation_%d' % i, torch.nn.Tanh()))

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out

''' Prediction '''


# material properties
rho = 7.860
k = 0.162
cp = 0.5
T0 = 40

layers = [2, 20, 20, 20, 20, 20, 20, 1]     # x,y


# load model
model1_vanilla = DNN(layers).to(device)
model1_vanilla.load_state_dict(torch.load('vanilla/model.pth'))
# model1_vanilla.load_state_dict(torch.load('drive/MyDrive/test/1D_solver_1mat/vanilla/model.pth'))
model1_vanilla.eval()



# point for prediction
nn1 = 201
nn2 = 271

min_input_x = 0
max_input_x = 2.5

min_input_t = 0
max_input_t = 27

min_output = 40
max_output = 100

cc_xx = (max_output-min_output) / (max_input_x-min_input_x)**2
cc_x = (max_output-min_output) / (max_input_x-min_input_x)
cc_t = (max_output-min_output) / (max_input_t-min_input_t)

domain1_x = np.linspace(0, 2.5, nn1)
domain1_t = np.linspace(0, 27, nn2)
x, t = np.meshgrid(domain1_x, domain1_t)
domain1_grid = np.hstack((x.flatten()[:,None], t.flatten()[:,None]))
domain_x_all = domain1_grid[:, 0:1]
domain_t_all = domain1_grid[:, 1:2]

domain1_point_x = torch.tensor(domain_x_all[:,0].reshape(-1,1), requires_grad=True).float().to(device)
domain1_point_t = torch.tensor(domain_t_all[:,0].reshape(-1,1), requires_grad=True).float().to(device)
domain1_point_x_normal = (domain1_point_x-min_input_x) / (max_input_x-min_input_x)
domain1_point_t_normal = (domain1_point_t-min_input_t) / (max_input_t-min_input_t)
domain1_point_all_normal = torch.cat([domain1_point_x_normal, domain1_point_t_normal], dim=1)



# prediction
T_pred1_vanilla = model1_vanilla(domain1_point_all_normal)
T_pred1_vanilla = T_pred1_vanilla.detach().cpu().numpy()
T_pred1_vanilla = T_pred1_vanilla * (max_output-min_output) + min_output
T_pred_all_vanilla = np.reshape(T_pred1_vanilla, (nn2,nn1))

domain_x_all = np.reshape(domain_x_all, (nn2,nn1))
domain_t_all = np.reshape(domain_t_all, (nn2,nn1))



# load ground turth
path0 = 'data/'
# path0 = 'drive/MyDrive/test/1D_solver_1mat/data/'

t = pd.read_csv(path0 + '1D_Bar_time.csv', header=None)
t = t.values.tolist()
t = np.array(t)
x = pd.read_csv(path0 + '1D_Bar_x.csv', header=None)
x = x.values.tolist()
x = np.array(x)
temp1 = pd.read_csv(path0 + 'Run1_13.5_10.0_temp.csv', header=None)
temp1 = temp1.values.tolist()
temp1 = np.array(temp1)



min_T = np.min([T_pred_all_vanilla, temp1])
max_T = np.max([T_pred_all_vanilla, temp1])

min_err = np.min(np.abs(T_pred_all_vanilla-temp1))
max_err = np.max(np.abs(T_pred_all_vanilla-temp1))




plt.rcParams['font.size'] = 14

plt.figure(figsize=(15, 10))

# ground truth
plt.subplot(2, 2, 1)
plt.contourf(t,x , temp1, list(np.linspace(min_T,max_T,28)),cmap='jet')
cbar=plt.colorbar()
cbar.set_label(r"Temperature $[℃]$", labelpad=10, rotation=90, size=14)
plt.xlabel(r"Time $[s]$")
plt.ylabel(r"x $[cm]$")
plt.title("Ground truth", size=14)


# vanilla PINN
plt.subplot(2, 2, 3)
plt.contourf(domain_t_all,domain_x_all , T_pred_all_vanilla, list(np.linspace(min_T,max_T,28)),cmap='jet')
cbar=plt.colorbar()
cbar.set_label(r"Temperature $[℃]$", labelpad=10, rotation=90, size=14)
plt.xlabel(r"Time $[s]$")
plt.ylabel(r"x $[cm]$")
plt.title('Prediction (vanilla PINN)', size=14)

plt.subplot(2, 2, 4)
plt.contourf(domain_t_all, domain_x_all, np.abs(T_pred_all_vanilla-temp1),list(np.linspace(min_err,max_err,28)),cmap='jet')
cbar=plt.colorbar()
cbar.set_label(r"Point-wise error $[℃]$", labelpad=10, rotation=90, size=14)
plt.xlabel(r"Time $[s]$")
plt.ylabel(r"x $[cm]$")
plt.title('Point-wise error (vanilla PINN)', size=14)



plt.tight_layout()




''' training history '''

# vanilla PINN
epoch_vanilla = []
Loss_vanilla = []
Relative_L2_vanilla = []
for epoch in range(100000,200000,100000):
    result = pd.read_csv(r"vanilla/Result_domain1_epch" + str(epoch) + ".csv", header=0)
    # result = pd.read_csv(r"drive/MyDrive/test/1D_solver_1mat/vanilla/Result_domain1_epch" + str(epoch) + ".csv", header=0)
    epoch_vanilla = epoch_vanilla + result.values[:,1].tolist()
    Loss_vanilla = Loss_vanilla + result.values[:,3].tolist()
    Relative_L2_vanilla = Relative_L2_vanilla + result.values[:,4].tolist()
result = pd.read_csv(r"vanilla/Result_domain1.csv", header=0)
# result = pd.read_csv(r"drive/MyDrive/test/1D_solver_1mat/vanilla/Result_domain1.csv", header=0)
epoch_vanilla = epoch_vanilla + result.values[:,1].tolist()
Loss_vanilla = Loss_vanilla + result.values[:,3].tolist()
Relative_L2_vanilla = Relative_L2_vanilla + result.values[:,4].tolist()

epoch_vanilla = np.array(epoch_vanilla)
Loss_vanilla = np.array(Loss_vanilla)
Relative_L2_vanilla = np.array(np.array(Relative_L2_vanilla))



plt.rcParams['font.size'] = 14


# Relative_L2
plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.semilogy(epoch_vanilla, Relative_L2_vanilla, '-k', linewidth=1.5, label='vanilla PINN')
plt.xlabel(r"Epoch")
plt.ylabel(r"Relative_L2_error")
plt.title('Relative_L2_error history', size=14)
plt.xlim(left=0, right=epoch_vanilla[-1])
plt.ylim(top=1)
plt.grid()
plt.legend(prop={'size':14}, loc='upper right')


# loss
plt.figure(figsize=(9, 6))
ax = plt.subplot(111)
plt.semilogy(epoch_vanilla, Loss_vanilla, '-k', linewidth=1.5, label='vanilla PINN')
plt.xlabel(r"Epoch")
plt.ylabel(r"Total Loss w/o weights")
plt.title('Loss history', size=14)
plt.xlim(left=0, right=epoch_vanilla[-1])
plt.ylim(top=2e1)
plt.grid()
plt.legend(prop={'size':14}, loc='upper right')

plt.show()