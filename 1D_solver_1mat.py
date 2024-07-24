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


class DNN(torch.nn.Module):     # DNN layer 짜기
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


class PINN():
    def __init__(self, X, layers):

        self.x = X
        self.dnn1 = DNN(layers).to(device)  # PINN에서 DNN 불러옴
        self.optimizer_Adam1 = torch.optim.Adam(self.dnn1.parameters(), lr=0.0005)

    def net_f(self, dnn, x, t, rho, k, cp, cc_x, cc_xx, cc_t,):
        T = dnn(torch.cat([x, t], dim=1))

        T_x = torch.autograd.grad(
            T, x,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]

        T_xx = torch.autograd.grad(
            T_x, x,
            grad_outputs=torch.ones_like(T_x),
            retain_graph=True,
            create_graph=True
        )[0]

        T_t = torch.autograd.grad(
            T, t,
            grad_outputs=torch.ones_like(T),
            retain_graph=True,
            create_graph=True
        )[0]

        f = k * T_xx * cc_xx - rho*cp*(T_t*cc_t)
        flux_x = - k * T_x * cc_x
        return T, f, flux_x


    def train(self, nIter_adam):

        def loss_dnn1(T_pred1, f_pred1, flux_x_pred1):

            ## heat flux bc
            id1 = domain1_point_all[cond_domain1_heat_bc,1] <=13.5
            id2 = domain1_point_all[cond_domain1_heat_bc,1] > 13.5
            q1 = domain1_point_all[cond_domain1_heat_bc[id1],1] * 10/13.5
            q2 = domain1_point_all[cond_domain1_heat_bc[id2],1] * -10/13.5 + 20
            heat_bc = ( torch.cat([ q1[:,None] - flux_x_pred1[cond_domain1_heat_bc[id1]],
                                    q2[:,None] - flux_x_pred1[cond_domain1_heat_bc[id2]]], dim=0)  ) ** 2

            ## insulated bc
            insul = flux_x_pred1[cond_domain1_insul_bc]**2

            domain1_Loss_NBC = torch.mean( torch.cat([heat_bc, insul], dim=0) )

            ## T0, intial condition
            T0_normal = (T0 - min_output) / (max_output - min_output)
            domain1_Loss_IC = torch.mean((T0_normal - T_pred1[cond_domain1_T0_IC]) ** 2)

            ## PDE
            domain1_Loss_pde = torch.mean(f_pred1 ** 2)

            domain1_Loss_all_noweight = domain1_Loss_pde + domain1_Loss_NBC + domain1_Loss_IC

            return domain1_Loss_all_noweight, domain1_Loss_pde, domain1_Loss_NBC, domain1_Loss_IC


        # min-max 정규화를 이렇게 직접 해줬음
        min_input_x = np.min(self.x[:,0])
        max_input_x = np.max(self.x[:,0])

        min_input_t = np.min(self.x[:,1])
        max_input_t = np.max(self.x[:,1])

        min_output = 40
        max_output = 100

        # 정규화 과정에서 붙음 coefficient 같은거
        cc_xx = (max_output-min_output) / (max_input_x-min_input_x)**2
        cc_x = (max_output-min_output) / (max_input_x-min_input_x)
        cc_t = (max_output-min_output) / (max_input_t-min_input_t)

        input_x_normalized = (self.x[:,0].copy() - min_input_x) / (max_input_x - min_input_x)
        input_t_normalized = (self.x[:,1].copy() - min_input_t) / (max_input_t - min_input_t)

        # 각 loss의 weight는 1로 설정
        pde_weight = 1
        NBC_weight =1
        IC_weight = 1


        loss_sum_all_noweight = []
        Relative_L2_err_all = []

        best_loss = 1e22
        best_epoch = 0





        '''        point 구분        '''
        domain1_point_x = torch.tensor(self.x[:,0].reshape(-1,1)).float().to(device)
        domain1_point_t = torch.tensor(self.x[:,1].reshape(-1,1)).float().to(device)

        domain1_point_all = torch.cat([domain1_point_x, domain1_point_t], dim=1)

        domain1_point_x_normal = torch.tensor(input_x_normalized.reshape(-1,1), requires_grad=True).float().to(device)
        domain1_point_t_normal = torch.tensor(input_t_normalized.reshape(-1,1), requires_grad=True).float().to(device)


        '''        BC, IC index        '''
        cond_domain1_heat_bc = ( (domain1_point_all[:,0]==0) ).nonzero(as_tuple=True)[0]
        cond_domain1_insul_bc = ( (domain1_point_all[:,0]==2.5)  ).nonzero(as_tuple=True)[0]
        cond_domain1_T0_IC = ( (domain1_point_all[:,1]==0)  ).nonzero(as_tuple=True)[0]



        start = time.time()
        best_start = time.time()
        best_time = 0

        domain1_Save_result = pd.DataFrame()


        self.dnn1.train()
        for epoch in range(1,nIter_adam+2):     # 여기서부터 epoch에 따라 train 진행
            self.optimizer_Adam1.zero_grad(set_to_none=False)

            # 여기를 돌면서 T, flux 값들을 계산함
            T_pred1, f_pred1, flux_x_pred1 = self.net_f(self.dnn1, domain1_point_x_normal, domain1_point_t_normal, rho, k, cp, cc_x, cc_xx, cc_t)

            domain1_Loss_all_noweight, domain1_Loss_pde, domain1_Loss_NBC, domain1_Loss_IC = loss_dnn1(T_pred1, f_pred1, flux_x_pred1)

            # Relative L2 error
            T_pred1_inv_normal = T_pred1.detach() * (max_output-min_output) + min_output
            Relative_L2_err = torch.norm(T_pred1_inv_normal.flatten()-temp1_vec) / torch.norm(temp1_vec)
            Relative_L2_err_all = np.hstack((Relative_L2_err_all,Relative_L2_err.item()))



            Loss_all = domain1_Loss_pde*pde_weight + domain1_Loss_NBC*NBC_weight + domain1_Loss_IC*IC_weight

            loss_sum_all_noweight = np.hstack((loss_sum_all_noweight,domain1_Loss_all_noweight.item()))


            if domain1_Loss_all_noweight.item() < best_loss:
                best_loss = domain1_Loss_all_noweight.item()
                best_Relative_L2_err = Relative_L2_err.item()
                best_epoch = epoch
                best_time = time.time() - best_start

                best_loss_PDE = domain1_Loss_pde.item()
                best_loss_NBC = domain1_Loss_NBC.item()
                best_loss_IC = domain1_Loss_IC.item()

                PATH = drive_path + 'model.pth'
                torch.save(self.dnn1.state_dict(), PATH)


            tr_time = time.time() - start

            ## save csv -- domain1
            domain1_df = pd.DataFrame({'Training time' : tr_time,
                                    'Current_Epoch' : epoch,
                                    'Total Domain Loss': domain1_Loss_all_noweight.item(),
                                    'Total Loss' : loss_sum_all_noweight[-1],
                                    'Relative_L2_err' : Relative_L2_err_all[-1],
                                    'Best time' : best_time,
                                    'Best Epoch' : best_epoch,
                                    'Best Total Loss' : best_loss,
                                    'Best Relative_L2_err' : best_Relative_L2_err,
                                    'Domain PDE Loss' : domain1_Loss_pde.item(),
                                    'Domain NBC Loss' : domain1_Loss_NBC.item(),
                                    'Domain IC Loss' : domain1_Loss_IC.item(),
                                    }, index = [0])


            domain1_Save_result = domain1_Save_result.append(domain1_df, ignore_index=True)

            if (epoch) % 100 == 0:
                print('')
                print('Training time: %d s' %(tr_time))
                print('Best time: %d s' %(best_time))
                print('Total Loss: %.3e' %(loss_sum_all_noweight[-1]))

                print(
                    'Model 1: Epoch: %d, Total Loss: %.3e, PDE Loss: %.3e, NBC Loss: %.3e, IC Loss: %.3e, Rel_L2_err: %.3e' %
                    (
                        epoch,
                        domain1_Loss_all_noweight.item(),
                        domain1_Loss_pde.item(),
                        domain1_Loss_NBC.item(),
                        domain1_Loss_IC.item(),
                        Relative_L2_err.item(),
                    )
                )
                print('Best: Epoch: %d, Total_Loss: %.3e, PDE Loss: %.3e, NBC Loss: %.3e, IC Loss: %.3e, Rel_L2_err: %.3e' %
                    (
                        best_epoch,
                        best_loss,
                        best_loss_PDE,
                        best_loss_NBC,
                        best_loss_IC,
                        best_Relative_L2_err,
                    )
                )

                plt.clf()
                plt.semilogy(range(1,epoch+1), loss_sum_all_noweight, '-k', label='Total')
                plt.legend(prop={'size':12}, loc='upper left')
                plt.xlabel("Epochs",fontsize=18)
                plt.ylabel("Total Loss w/o weights",fontsize=18)
                plt.xlim(right=epoch*1.2)
                plt.savefig(drive_path + 'fig_loss.png', dpi=100 ,bbox_inches='tight')
                
                plt.clf()
                plt.semilogy(range(1,epoch+1), Relative_L2_err_all, '-k', label='Relative L2 error')
                plt.legend(prop={'size':12}, loc='upper left')
                plt.xlabel("Epochs",fontsize=18)
                plt.ylabel("Relative L2 error",fontsize=18)
                plt.xlim(right=epoch*1.2)
                plt.savefig(drive_path + 'fig_Rel_L2_err.png', dpi=100 ,bbox_inches='tight')

                domain1_Save_result.to_csv(drive_path + 'Result_domain1.csv', index=None)


            if ((epoch) % 100000 == 0):
                domain1_Save_result = pd.DataFrame()
                shutil.copy(drive_path + 'Result_domain1.csv', drive_path + 'Result_domain1_epch' + str(epoch) + '.csv')


            if epoch == nIter_adam+1:
                break
            else:
                Loss_all.backward()

                self.optimizer_Adam1.step()



drive_path = 'vanilla/'
# drive_path = 'drive/MyDrive/test/1D_solver_1mat/vanilla/'



# material properties
rho = 7.860
k = 0.162
cp = 0.5
T0 = 40



layers = [2, 20, 20, 20, 20, 20, 20, 1]     # x,y

domain1_x = np.linspace(0, 2.5, 21)
domain1_t = np.arange(0, 27+0.1, 0.1)
x, t = np.meshgrid(domain1_x, domain1_t)
domain1_grid = np.hstack((x.flatten()[:,None], t.flatten()[:,None]))

point_all = domain1_grid
indexes  = np.unique(point_all, axis=0, return_index=True)[1]
point_all = point_all[np.sort(indexes),:]
print(point_all.shape)



# ground truth
path0 = 'data/'
# path0 = 'drive/MyDrive/test/1D_solver_1mat/data/'
temp1 = pd.read_csv(path0 + 'Run1_13.5_10.0_temp.csv', header=None)
temp1 = temp1.values[:,::10]
temp1_vec = torch.tensor(temp1.flatten()).float().to(device)



model = PINN(point_all, layers)
model.train(200000)