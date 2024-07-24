import sys, os
import torch
from collections import OrderedDict
import time
import numpy as np
import matplotlib.pyplot as plt
import warnings     # Jupyter에서 warnings를 무시하거나 다시 활성화 하려면 이 모듈을 사용
import pandas as pd
from torch import Tensor
import shutil       # 파일 이동, 복사를 쉽게 하기 위한 모듈
os.chdir(os.path.dirname(__file__))
# from google.colab import drive
# drive.mount('drive')

# CUDA support
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

warnings.filterwarnings('ignore')   # 여기서 warnings를 무시함

np.random.seed(1234)
torch.manual_seed(8)


class DNN(torch.nn.Module):     # DNN layer 짜기
    def __init__(self, layers):
        super(DNN, self).__init__()

        self.depth = len(layers) - 1   # layers 수만큼 받아서 마지막에 13개를 뺀다? 이거 - 1이 맞는거 같은데? 그래야 나중에 레이어 수가 맞는데?
        
        layer_list = list()
        for i in range(self.depth - 1):
            layer_list.append(('layer_%d' % i, torch.nn.Linear(layers[i], layers[i+1])))    # 0~1, 1~2, ..., 
            layer_list.append(('activation_%d' % i, torch.nn.Tanh()))       # 여기서 Tanh를 썼음

        layer_list.append(('layer_%d' % (self.depth - 1), torch.nn.Linear(layers[-2], layers[-1])))  # 우리는 20개 뉴론이 6개 레이어로 구성됨
        layerDict = OrderedDict(layer_list)

        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class PINN():       # X와 layers를 초기 인자로 받아옴
    
    def __init__(self, X, layers):  # 첨에 X로 받아오는게 뭘까?왜 t는 안 받아 올까? 저장경로는 아예 안받아옴 <------------X가 아예 (5690,2)= (x,t) 형태로 받아서 넣는거 같음(뒤에 그렇게 보여)

        self.x = X
        self.dnn1 = DNN(layers).to(device)  # PINN에서 DNN 불러옴
        self.optimizer_Adam1 = torch.optim.Adam(self.dnn1.parameters(), lr=0.0005)

    def net_f(self, dnn, x, t, rho, k, cp, cc_x, cc_xx, cc_t,): # net_f 가기 전에 x, t, rho, k, cp, cc_x, cc_xx, cc_t 안받아 오네?
        T = dnn(torch.cat([x, t], dim=1))   # x,t를 불러와서 열벡터로 각각 쌓는다. 그래서 그 결과는 T 하나만 내어놓자!
        # layers = [2, 20, 20, 20, 20, 20, 20, 1]     # 첨에 시간t, 거리 x로 2개 input feature 받아서 마지막에 T 한개의 output feature만 내놓음
        ### 왜 위에서 dnn1을 넣어서 계산하지 않고 dnn으로 불러왔나?

        # 미분항 표현하는 autograd. 단, grad_outputs= 이부분은 가중치를 쓴다고 해설집에 나와있던거 같은데 실제로는 dT/dx할때처럼 미분대상이 써지는 듯(240607)
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

        f = k * T_xx * cc_xx - rho*cp*(T_t*cc_t)    # cc_t, cc_x, cc_xx는 뭘 의미하는 것인가?어쨌든 f는 Loss function의 loss를 의미하게끔 설정됨!
        flux_x = - k * T_x * cc_x                   # flux_x를 정의로써 아예 만들어 놓음
        # cc_x는 min_max normalization에서 나온 변수라고 함!(고명석 박사 설명)

        return T, f, flux_x     # 앞에 input feature들을 받아서 T, dT/dx, d2T/dx2, dT/dt 구한다구음에 loss function, flux_x를 구해냄
                                # return되는 3개의 값이 T_pred1, f_pred1, flux_x_pred1로 받은 다음에 그걸 loss_dnn1에 인수로 넣어서 loss function을 계산함(240607)

    def train(self, nIter_adam):

        def loss_dnn1(T_pred1, f_pred1, flux_x_pred1):

            ## heat flux bc
            # 2.5cm를 0.125cm 단위+ 21개 포인트로 쪼갰고, 0~27초를 0.1초 단위로 쪼개서 + 271개로 
            # domain1_point_all => (5691,2) 0번째 열이 0~2.5까지 갈때, 1번째 열은 0, 0.1, ... 이런식으로 27까지 간다. 즉, x는 0번째 열, 1번째 열은 시간
            # cond_domain1_heat_bc => (271) 0, 21, 42, ... , 5670   # 가장 왼쪽 x지점에서의 값이고, 이값은 heat flux가 들어오는 좌표
            # id1 => (271) true, true, ... , false  # time t가 13.5보다 작거나 같은 지점만 1 그래서 1,1,1,1,.... 0,0,0,0
            # id2 => (271) false, false, ..., true  # time t가 13.5보다 큰 점만 1 그래서 0,0,0,0,..., 1,1,1,1
            # q1 => (136) 0, 0.07, ..., 10
            # q2 => (135) 9.9, 9.85, ..., 0
            id1 = domain1_point_all[cond_domain1_heat_bc,1] <=13.5  # domain1_point_all의 값 중에서 2번째 column에 있는 값 중에서 13.5다 작으면 전부 id1에 넣었음. bool 연산 0,1 로 표현됨
            id2 = domain1_point_all[cond_domain1_heat_bc,1] > 13.5  # domain1_point_all의 값 중에서 2번째 column에서 13.5 이상만 1, 나머지는 0으로 표시
            
            # 시간에 따라 heat_flux를 다르게 주는데, 이때 q(t)로 만들었음
            q1 = domain1_point_all[cond_domain1_heat_bc[id1],1] * 10/13.5   # id1에서 true일때 값만 cond_domain1에서 살려서 표현!
            q2 = domain1_point_all[cond_domain1_heat_bc[id2],1] * -10/13.5 + 20
            print("q1:", q1)
            print("q2", q2)
            print()

            # 이게 heat_bc의 Loss 값인듯? q1은 실제 heat_flux이고, flux_x_pred1은 예측값
            heat_bc = ( torch.cat([ q1[:,None] - flux_x_pred1[cond_domain1_heat_bc[id1]],
                                    q2[:,None] - flux_x_pred1[cond_domain1_heat_bc[id2]]], dim=0)  ) ** 2

            ## insulated bc in predicted values
            insul = flux_x_pred1[cond_domain1_insul_bc]**2  # insulation 되어 있으니까 true value=0이 맞음. 그래서 그냥 predicted value는 제곱하고 있음

            # 2개의 value 값을 1개의 열벡터로 연결해서 붙이고, 그 평균값을 구했음
            domain1_Loss_NBC = torch.mean( torch.cat([heat_bc, insul], dim=0) )

            ## T0, intial condition(240626 여기서부터 다시하기)
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

        # 왜 40, 100으로 이뤄졌을까? 그냥 그 사이에서 가장 작은 숫자가 40이고 가장 큰값이 80정도 되니까 그정도로 정규화 한듯?(240701)
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
        domain1_point_x = torch.tensor(self.x[:,0].reshape(-1,1)).float().to(device)    # x 값만 다시 받아옴
        domain1_point_t = torch.tensor(self.x[:,1].reshape(-1,1)).float().to(device)    # t 값만 다시 받아옴

        # 240622 김영중 수정
        domain1_point_all = torch.cat([domain1_point_x, domain1_point_t], dim=1)    # domain1_point_x와 domain1_point_t를 옆으로 쌓아? 그럼 다시 X 형태의 (5691,2) 인거 같은데?

        # 어쨌든 input 은 normalize 했다는 뜻
        domain1_point_x_normal = torch.tensor(input_x_normalized.reshape(-1,1), requires_grad=True).float().to(device)  # domain 값을 normalized 시행
        domain1_point_t_normal = torch.tensor(input_t_normalized.reshape(-1,1), requires_grad=True).float().to(device)  # domain 값을 normalized 시행


        '''        BC, IC index        '''
        # x= 0, 2.5 인 위치의 index를 각각 저장했음
        cond_domain1_heat_bc = ( (domain1_point_all[:,0]==0) ).nonzero(as_tuple=True)[0]    # 여기 자세히 들여다봐야될듯 어떻게 나누고 있을까?240609
        cond_domain1_insul_bc = ( (domain1_point_all[:,0]==2.5)  ).nonzero(as_tuple=True)[0]
        # t = 0인 IC 조건의 index를 저장했음
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
            T_pred1_inv_normal = T_pred1.detach() * (max_output-min_output) + min_output        # 정규화된 T_pred의 값을 원래 스케일 대로 복원하는 과정
            Relative_L2_err = torch.norm(T_pred1_inv_normal.flatten()-temp1_vec) / torch.norm(temp1_vec)        # torch.norm 은 2-norm을 계산하게 되어있음. 즉 temp1_vec(참값) 대비 얼마나 오차가 있는지 계산하는 과정임
            Relative_L2_err_all = np.hstack((Relative_L2_err_all,Relative_L2_err.item()))   # relative_L2_err_all에 상대오차를 계속 누적시키면서 본다.



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

                # 240630 여기부터 다시 보고 위에 normalizing 하는것만 나중에 다시한번 확인하기
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


############################################ 여기서부터 constant value 가지고 학습하는 과정임 ##############
drive_path = 'vanilla/'
# drive_path = 'drive/MyDrive/test/1D_solver_1mat/vanilla/'



# material properties
rho = 7.860
k = 0.162
cp = 0.5
T0 = 40     # 이건 뭘까? 초기 온도값인가? 서류 찾아보기



layers = [2, 20, 20, 20, 20, 20, 20, 1]     # 첨에 시간t, 거리 x로 2개 input feature 받아서 마지막에 T 한개의 output feature만 내놓음

# domain x, t를 만든다!
domain1_x = np.linspace(0, 2.5, 21)
domain1_t = np.arange(0, 27+0.1, 0.1)


x, t = np.meshgrid(domain1_x, domain1_t)    # x가 0, 0.125, ... 2.5까지 만들어진 행이 계속 아래로 똑같이 반복
                                            # t가 0, 0.1, ... , 27까지 만들어진 열이 계속 오른쪽으로 반복

# hstack을 통해서 domain1_grid를 2개의 열벡터를 가진 행렬로 만들어냄 ㅎㄷㄷ[5691,2]
domain1_grid = np.hstack((x.flatten()[:,None], t.flatten()[:,None]))    # 뒤에 [:, None]에서 이거 없이 flatten만 하면 그냥 옆으로 쭉 퍼진 행만 있음. list는 아니고 그냥 1차원 행벡터
                                                                        # 뒤에 [,None]을 붙이면 열벡터로 전환됨

# [5691,2] 행렬을 point_all에 넣음
point_all = domain1_grid

# 겹치는 거 빼고 각각 한개의 요소만 index 기억하고 줄인다. 마지막에[1]은 결국 인덱스만 받아온다. 겹치는게 없을테니 indexes는 결국 5691개 요소를 가진 열벡터가 나올듯
indexes  = np.unique(point_all, axis=0, return_index=True)[1]   # 0, 21, 42, ..., 이런식으로 56xx까지 한 뒤에 1, 22, 이런식으로 또 쭉 한바퀴. 그래서 총 (5691,1)개로 나옴

point_all = point_all[np.sort(indexes),:]   # np.sort() 하면 결국 0,1,2,3,...,5690 이렇게 끝. 결국 point_all에 있는 걸 그대로 받아온거.
                                            # 이건 어떤 특이한 데이터를 받아와서 전처리 했을때 이렇게 썼을 것이라고 추정됨
print(point_all.shape)



# ground truth
path0 = 'data/'
# path0 = 'drive/MyDrive/test/1D_solver_1mat/data/'
temp1 = pd.read_csv(path0 + 'Run1_13.5_10.0_temp.csv', header=None)
temp1 = temp1.values[:,::10]        # pandas에서 읽어온 데이터 프레임의 형식을 numpy 형식으로 변경하는 것. 시간에 따른 각 위치의 값들을 행방향으로 쭉 1개로 쌓음
temp1_vec = torch.tensor(temp1.flatten()).float().to(device)



model = PINN(point_all, layers)
model.train(200000)