import argparse
import math
import time
import torch
import torch.nn as nn
from net import gtnet
import numpy as np
import importlib
import random
from util import *
from trainer import Optim
import sys
from random import randrange
from matplotlib import pyplot as plt
import time
import os
from pathlib import Path

plt.rcParams['savefig.dpi'] = 1200


# [개선 2] 상관계수 기반 그래프 생성 함수
def build_corr_adj(raw_TN: torch.Tensor, topk: int = 20):
    """
    raw_TN: [T, N] (훈련 구간 데이터)
    returns: [N, N] adjacency matrix (상관계수 기반)
    """
    x = raw_TN - raw_TN.mean(dim=0, keepdim=True)
    num = x.t() @ x  # [N, N]
    den = torch.sqrt((x**2).sum(dim=0, keepdim=True)).t() @ torch.sqrt((x**2).sum(dim=0, keepdim=True))
    corr = num / (den + 1e-8)
    corr = corr.abs()
    corr.fill_diagonal_(0)
    
    k = min(topk, corr.size(1))
    vals, idx = corr.topk(k, dim=1)
    A = torch.zeros_like(corr)
    A.scatter_(1, idx, vals)
    
    # 대칭화
    A = torch.maximum(A, A.t())
    return A
def inverse_diff_2d(output, I,shift):
    output[0,:]=torch.exp(output[0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[0]):
        output[i,:]= torch.exp(output[i,:]+torch.log(output[i-1,:]+shift))-shift
    return output

def inverse_diff_3d(output, I,shift):
    output[:,0,:]=torch.exp(output[:,0,:]+torch.log(I+shift))-shift
    for i in range(1,output.shape[1]):
        output[:,i,:]=torch.exp(output[:,i,:]+torch.log(output[:,i-1,:]+shift))-shift
    return output


def plot_data(data,title):
    x=range(1,len(data)+1)
    plt.plot(x,data,'b-',label='Actual')
    plt.legend(loc="best",prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03,fontsize=18)
    plt.ylabel("Trend",fontsize=15)
    plt.xlabel("Month",fontsize=15)
    locs, labs = plt.xticks() 
    plt.xticks(rotation='vertical',fontsize=13) 
    plt.yticks(fontsize=13)
    fig = plt.gcf()
    plt.show()


# for figure display, we rename columns
def consistent_name(name):

    if name=='CAPTCHA' or name=='DNSSEC' or name=='RRAM':
        return name

    #e.g., University of london
    if not name.isupper():
        words=name.split(' ')
        result=''
        for i,word in enumerate(words):
            if len(word)<=2: #e.g., "of"
                result+=word
            else:
                result+=word[0].upper()+word[1:]
            
            if i<len(words)-1:
                result+=' '

        return result
    

    words= name.split(' ')
    result=''
    for i,word in enumerate(words):
        if len(word)<=3 or '/' in word or word=='MITM' or word =='SIEM':
            result+=word
        else:
            result+=word[0]+(word[1:].lower())
        
        if i<len(words)-1:
            result+=' '
        
    return result

#computes and saves validation/testing error to a text file given a single node's prediction and actual curve values
def save_metrics_1d(predict, test, title, type):
    #RRSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator

    #Relative Absolute Error RAE  - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))

    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s) # calculate the mean of each column in test
    diff_r = test_s - mean_all # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    if root_sum_squared_r == 0:
        rrse = 0.0
    else:
        rrse = root_sum_squared / root_sum_squared_r

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements
    
    #Relative Absolute Error RAE
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()

    from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    save_path = str(PROJECT_DIR / 'model' / 'Bayesian' / type)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    title=title.replace('/','_')
    # 수정된 save_path 사용
    file_path = Path(save_path) / (title + '_' + type + '.txt')
    with open(file_path, "w", encoding="utf-8") as f:
      f.write('rse:'+str(rrse)+'\n')
      f.write('rae:'+str(rae)+'\n')
      f.close()


#plots predicted curve with actual curve. The x axis can be adjusted as needed
def plot_predicted_actual(predicted, actual, title, type, variance=None, confidence_95=None):

    months=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    M=[]
    
    # [수정] 데이터 시작(2011-01)부터 끝(2025-09)까지 정확하게 라벨 생성
    for year in range (11, 26):   # 11년 ~ 25년
        for month in months:
            # 2025년은 9월까지만 데이터가 있으므로 10,11,12월 제외
            if year == 25 and month in ['Oct', 'Nov', 'Dec']:
                continue
            M.append(month+'-'+str(year))   
    
    M2=[]
    p=[]
    
    # Testing 모드: 생성된 M의 "끝부분"을 가져옴 (이제 끝이 Sep-25로 고정되어 정확함)
    if type=='Testing':
        M = M[-len(predicted):] 
        
        for index, value in enumerate(M):
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1) 
    
    else: ## Validation x axis: 2023-2025 (수정됨)
        M = M[144:180] 
        
        for index, value in enumerate(M):
            # 3개월 단위로 x축 라벨 표시
            if 'Dec' in M[index] or 'Mar' in M[index] or 'Jun' in M[index] or 'Sep' in M[index]:
                M2.append(M[index])
                p.append(index+1)

    x = range(1, len(predicted) + 1)
    
    # --- 그래프 그리기 ---
    plt.figure(figsize=(12, 6)) # 그래프 크기 설정 (옵션)
    plt.plot(x, actual, 'b-', label='Actual')
    plt.plot(x, predicted, '--', color='purple', label='Predicted')
    
    # 신뢰구간 그리기 (None-safe)
    if (variance is not None) and (confidence_95 is not None):
        plt.fill_between(x, predicted - confidence_95, predicted + confidence_95, alpha=0.5, color='pink', label='95% Confidence')
    
    plt.legend(loc="best", prop={'size': 11})
    plt.axis('tight')
    plt.grid(True)
    plt.title(title, y=1.03, fontsize=18)
    plt.ylabel("Trend", fontsize=15)
    plt.xlabel("Month", fontsize=15)
    
    # x축 틱 설정
    plt.xticks(ticks=p, labels=M2, rotation='vertical', fontsize=13)
    plt.yticks(fontsize=13)
    
    # 저장 경로 설정
    from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    save_path = str(PROJECT_DIR / 'model' / 'Bayesian' / type)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    title = title.replace('/', '_')
    
    # 파일 저장
    file_path = Path(save_path) / (title + '_' + type + '.png')
    plt.savefig(str(file_path), bbox_inches="tight")
    # plt.savefig(save_path + title + '_' + type + ".pdf", bbox_inches = "tight", format='pdf') # pdf는 필요하면 주석 해제

    # plt.show(block=False) # 코랩 환경에서는 show()가 루프를 멈추게 할 수 있어 주석 처리하거나 주의 필요
    # plt.pause(2)
    plt.close()


#symmetric mean absolute percentage error (optional)
def s_mape(yTrue,yPred):
  mape=0
  for i in range(len(yTrue)):
    den = abs(yTrue[i]) + abs(yPred[i]) + 1e-8
    mape+= abs(yTrue[i]-yPred[i])/ den
  mape/=len(yTrue)

  return mape

#for testing the model on unseen data, a sliding window can be used when the output period of the model is smaller than the target period to be forecasted.
#The sliding window uses the output from previous step as input of the next step.
#In our case, the window was not slided (we predicted 36 months and the model by default predicts 36 months)
def evaluate_sliding_window(data, test_window, model, evaluateL2, evaluateL1, n_input, is_plot):
    # [개선 3] eval() 모드로 고정 (드롭아웃 비활성화, 성능 최우선)
    model.eval()
    
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    predictions = []
    sum_squared_diff=0
    sum_absolute_diff=0
    r=random.randint(0, 141)
    r=0 # we can choose any random node index for printing
    print('testing r=',str(r))
    scale = data.scale.expand(test_window.size(0), data.m) #scale will have the max of each column (142 max values)
    print('Test Window Feature:',test_window[:,r])
    
    x_input = test_window[0:n_input, :].clone() # Generate input sequence

    for i in range(n_input, test_window.shape[0],data.out_len):

        print('**************x_input*******************')
        print(x_input[:,r])#prints 1 random column in the sliding window
        print('**************-------*******************')

        X = torch.unsqueeze(x_input,dim=0)
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        X = X.to(torch.float)


        y_true = test_window[i: i+data.out_len,:].clone() 


        # [개선 3] torch.no_grad()로 감싸기 (메모리 효율 + 속도)
        with torch.no_grad():
            # Bayesian estimation은 선택사항 -> 일단 1회 forward로 성능 최우선
            output = model(X)
            y_pred = output[0, :, :, -1].clone()
            #if this is the last predicted window and it exceeds the test window range
            if y_pred.shape[0]>y_true.shape[0]:
                y_pred=y_pred[:-(y_pred.shape[0]-y_true.shape[0]),]
        
        # [개선 3] outputs는 단순 단일 예측값으로 사용
        outputs = y_pred
        
        # [개선 3] variance/confidence는 선택사항 (성능 최우선)
        var = None
        std_dev = None
        confidence = None



        #shift the sliding window
        if data.P<=data.out_len:
            x_input = y_pred[-data.P:].clone()
        else:
            x_input = torch.cat([x_input[ -(data.P-data.out_len):, :].clone(), y_pred.clone()], dim=0)


        print('----------------------------Predicted months',str(i-n_input+1),'to',str(i-n_input+data.out_len),'--------------------------------------------------')
        print(y_pred.shape,y_true.shape)
        y_pred_o=y_pred
        y_true_o=y_true
        for z in range(y_true.shape[0]):
            print(y_pred_o[z,r],y_true_o[z,r]) #only one col
        print('------------------------------------------------------------------------------------------------------------')


        if predict is None:
            predict = y_pred
            test = y_true
            variance=None
            confidence_95=None
        else:
            predict = torch.cat((predict, y_pred))
            test = torch.cat((test, y_true))


    scale = data.scale.expand(test.size(0), data.m) #scale will have the max of each column (142 max values)

    #inverse normalisation
    predict*=scale
    test*=scale
    # [개선 3] variance/confidence는 선택사항 제거


    #Relative Squared Error RSE according to Lai et.al - numerator
    sum_squared_diff = torch.sum(torch.pow(test - predict, 2))
    #Relative Absolute Error RAE - numerator
    sum_absolute_diff= torch.sum(torch.abs(test - predict))# numerator


    #Root Relative Squared Error RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #Root Relative Squared Error RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=0) # calculate the mean of each column in test call it Yj-
    diff_r = test_s - mean_all.expand(test_s.size(0), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r
    print('rrse=',root_sum_squared,'/',root_sum_squared_r)

    #Relative Absolute Error RAE - denominator
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    #Relative Absolute Error RAE
    rae=sum_absolute_diff/sum_absolute_r 
    rae=rae.item()
###########################################################################################################


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    den = (sigma_p * sigma_g) + 1e-8
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0) / den #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for z in range(Ytest.shape[1]):
        smape+=s_mape(Ytest[:,z],predict[:,z])
    smape/=Ytest.shape[1]

    #plot predicted vs actual and save errors to file
    counter = 0
    if is_plot:
        print("\n[Plotting] Saving graphs to Testing folder...")
        
        # 전체 142개를 다 그리지 않고, 환율 관련 주요 변수만 그리고 싶다면 범위를 조절하거나 조건을 거세요.
        # 예: 현재 r부터 r+142까지 돌지만, 너무 많다면 아래처럼 수정 가능
        # for v in range(data.m): # 전체 변수 순회
        
        for v in range(r, r+142): 
            col = v % data.m
            
            node_name = data.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name = consistent_name(node_name)
            
            # [옵션] 'fx'가 포함된 환율 컬럼만 그리고 싶다면 주석 해제
            if 'fx' not in node_name :
                continue 

            # save error to file
            save_metrics_1d(torch.from_numpy(predict[:,col]), torch.from_numpy(Ytest[:,col]), node_name, 'Testing')
            
            # plot - None-safe
            var_col = variance[:, col] if variance is not None else None
            ci_col = confidence_95[:, col] if confidence_95 is not None else None
            plot_predicted_actual(predict[:,col], Ytest[:,col], node_name, 'Testing', var_col, ci_col)
            counter += 1
            
        print(f"[Done] Saved {counter} graphs.")

    return rrse, rae, correlation, smape



def evaluate(data, X, Y, model, evaluateL2, evaluateL1, batch_size, is_plot):
    #model.eval()# To get Bayesian estimation, we must comment out this line
    total_loss = 0
    total_loss_l1 = 0
    n_samples = 0
    predict = None
    test = None
    variance=None
    confidence_95=None
    sum_squared_diff=0
    sum_absolute_diff=0
    r=0 #we choose any node index for printing (debugging)
    print('validation r=',str(r))

    for X, Y in data.get_batches(X, Y, batch_size, False):
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)

        # Bayesian estimation
        num_runs = 10

        # Create a list to store the outputs
        outputs = []

        # Run the model multiple times (10)
        with torch.no_grad():
            for _ in range(num_runs):
                output = model(X)
                output = output.squeeze(3)
                outputs.append(output)
            

        # Stack the outputs along a new dimension
        outputs = torch.stack(outputs)

        # Calculate mean, variance, and standard deviation
        mean = torch.mean(outputs, dim=0)
        var = torch.var(outputs, dim=0)#variance
        std_dev = torch.std(outputs, dim=0)#standard deviation

        # Calculate 95% confidence interval
        z=1.96
        confidence=z*std_dev/torch.sqrt(torch.tensor(num_runs))

        output=mean #we will consider the mean to be the prediction

        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (142 max values)
        
        #inverse normalisation
        output*=scale
        Y*=scale
        var*=scale
        confidence*=scale

        if predict is None:
            predict = output
            test = Y
            variance=var
            confidence_95=confidence
        else:
            predict = torch.cat((predict, output))
            test = torch.cat((test, Y))
            variance= torch.cat((variance, var))
            confidence_95=torch.cat((confidence_95,confidence))


        print('EVALUATE RESULTS:')
        scale = data.scale.expand(Y.size(0), Y.size(1), data.m) #scale will have the max of each column (142 max values)
        y_pred_o=output
        y_true_o=Y
        for z in range(Y.shape[1]):
            print(y_pred_o[0,z,r],y_true_o[0,z,r]) #only one col
        
        total_loss += evaluateL2(output, Y).item()
        total_loss_l1 += evaluateL1(output, Y).item()
        n_samples += (output.size(0) * output.size(1) * data.m)

        #RRSE according to Lai et.al
        sum_squared_diff += torch.sum(torch.pow(Y - output, 2))
        #Relative Absolute Error RAE - numerator
        sum_absolute_diff+=torch.sum(torch.abs(Y - output))

    #The below 2 lines are not used
    rse = math.sqrt(total_loss / n_samples) / data.rse 
    rae = (total_loss_l1 / n_samples) / data.rae 

    #RRSE according to Lai et.al - numerator
    root_sum_squared= math.sqrt(sum_squared_diff) #numerator
    
    #RRSE according to Lai et.al - denominator
    test_s=test
    mean_all = torch.mean(test_s, dim=(0,1)) # calculate the mean of each column in test
    diff_r = test_s - mean_all.expand(test_s.size(0), test_s.size(1), data.m) # subtract the mean from each element in the tensor test
    sum_squared_r = torch.sum(torch.pow(diff_r, 2))# square the result and sum over all elements
    root_sum_squared_r=math.sqrt(sum_squared_r)#denominator

    #RRSE according to Lai et.al
    rrse=root_sum_squared/root_sum_squared_r #RRSE

    #Relative Absolute Error RAE
    sum_absolute_r=torch.sum(torch.abs(diff_r))# absolute the result and sum over all elements - denominator
    rae=sum_absolute_diff/sum_absolute_r # RAE
    rae=rae.item()


    predict = predict.data.cpu().numpy()
    Ytest = test.data.cpu().numpy()
    sigma_p = (predict).std(axis=0)
    sigma_g = (Ytest).std(axis=0)
    mean_p = predict.mean(axis=0)
    mean_g = Ytest.mean(axis=0)
    index = (sigma_g != 0)
    correlation = ((predict - mean_p) * (Ytest - mean_g)).mean(axis=0)/ (sigma_p * sigma_g) #Pearson's correlation coefficient?
    correlation = (correlation[index]).mean()

    #s-mape
    smape=0
    for x in range(Ytest.shape[0]):
        for z in range(Ytest.shape[2]):
            smape+=s_mape(Ytest[x,:,z],predict[x,:,z])
    smape/=Ytest.shape[0]*Ytest.shape[2]


    #plot actual vs predicted curves and save errors to file
    counter = 0
    if is_plot:
        print("\n[Plotting] Saving Validation graphs (FX only)...")
        
        # 전체 변수(142개)를 순회하며 확인
        for v in range(r, r + 142):
            col = v % data.m
            
            # 노드 이름(변수명) 가공
            node_name = data.col[col].replace('-ALL','').replace('Mentions-','Mentions of ').replace(' ALL','').replace('Solution_','').replace('_Mentions','')
            node_name = consistent_name(node_name)
            
            # [추가된 부분] 'fx'가 포함되지 않은 변수는 건너뛰기 (저장 안 함)
            if 'fx' not in node_name.lower(): 
                continue

            # [중요 수정] 데이터 1개가 아닌 전체 시퀀스를 가져오도록 수정 (오차율 inf 방지)
            # 3차원 텐서일 경우와 2차원일 경우를 모두 고려
            if predict.ndim > 2:
                pred_save = predict[:, 0, col].flatten()
                y_save = Ytest[:, 0, col].flatten()
                
                # 그래프 그릴 때 필요한 분산/신뢰구간도 차원 맞춤 (텐서 -> numpy 변환)
                var_save = variance[:, 0, col].cpu().numpy().flatten() if variance is not None else None
                conf_save = confidence_95[:, 0, col].cpu().numpy().flatten() if confidence_95 is not None else None
            else:
                pred_save = predict[:, col]
                y_save = Ytest[:, col]
                var_save = variance[:, col].cpu().numpy() if variance is not None else None
                conf_save = confidence_95[:, col].cpu().numpy() if confidence_95 is not None else None

            # 1. 텍스트 파일로 오차율 저장
            save_metrics_1d(torch.from_numpy(pred_save), torch.from_numpy(y_save), node_name, 'Validation')
            
            # 2. 그래프 이미지 저장
            plot_predicted_actual(pred_save, y_save, node_name, 'Validation', var_save, conf_save)
            
            counter += 1
            
        print(f"[Done] Saved {counter} FX graphs to Validation folder.")

    return rrse, rae, correlation, smape


def train(data, X, Y, model, criterion, optim, batch_size):
    model.train()
    total_loss = 0
    n_samples = 0
    iter = 0

    for X, Y in data.get_batches(X, Y, batch_size, True):
        model.zero_grad()
        X = torch.unsqueeze(X,dim=1)
        X = X.transpose(2,3)
        if iter % args.step_size == 0:
            perm = np.random.permutation(range(args.num_nodes))
        num_sub = int(args.num_nodes / args.num_split)

        for j in range(args.num_split):
            if j != args.num_split - 1:
                id = perm[j * num_sub:(j + 1) * num_sub]
            else:
                id = perm[j * num_sub:]

            id = torch.tensor(id).to(device)
            tx = X[:, :, :, :] 
            ty = Y[:, :, :] 
            output = model(tx)           
            output = torch.squeeze(output,3)
            scale = data.scale.expand(output.size(0), output.size(1), data.m)
            scale = scale[:,:,:] 
            
            # scale 변수는 나중에 결과 출력용으로만 쓰고, 학습(Loss)에는 쓰지 않습니다.
            # scale = data.scale.expand(output.size(0), output.size(1), data.m) 
            # output *= scale  <-- 주석 처리 또는 삭제
            # ty *= scale      <-- 주석 처리 또는 삭제

            # [해결] 0~1 사이 값 그대로 오차를 계산합니다.
            loss = criterion(output, ty)
            loss.backward()
            total_loss += loss.item()
            n_samples += (output.size(0) * output.size(1) * data.m)
            
            grad_norm = optim.step()

        if iter%1==0:
            print('iter:{:3d} | loss: {:.3f}'.format(iter,loss.item()/(output.size(0) * output.size(1)* data.m)))
        iter += 1
    return total_loss / n_samples


parser = argparse.ArgumentParser(description='PyTorch Time series forecasting')
parser.add_argument('--data', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'FX_Data', 'ExchangeRate_dataset.csv'),
                    help='location of the data file')
parser.add_argument('--log_interval', type=int, default=2000, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default=os.path.join(os.path.dirname(__file__), '..', 'model', 'Bayesian', 'model.pt'),
                    help='path to save the final model')
parser.add_argument('--optim', type=str, default='adam')
parser.add_argument('--L1Loss', type=bool, default=True)
parser.add_argument('--normalize', type=int, default=2)
parser.add_argument('--device',type=str,default='cuda:1',help='')
parser.add_argument('--gcn_true', type=bool, default=True, help='whether to add graph convolution layer')
parser.add_argument('--buildA_true', type=bool, default=True, help='whether to construct adaptive adjacency matrix')
parser.add_argument('--gcn_depth',type=int,default=2,help='graph convolution depth')
parser.add_argument('--num_nodes',type=int,default=142,help='number of nodes/variables')
parser.add_argument('--dropout',type=float,default=0.15,help='dropout rate (권장: 0.3->0.15, RRSE 크면 낮추기)')
parser.add_argument('--subgraph_size',type=int,default=20,help='k')
parser.add_argument('--node_dim',type=int,default=40,help='dim of nodes')
parser.add_argument('--dilation_exponential',type=int,default=2,help='dilation exponential')
parser.add_argument('--conv_channels',type=int,default=16,help='convolution channels')
parser.add_argument('--residual_channels',type=int,default=16,help='residual channels')
parser.add_argument('--skip_channels',type=int,default=32,help='skip channels')
parser.add_argument('--end_channels',type=int,default=64,help='end channels')
parser.add_argument('--in_dim',type=int,default=1,help='inputs dimension')
parser.add_argument('--seq_in_len',type=int,default=10,help='input sequence length')
parser.add_argument('--seq_out_len',type=int,default=1,help='output sequence length')
parser.add_argument('--horizon', type=int, default=1) 
parser.add_argument('--layers',type=int,default=5,help='number of layers')

parser.add_argument('--batch_size',type=int,default=8,help='batch size')
parser.add_argument('--lr',type=float,default=0.001,help='learning rate')
parser.add_argument('--weight_decay',type=float,default=0.00001,help='weight decay rate')

parser.add_argument('--clip',type=int,default=10,help='clip')

parser.add_argument('--propalpha',type=float,default=0.05,help='prop alpha')
parser.add_argument('--tanhalpha',type=float,default=3,help='tanh alpha')

# [개선 4] 학습 설정 권장값으로 변경
parser.add_argument('--epochs',type=int,default=50,help='epochs (권장: 5->50)')
parser.add_argument('--num_split',type=int,default=1,help='number of splits for graphs')
parser.add_argument('--step_size',type=int,default=100,help='step_size')


args = parser.parse_args()
device = torch.device('cpu')
torch.set_num_threads(3)

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixed_seed = 123

def main(experiment):
    # Set fixed random seed for reproducibility
    set_random_seed(fixed_seed)

    #model hyper-parameters
    gcn_depths=[1,2,3]
    lrs=[0.01,0.001,0.0005,0.0008,0.0001,0.0003,0.005]#[0.00001,0.0001,0.0002,0.0003]
    convs=[4,8,16]
    ress=[16,32,64]
    skips=[64,128,256]
    ends=[256,512,1024]
    layers=[1,2]
    ks=[20,30,40,50,60,70,80,90,100]
    dropouts=[0.2,0.3,0.4,0.5,0.6,0.7]
    dilation_exs=[1,2,3]
    node_dims=[20,30,40,50,60,70,80,90,100]
    prop_alphas=[0.05,0.1,0.15,0.2,0.3,0.4,0.6,0.8]
    tanh_alphas=[0.05,0.1,0.5,1,2,3,5,7,9]


    best_val = 10000000
    best_rse=  10000000
    best_rae=  10000000
    best_corr= -10000000
    best_smape=10000000
    
    best_test_rse=10000000
    best_test_corr=-10000000

    best_hp=[]


    #random search
    for q in range(1):

        #hps - args 값을 직접 사용 (랜덤 서치 제거)
        gcn_depth = args.gcn_depth
        lr = args.lr
        conv = args.conv_channels
        res = args.residual_channels
        skip = args.skip_channels
        end = args.end_channels
        layer = args.layers
        k = args.subgraph_size
        dropout = args.dropout
        dilation_ex = args.dilation_exponential
        node_dim = args.node_dim
        prop_alpha = args.propalpha
        tanh_alpha = args.tanhalpha
        

        Data = DataLoaderS(args.data, 0.6, 0.2, device, args.horizon, args.seq_in_len, normalize=args.normalize, out=args.seq_out_len)

        args.num_nodes = Data.train[0].shape[2]
        
        # [개선 2] 그래프가 비어있으면 상관계수 기반 그래프 생성
        train_end = int(0.6 * Data.n)
        if (Data.adj is None) or (Data.adj.numel() == 0) or (Data.adj.sum().item() == 0):
            print("\n[GRAPH] predefined graph is empty -> building correlation graph from training data")
            A_corr = build_corr_adj(Data.dat[:train_end, :], topk=args.subgraph_size)
            Data.adj = A_corr
            print(f"[GRAPH] correlation adjacency created: shape={Data.adj.shape}, edges={Data.adj.nonzero().size(0)}")
        
        # [데이터 품질 점검] 30초 진단
        print("\n=== Data Quality Check ===")
        print("rawdat shape:", Data.rawdat.shape)
        print("rawdat min/max:", Data.rawdat.min().item(), "/", Data.rawdat.max().item())
        col_zero_ratio = (Data.rawdat == 0).float().mean(dim=0)
        print("zero ratio per column (top5):", col_zero_ratio.topk(5).values)
        print("================================\n")
    

        print('train X:',Data.train[0].shape)
        print('train Y:', Data.train[1].shape)
        print('valid X:',Data.valid[0].shape)
        print('valid Y:',Data.valid[1].shape)
        print('test X:',Data.test[0].shape)
        print('test Y:',Data.test[1].shape)
        print('test window:', Data.test_window.shape)

        print('length of training set=',Data.train[0].shape[0])
        print('length of validation set=',Data.valid[0].shape[0])
        print('length of testing set=',Data.test[0].shape[0])
        print('valid=',int((0.43 + 0.3) * Data.n))
        
       
        
        model = gtnet(args.gcn_true, args.buildA_true, gcn_depth, args.num_nodes,
                    device, Data.adj, dropout=dropout, subgraph_size=k,
                    node_dim=node_dim, dilation_exponential=dilation_ex,
                    conv_channels=conv, residual_channels=res,
                    skip_channels=skip, end_channels= end,
                    seq_length=args.seq_in_len, in_dim=args.in_dim, out_dim=args.seq_out_len,
                    layers=layer, propalpha=prop_alpha, tanhalpha=tanh_alpha, layer_norm_affline=False)
        

        print(args)
        print('\n' + '='*60)
        print('[GNN CHECK] --gcn_true:', args.gcn_true)
        print('[GNN CHECK] --buildA_true:', args.buildA_true)
        print('[GNN CHECK] Graph adjacency matrix shape:', Data.adj.shape)
        print('[GNN CHECK] Graph edges (non-zero):', (Data.adj > 0).sum().item())
        print('='*60 + '\n')
        print('The recpetive field size is', model.receptive_field)
        nParams = sum([p.nelement() for p in model.parameters()])
        print('Number of model parameters is', nParams, flush=True)

        # [개선 4] SmoothL1Loss (Huber) 사용: outlier에 덜 민감
        criterion = nn.SmoothL1Loss(reduction='sum').to(device)
        evaluateL2 = nn.MSELoss(reduction='sum').to(device) #MSE
        evaluateL1 = nn.L1Loss(reduction='sum').to(device) #MAE

        optim = Optim(
            model.parameters(),
            args.optim,
            lr,
            args.clip,
            weight_decay=args.weight_decay,
            lr_gamma=None
        )
        
        es_counter=0 #early stopping
        # At any point you can hit Ctrl + C to break out of training early.
        try:
            print('begin training')
            for epoch in range(1, args.epochs + 1):
                print('Experiment:',(experiment+1))
                print('Iter:',q)
                print('epoch:',epoch)
                print('hp=',[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch])
                print('best sum=',best_val)
                print('best rrse=',best_rse)
                print('best rrae=',best_rae)
                print('best corr=',best_corr)
                print('best smape=',best_smape)       
                print('best hps=',best_hp)
                print('best test rse=',best_test_rse)
                print('best test corr=',best_test_corr)

                
                es_counter+=1 # feel free to use this for early stopping (not used)

                epoch_start_time = time.time()
                train_loss = train(Data, Data.train[0], Data.train[1], model, criterion, optim, args.batch_size)
                val_loss, val_rae, val_corr, val_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                                 args.batch_size,False)
                print(
                    '| end of epoch {:3d} | time: {:5.2f}s | train_loss {:5.4f} | valid rse {:5.4f} | valid rae {:5.4f} | valid corr  {:5.4f} | valid smape  {:5.4f}'.format(
                        epoch, (time.time() - epoch_start_time), train_loss, val_loss, val_rae, val_corr, val_smape), flush=True)
                # Save the model if the validation loss is the best we've seen so far.
                sum_loss=val_loss+val_rae-val_corr
                if (not math.isnan(val_corr)) and val_loss < best_rse:
                    with open(args.save, 'wb') as f:
                        torch.save(model, f)
                    best_val = sum_loss
                    best_rse= val_loss
                    best_rae= val_rae
                    best_corr= val_corr
                    best_smape=val_smape

                    best_hp=[gcn_depth,lr,conv,res,skip,end, k, dropout, dilation_ex, node_dim, prop_alpha, tanh_alpha, layer, epoch]
                    
                    es_counter=0
                    
                    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                           args.seq_in_len, False) 
                    print('********************************************************************************************************')
                    print("test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f}| test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape), flush=True)
                    print('********************************************************************************************************')
                    best_test_rse=test_acc
                    best_test_corr=test_corr

        except KeyboardInterrupt:
            print('-' * 89)
            print('Exiting from training early')

    print('best val loss=',best_val)
    print('best hps=',best_hp)
    #save best hp to desk
    from pathlib import Path
    PROJECT_DIR = Path(__file__).resolve().parents[1]
    hp_path = PROJECT_DIR / 'model' / 'Bayesian' / 'hp.txt'
    hp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hp_path, "w", encoding="utf-8") as f:
        f.write(str(best_hp))
        f.close()
    # Load the best saved model.
    with open(args.save, 'rb') as f:
        model = torch.load(f, weights_only=False)

    vtest_acc, vtest_rae, vtest_corr, vtest_smape = evaluate(Data, Data.valid[0], Data.valid[1], model, evaluateL2, evaluateL1,
                                         args.batch_size, True)

    test_acc, test_rae, test_corr, test_smape = evaluate_sliding_window(Data, Data.test_window, model, evaluateL2, evaluateL1,
                                         args.seq_in_len, True) 
    print('********************************************************************************************************')    
    print("final test rse {:5.4f} | test rae {:5.4f} | test corr {:5.4f} | test smape {:5.4f}".format(test_acc, test_rae, test_corr, test_smape))
    print('********************************************************************************************************')
    return vtest_acc, vtest_rae, vtest_corr, vtest_smape, test_acc, test_rae, test_corr, test_smape

if __name__ == "__main__":
    vacc = []
    vrae = []
    vcorr = []
    vsmape=[]
    acc = []
    rae = []
    corr = []
    smape=[]
    for i in range(1):
        val_acc, val_rae, val_corr, val_smape, test_acc, test_rae, test_corr, test_smape = main(i)
        vacc.append(val_acc)
        vrae.append(val_rae)
        vcorr.append(val_corr)
        vsmape.append(val_smape)
        acc.append(test_acc)
        rae.append(test_rae)
        corr.append(test_corr)
        smape.append(test_smape)
    print('\n\n')
    print('1 run average')
    print('\n\n')
    print("valid\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(vacc), np.mean(vrae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(vacc), np.std(vrae)))
    print('\n\n')
    print("test\trse\trae")
    print("mean\t{:5.4f}\t{:5.4f}".format(np.mean(acc), np.mean(rae)))
    print("std\t{:5.4f}\t{:5.4f}".format(np.std(acc), np.std(rae)))

