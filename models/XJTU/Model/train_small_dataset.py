import torch
import torch.nn as nn
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import random
import time

pretrain_model_path = "./model/" + "N30"
save_figure_path = "./figure/" + "Material B"
save_model_path = "./modeln/" + "Material B"
data_path = "../Test_Data/Training/" + "Material B"

seq = pd.read_csv(data_path+"/B_Field.csv", header=None)
f = pd.read_csv(data_path+"/Frequency.csv", header=None)
T = pd.read_csv(data_path+"/Temperature.csv", header=None)
lab = pd.read_csv(data_path+"/Volumetric_Loss.csv", header=None)

seq = np.array(seq)
f = np.array(f)
T = np.array(T)
lab = np.array(lab)

random.seed(2)
data = np.column_stack((seq,f,T,lab))
np.random.shuffle(data)   # 随机打乱样本位置

seq = data[:,0:1024]
f = np.log(data[:,1024])
T = (data[:,1025])
lab = np.log(data[:,1026])

Bm=[]
for i in range(seq.shape[0]):
    Bm.append(np.max(seq[i,:]))
Bm = np.log(Bm)


data_size = data.shape[0]
train_size = round(0.95*data_size)
#train_size = 1400
valid_size = round(0*data_size)
#valid_size = 800
test_size = round(0.05*data_size)-1
#test_size = 1000


scaler = MinMaxScaler(feature_range=(-1, 1))
seq_normalized = scaler.fit_transform(seq.reshape(-1,1)).reshape(data_size,1024)
f_normalized = scaler.fit_transform(f.reshape(-1,1)).reshape(data_size,1)
T_normalized = scaler.fit_transform(T.reshape(-1,1)).reshape(data_size,1)
Bm_normalized = scaler.fit_transform(Bm.reshape(-1,1)).reshape(data_size,1)
lab_normalized = scaler.fit_transform(lab.reshape(-1,1)).reshape(data_size,1)


train_seq_normalized = seq_normalized[0:train_size]
train_f_normalized = f_normalized[0:train_size]
train_T_normalized = T_normalized[0:train_size]
train_Bm_normalized = Bm_normalized[0:train_size]
train_lab_normalized = lab_normalized[0:train_size]
valid_seq_normalized = seq_normalized[train_size:train_size+valid_size]
valid_f_normalized = f_normalized[train_size:train_size+valid_size]
valid_T_normalized = T_normalized[train_size:train_size+valid_size]
valid_Bm_normalized = Bm_normalized[train_size:train_size+valid_size]
valid_lab_normalized = lab_normalized[train_size:train_size+valid_size]
test_seq_normalized = seq_normalized[train_size+valid_size:train_size+valid_size+test_size]
test_f_normalized = f_normalized[train_size+valid_size:train_size+valid_size+test_size]
test_T_normalized = T_normalized[train_size+valid_size:train_size+valid_size+test_size]
test_Bm_normalized = Bm_normalized[train_size+valid_size:train_size+valid_size+test_size]
test_lab_normalized = lab_normalized[train_size+valid_size:train_size+valid_size+test_size]


train_seq = torch.FloatTensor(train_seq_normalized)
train_f = torch.FloatTensor(train_f_normalized)
train_T = torch.FloatTensor(train_T_normalized)
train_Bm = torch.FloatTensor(train_Bm_normalized)
train_lab = torch.FloatTensor(train_lab_normalized)
valid_seq = torch.FloatTensor(valid_seq_normalized)
valid_f = torch.FloatTensor(valid_f_normalized)
valid_T = torch.FloatTensor(valid_T_normalized)
valid_Bm = torch.FloatTensor(valid_Bm_normalized)
valid_lab = torch.FloatTensor(valid_lab_normalized)
test_seq = torch.FloatTensor(test_seq_normalized)
test_f = torch.FloatTensor(test_f_normalized)
test_T = torch.FloatTensor(test_T_normalized)
test_Bm = torch.FloatTensor(test_Bm_normalized)
test_lab = torch.FloatTensor(test_lab_normalized)

train_seq = train_seq.unsqueeze(2)
valid_seq = valid_seq.unsqueeze(2)
test_seq = test_seq.unsqueeze(2)


input_size = 16
hidden_size = 32
num_layers = 1
dropout = 0.2
output_size = 1
batch_size = 32
epochs = 500
batch = int(train_size/batch_size)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super().__init__()
        self.input_size= input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size= output_size
        self.num_directions = 2
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=25),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=10, stride=10)
            )
        # 输入通道数in_channels，输出通道数out_channels，卷积核大小25
        # 池化层最大池化，卷积核大小25，移动步长1
        # 卷积层输入为（batch_size, feature or out_channels, seq_len）
        # 卷积层输出为（batch_size, feature or out_channels, seq_len）
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers, batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(
            nn.Linear(self.num_directions * self.hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 13)
            )
        self.fc2 = nn.Sequential(
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
            )


    def forward(self, x, y, z, w):
        x = x.permute(0, 2, 1)   # 第二维度和第三维度交换位置
        x = self.conv(x)
        x = x.permute(0, 2, 1)   # 第二维度和第三维度交换位置
        h_0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)  # 构造h0
        c_0 = torch.zeros(self.num_directions * self.num_layers, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = out[:,-1,:]
        out = self.fc1(out)
        out = self.fc2(torch.cat((out,y,z,w),1))
        return out


model = LSTM(input_size, hidden_size, num_layers, output_size)  # 创建LSTM类对象
net = torch.load(pretrain_model_path+"/model.pth")
model.load_state_dict(net)
loss_function = nn.MSELoss()  # 损失函数
lr_list = []
init_lr=0.0001

for name, param in model.named_parameters():
    if "lstm" in name:
        param.requires_grad = False
    if "conv" in name:
        param.requires_grad = False
        
optimizer = torch.optim.Adam(filter(lambda p:p.requires_grad, model.parameters()), lr = init_lr)  # 优化器


lambda1 = lambda epoch: (1-epoch/epochs)**4   # 动态学习率
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,lr_lambda = lambda1)


# 模型训练
epochs_t = np.arange(1, epochs+1, 1)
train_loss_t = []
valid_loss_t = []
start_time = time.time()
for epoch in range(epochs):
    i = epoch
    for j in range(batch):
        y_pred = model(train_seq[batch_size*j:batch_size*(j+1)],train_f[batch_size*j:batch_size*(j+1)],
                       train_T[batch_size*j:batch_size*(j+1)],train_Bm[batch_size*j:batch_size*(j+1)])
        loss = loss_function(y_pred, train_lab[batch_size*j:batch_size*(j+1)])
        print(f'epoch:{i:3}    batch:{j:3}')
        print('train_error：',loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    scheduler.step()
    lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])

    train_pred = model(train_seq,train_f,train_T,train_Bm)
    train_loss = loss_function(train_pred, train_lab)
    train_loss_t.append(train_loss.item())
    valid_pred = model(valid_seq,valid_f,valid_T,valid_Bm)
    valid_loss = loss_function(valid_pred, valid_lab)
    valid_loss_t.append(valid_loss.item())
    
train_end_time = time.time()    
print(f"训练共计时{train_end_time - start_time:.2f}秒")



# 模型预测
model.eval()
with torch.no_grad():
    pred = model(test_seq,test_f,test_T,test_Bm)

pred = scaler.inverse_transform(pred.reshape(-1,1))
lab = scaler.inverse_transform(test_lab.reshape(-1,1))
pred = np.exp(pred)
lab = np.exp(lab)
test_error = np.zeros(test_size)

for i in range(test_size):
    test_error[i] = abs((pred[i]-lab[i])/lab[i]*100)

mean_error = np.mean(test_error)
max_error = np.max(test_error)
per95_error = np.percentile(test_error, 95)
print('mean_error:{:.2f}%'.format(mean_error))
print('max_error:{:.2f}%'.format(max_error))
print('per95_error:{:.2f}%'.format(per95_error))

'''
font = 'Times New Roman'
fontdict = {'family' : 'Times New Roman', 'size':16}
legend_fontdict = {'family' : 'Times New Roman', 'size':14}

plt.figure(1)  # 迭代误差图
plt.plot(epochs_t,train_loss_t,'b',epochs_t,valid_loss_t,'r')
plt.title('Error',fontdict = fontdict)
plt.xlabel('Epochs',fontdict = fontdict)
plt.ylabel('Error',fontdict = fontdict)
plt.xticks(fontproperties = font, size = 15)
plt.yticks(fontproperties = font, size = 15)
plt.legend(["Train dataset","Validation dataset"],prop = legend_fontdict,loc = 'upper right')
plt.savefig(save_figure_path+"/Error.png", bbox_inches='tight')


plt.figure(2)   # 测试集数据和预测结果对比
plt.plot(range(test_size),pred/1000,'b',range(test_size),lab/1000,'r')
plt.title('Predicted and measured results of test dataset',fontdict = fontdict)
plt.xlabel('Data',fontdict = fontdict)
plt.ylabel('Core loss (kW·m-3)',fontdict = fontdict)
plt.xticks(fontproperties = font, size = 15)
plt.yticks(fontproperties = font, size = 15)
plt.legend(["Predicted","Measured"],prop = legend_fontdict,loc='upper right')
plt.savefig(save_figure_path+"/Predicted and measured results of test dataset.png", bbox_inches='tight')


mean_error = np.mean(test_error)
max_error = np.max(test_error)
per95_error = np.percentile(test_error, 95)
print('mean_error:{:.2f}%'.format(mean_error))
print('max_error:{:.2f}%'.format(max_error))
print('per95_error:{:.2f}%'.format(per95_error))

plt.figure(3)   # 误差散点图
plt.semilogx(lab/1000,test_error,'.', color="blue")
plt.title('Scatter plot of core loss error',fontdict = fontdict)
plt.xlabel('Core loss (kW·m-3)',fontdict = fontdict)
plt.ylabel('Error (%)',fontdict = fontdict)
plt.xticks(fontproperties = font, size = 15)
plt.yticks(fontproperties = font, size = 15)
plt.savefig(save_figure_path+"/Scatter plot of core loss error.png", bbox_inches='tight')


plt.figure(4)   # 误差直方图
plt.hist(test_error,bins=50,density=True,facecolor='b')
plt.axvline(mean_error, 0, 0.9, color='r', linestyle='dashed', linewidth=1)
plt.axvline(max_error, 0, 0.1, color='r', linestyle='dashed', linewidth=1)
plt.axvline(per95_error, 0, 0.3, color='r', linestyle='dashed', linewidth=1)
min_ylim, max_ylim = plt.ylim()
plt.text(mean_error, max_ylim*0.9, 'Mean={:.2f}%'.format(mean_error),
         fontsize=13, color='r',family='Times New Roman')
plt.text(max_error*0.8, max_ylim*0.1, 'Max={:.2f}%'.format(max_error),
         fontsize=13, color='r',family='Times New Roman')
plt.text(per95_error, max_ylim*0.3, 'Prct_95={:.2f}%'.format(per95_error),
         fontsize=13, color='r',family='Times New Roman')
plt.title('Histogram of core loss error',fontdict = fontdict)
plt.xlabel('Relative error (%)',fontdict = fontdict)
plt.ylabel('Ratio of data points',fontdict = fontdict)
plt.xticks(fontproperties = font, size = 15)
plt.yticks(fontproperties = font, size = 15)
plt.savefig(save_figure_path+"/Histogram of core loss error.png", bbox_inches='tight')


plt.figure(5)   # 动态学习率
plt.plot(range(epochs),lr_list,color = 'r')
plt.title('Dynamic learning rate',fontdict = fontdict)
plt.xlabel('Epochs',fontdict = fontdict)
plt.ylabel('Learning rate',fontdict = fontdict)
plt.xticks(fontproperties = font, size = 15)
plt.yticks(fontproperties = font, size = 15)
plt.savefig(save_figure_path+"/Dynamic learning rate.png", bbox_inches='tight')

plt.show()
print(time.ctime())


torch.save(model.state_dict(), save_model_path+"/model.pth") 

'''
