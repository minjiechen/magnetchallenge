from typing import Any
import torch
import math

from torch import optim, nn, Tensor
import lightning.pytorch as pl
import numpy as np

class Transformer(nn.Module):
    def __init__(self, 
        B_in_channel=1024,
        dim_hidden=24,  
        dim_proj_fusion=40,
        n_encoder_layers=1,
        n_heads=4,
        dropout_encoder=0.0,
        dropout_pos_enc=0.0,
        dim_feedforward_encoder=40,
        ): 

        #   Args:
        #    in_channel: int, number of input variables.
        #    dim_hidden: int, aka d_model. All sub-layers in the model produce outputs of dimension dim_val
        #    dim_proj_fusion :int, number of neurons in the linear layer of the fusion projecter
        #    n_encoder_layers: int, number of stacked encoder layers in the encoder
        #    n_heads: int, the number of attention heads (aka parallel attention layers)
        #    dropout_encoder: float, the dropout rate of the encoder
        #    dropout_pos_enc: float, the dropout rate of the positional encoder
        #    dim_feedforward_encoder: int, number of neurons in the linear layer of the encoder

        super().__init__() 

        self.proj_B = nn.Sequential(
            nn.Linear(1, dim_hidden),
            nn.Tanh(),
            nn.Linear(dim_hidden, dim_hidden))

        self.positional_encoding_layer = PositionalEncoding(d_model=dim_hidden, 
                                                            dropout=dropout_pos_enc, 
                                                            max_len=B_in_channel)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_hidden, 
            nhead=n_heads,
            dim_feedforward=dim_feedforward_encoder,
            dropout=dropout_encoder,
            activation="relu",
            batch_first=True
            ) 
        
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, 
                                             num_layers=n_encoder_layers, 
                                             norm=None)
        
        self.proj_fusion = nn.Sequential(
            nn.Linear(dim_hidden+2, dim_proj_fusion),
            nn.Tanh(),
            nn.Linear(dim_proj_fusion, dim_proj_fusion),
            nn.Tanh(),
            nn.Linear(dim_proj_fusion, 1))
        
        self.regressor = nn.Sequential(
            nn.Linear(B_in_channel, 1))

    def forward(self, B_curve: Tensor, Temp: Tensor, Freq: Tensor) -> Tensor:
        batch_size, len_seq, feat_dim = B_curve.shape
        B_curve = self.proj_B(B_curve) # (bs,1024,1)->(bs,1024,24)

        # input: (batch, seq, feature)
        B_curve = self.positional_encoding_layer(B_curve)
        B_curve = self.encoder(B_curve)

        Temp = Temp.unsqueeze(1).repeat(1, len_seq, 1) # (bs,1)->(bs,1024,1)
        Freq = Freq.unsqueeze(1).repeat(1, len_seq, 1) # (bs,1)->(bs,1024,1)

        feat = self.proj_fusion(torch.cat([B_curve, Temp, Freq], dim=2)) # (bs,1024,24+1+1)->(bs,1024,1)
        feat = feat.reshape(batch_size, -1) # (bs,1024,1)->(bs,1024)
        P_pred= self.regressor(feat) # (bs,1024)->(bs,1)

        return P_pred

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) 
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.size(1)]
        return self.dropout(x)

class Lit_model(pl.LightningModule):
    def __init__(self, net, learning_rate=0.003, save_dir='./', CLR_step_size=None, 
                 normB=None, normF=None, normP=None, sample_num=1024):
        super().__init__()
        self.net = net
        self.learning_rate = learning_rate
        self.CLR_step_size = CLR_step_size
        self.save_dir = save_dir
        self.normB = normB
        self.normF = normF
        self.normP = normP

        self.example_input_array = tuple((torch.Tensor(32, sample_num, 1), torch.Tensor(32, 1), torch.Tensor(32, 1), 
                                    torch.Tensor(32, 1), torch.Tensor(32, 1))) # a fake input for model summary

        self.metric = torch.nn.MSELoss()

        self.val_step_err = []
        self.val_step_loss = []
        self.test_pred_P = []
        self.test_step_err = []

        self.results = torch.tensor([]) # final results for submit

    def forward(self, in_B, in_F, in_T, out_P, gt_P):
        pred_P = self.net(in_B, in_F, in_T)
        return pred_P

    def training_step(self, batch, batch_idx):
        in_B, in_F, in_T, out_P, gt_P = batch

        pred_P = self.net(in_B, in_F, in_T)

        loss = self.metric(pred_P, out_P)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        in_B, in_F, in_T, out_P, gt_P = batch
        pred_P = self.net(in_B, in_F, in_T)

        retrans_pred_P = torch.exp(pred_P * self.normP[1] + self.normP[0])
        self.test_pred_P.append(retrans_pred_P)

        # If groundtruth is provided, calculate the errors.
        if not (gt_P == 0).all():
            Error_re = torch.abs(retrans_pred_P - gt_P) / torch.abs(gt_P) * 100
            self.test_step_err.append(Error_re)

    def on_test_epoch_end(self):
        # Print performance
        if len(self.test_step_err) > 0:
            test_epoch_err = torch.vstack(self.test_step_err)
            test_err = test_epoch_err.mean()
            test_err_95 = torch.quantile(test_epoch_err, 0.95, interpolation='nearest')
            print(f'test_err={test_err:.2f}, test_err95={test_err_95:.2f}')
            
            self.log("test_err", test_err)
            self.log('test_err95', test_err_95)

            with open(self.save_dir+'err.csv', "w") as f:
                np.savetxt(f, test_epoch_err.squeeze(1).cpu().numpy())
                f.close()   

        # Save results
        data_P = torch.vstack(self.test_pred_P)
        self.results = data_P.squeeze(1).cpu().numpy()

        # with open(self.save_dir+'pred_P.csv', "w") as f:
        #     np.savetxt(f, self.results)
        #     f.close()

    def validation_step(self, batch, batch_idx):
        in_B, in_F, in_T, out_P, gt_P = batch
        pred_P = self.net(in_B, in_F, in_T)
        val_loss = self.metric(pred_P, out_P)

        # compute relative error
        retrans_pred_P = torch.exp(pred_P * self.normP[1] + self.normP[0])
        Error_re = torch.abs(retrans_pred_P - gt_P) / torch.abs(gt_P) * 100

        self.log("val_loss", val_loss, prog_bar=True)

        self.val_step_err.append(Error_re)
        self.val_step_loss.append(val_loss)

    def on_validation_epoch_end(self):
        val_epoch_err = torch.vstack(self.val_step_err)
        val_err = val_epoch_err.mean()
        val_err_95 = torch.quantile(val_epoch_err, 0.95, interpolation='nearest')
        val_loss = torch.stack(self.val_step_loss).mean()
        print(f'\n val epoch={self.current_epoch}, loss={val_loss}, err={val_err:.2f}, '
              f'err95={val_err_95:.2f}')
        
        self.log("val_err",val_err)
        self.log('val_err_95', val_err_95)
        self.val_step_err.clear()
        self.val_step_loss.clear()   

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)

        max_lr = self.learning_rate
        base_lr = max_lr/4.0
        scheduler = optim.lr_scheduler.CyclicLR(optimizer,
            base_lr=base_lr,max_lr=max_lr,
            step_size_up=self.CLR_step_size, cycle_momentum=False) # stepsize=（样本个数/batchsize）*（2~10）
        
        return ([optimizer],[scheduler])