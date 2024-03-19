import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MetaLMgenerator(nn.Module):
    def __init__(self, n_features, n_channels, n_hid=64):
        super(MetaLMgenerator, self).__init__()
        self.linear1 = nn.Linear(n_features, n_hid)
        self.sig = nn.Sigmoid()
        self.linear2 = nn.Linear(n_hid, n_hid // 4)
        self.linear3 = nn.Linear(n_hid // 4, n_channels * 2)

    def forward(self, x, shared_weights=None):
        if shared_weights is not None:  # weight sharing
            self.linear1.weight = shared_weights[0]
            self.linear2.weight = shared_weights[1]

        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)
        x = self.sig(x)
        x = self.linear3(x)

        out = self.sig(x)
        return out, [self.linear1.weight, self.linear2.weight]
    
class MetaLMlayer(nn.Module):

    def __init__(self, n_metadata, n_channels):
        super(MetaLMlayer, self).__init__()

        self.batch_size = None
        self.data_length = None
        self.hidden_size = None
        self.generator = MetaLMgenerator(n_metadata, n_channels)
        # Add the parameters gammas and betas to access them out of the class.
        self.gammas = None
        self.betas = None

    def forward(self, feature_maps, context, w_shared):
        data_shape = feature_maps.data.shape
        if len(data_shape) == 3:
            _, self.data_length, self.hidden_size = data_shape
        else:
            raise ValueError("Data should be 1D (tensor length: 3), found shape: {}".format(data_shape))

        # Estimate the MetaLM parameters using a MetaLM generator from the contioning metadata
        #print("context:", context.shape)
        metalm_params, new_w_shared = self.generator(context, w_shared)

        # MetaLM applies a different affine transformation to each channel,
        # consistent accross spatial locations

        metalm_params = metalm_params.unsqueeze(1)
        #print("metalm_params:", metalm_params.shape)
        metalm_params = metalm_params.repeat(1,self.data_length,1)
        #print(metalm_params.shape)
        self.gammas = metalm_params[:,:,:self.hidden_size]
        self.betas = metalm_params[:,:,self.hidden_size:]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared
    
    


class UNetGRU(nn.Module):
    def __init__(self, meta_dim, input_dim, hidden_dim, output_dim, attention_channels):
        super(UNetGRU, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Encoder layers
        self.encoder1 = nn.GRU(input_dim, hidden_dim, batch_first=True)
        self.encoder2 = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # Attention module
        self.attention = nn.Sequential(
            nn.Conv1d(hidden_dim, attention_channels, kernel_size=1),  # Adjust attention_channels as needed
            nn.ReLU(inplace=True),
            nn.Conv1d(attention_channels, 1, kernel_size=1),
            nn.Sigmoid()  # Sigmoid to create attention weights
        )
        
        # Decoder layers
        self.decoder1 = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.decoder2 = nn.GRU(hidden_dim + input_dim, hidden_dim, batch_first=True)
        
        # Meta generator layers
        self.meta_layer1 = MetaLMlayer(meta_dim, hidden_dim)
        self.meta_layer2 = MetaLMlayer(meta_dim, hidden_dim)
        self.meta_layer3 = MetaLMlayer(meta_dim, hidden_dim)
        self.meta_layer4 = MetaLMlayer(meta_dim, hidden_dim)


        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, context=None):
        # Encoder
        enc1_output, enc1_hidden = self.encoder1(x)
        enc1_output, w_film = self.meta_layer1(enc1_output, context, None)
        enc2_output, enc2_hidden = self.encoder2(enc1_output)
        enc2_output, w_film = self.meta_layer2(enc2_output, context, None if 'w_film' not in locals() else w_film)

        

        # Attention module
        attention_weights = self.attention(enc2_output.permute(0, 2, 1))  # Permute for 1D convolution
        attended_features = enc2_output * attention_weights.permute(0, 2, 1)

        
        
        # Decoder
        dec1_output, _ = self.decoder1(torch.cat((attended_features, enc1_output), dim=2))
        dec1_output, w_film = self.meta_layer3(dec1_output, context, None if 'w_film' not in locals() else w_film)
        dec2_output, _ = self.decoder2(torch.cat((dec1_output, x), dim=2))
        dec2_output, w_film = self.meta_layer4(dec2_output, context, None if 'w_film' not in locals() else w_film)

        
        # Output prediction
        output = self.output(dec2_output)
        
        return output
