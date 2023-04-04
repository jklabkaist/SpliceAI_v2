import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import numpy as np
import math
import pickle
from sklearn.metrics import average_precision_score

from dc1d.nn import *



CL_max=10000
SL=5000


def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp



def load_obj(name ):
    with open(name+'.pkl', 'rb') as f:
        return pickle.load(f)


def shuffle(arr):
    return np.random.choice(arr, size=len(arr), replace=False)


def get_datapoint(X,Y):
    Y=np.asarray(Y,dtype=np.float32)

    X, Y = clip_datapoints(X, Y, 10000, 1)
    X=torch.Tensor(X)
    Y=torch.Tensor(Y[0])
    return X,Y


def print_topl_statistics(y_true, y_pred):
    # Prints the following information: top-kL statistics for k=0.5,1,2,4,
    # auprc, thresholds for k=0.5,1,2,4, number of true splice sites.

    idx_true = np.nonzero(y_true == 1)[0]
    argsorted_y_pred = np.argsort(y_pred)
    sorted_y_pred = np.sort(y_pred)

    topkl_accuracy = []
    threshold = []
    
    mean_value=0.0
    wrong_mean_value=0.0
    
    for top_length in [1]:

        idx_pred = argsorted_y_pred[-int(top_length*len(idx_true)):]
        wrong_idx_pred = argsorted_y_pred[:int(top_length*len(idx_true))]

        topkl_accuracy += [np.size(np.intersect1d(idx_true, idx_pred)) \
                  / float(min(len(idx_pred), len(idx_true)))]
        
        
        mean_value=np.mean(y_pred[idx_true])
        wrong_mean_value=np.mean(y_pred[wrong_idx_pred])
        

    auprc = average_precision_score(y_true, y_pred)

    print(("%.4f\t||\t%.4f\t%.4f\t%d") % (
          mean_value, topkl_accuracy[0],
          auprc, len(idx_true)))
    
    return topkl_accuracy[0]


def clip_datapoints(X, Y, CL, N_GPUS):
    # This function is necessary to make sure of the following:
    # (i) Each time model_m.fit is called, the number of datapoints is a
    # multiple of N_GPUS. Failure to ensure this often results in crashes.
    # (ii) If the required context length is less than CL_max, then
    # appropriate clipping is done below.
    # Additionally, Y is also converted to a list (the .h5 files store 
    # them as an array).

    rem = X.shape[0]%N_GPUS
    clip = (CL_max-CL)//2

    if rem == 0 and clip != 0:
        return X[:, clip:-clip], [Y[t] for t in range(1)]
    else:
        return X, [Y[t] for t in range(1)]

def BCE_loss(pred, target):
    loss = nn.BCELoss()
    output = loss(pred , target)
    return output


def CE_loss(pred, target):
    loss =  nn.CrossEntropyLoss()
    output = loss(pred , target)
    return output


def custom_CE_loss(y_pred, y_true):
    loss = -torch.mean(  y_true[:, :, 0] * torch.log(y_pred[:, :, 0]+1e-10) 
                       + y_true[:, :, 1] * torch.log(y_pred[:, :, 1]+1e-10)
                       + y_true[:, :, 2] * torch.log(y_pred[:, :, 2]+1e-10)
                      )
    return loss







class ResidualUnit(nn.Module):
    def __init__(self, out_channels, kernel_size, dilation_rate):
        super().__init__()
        # l, out_channels 
        # w, kernel_size
        # ar, dilation_rate
        

        self.bn1 = nn.BatchNorm1d(out_channels , eps = 0.001 , momentum = 0.99)
        self.act = nn.ReLU()
        pad_size = math.ceil(dilation_rate * (kernel_size - 1) /2)
        self.conv1 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad_size, dilation=dilation_rate)
        self.bn2 = nn.BatchNorm1d(out_channels  , eps = 0.001 , momentum = 0.99)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=pad_size, dilation=dilation_rate)

        self.init_weights()

    def forward(self, x):

        
        
        
        bn1 = self.bn1(x)
        act1 = self.act(bn1)
        conv1 = self.conv1(act1)
        bn2 = self.bn2(conv1)
        act2 = self.act(bn2)
        conv2 = self.conv2(act2)

        output = conv2 + x

        return output


    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             nn.init.xavier_uniform_(module.weight , gain=torch.nn.init.calculate_gain('relu'))
    
    
    
            if isinstance(module, (nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()

                

                

                
                
                
class ResidualDCN(nn.Module):
    def __init__(self, out_channels, kernel_size, dilation_rate):
        super().__init__()
        # l, out_channels 
        # w, kernel_size
        # ar, dilation_rate
        

        self.bn1 = nn.BatchNorm1d(out_channels , eps = 0.001 , momentum = 0.99)
        self.act = nn.ReLU()
        pad_size = math.ceil(dilation_rate * (kernel_size - 1) /2)
        self.bn2 = nn.BatchNorm1d(out_channels  , eps = 0.001 , momentum = 0.99)

        self.offset1 = nn.Conv1d(out_channels, kernel_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        self.offset2 = nn.Conv1d(out_channels, kernel_size, kernel_size=kernel_size, padding=(kernel_size-1)//2)
        
        self.deconv1 = DeformConv1d( in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = "same", dilation = dilation_rate, groups = 1, bias = True, padding_mode='zeros')
        self.deconv2 = DeformConv1d( in_channels = out_channels, out_channels = out_channels, kernel_size = kernel_size, padding = "same", dilation = dilation_rate, groups = 1, bias = True, padding_mode='zeros')
        
        
        self.offset1.bias.data.zero_()
        self.offset2.bias.data.zero_()
        
        self.deconv1.bias.data.zero_()
        self.deconv2.bias.data.zero_()
        
        
        self.init_weights()

    def forward(self, x):
        
        bn1 = self.bn1(x)
        act1 = self.act(bn1)
        
        offset1 = self.offset1(act1)
        offset1 = torch.reshape(offset1,(offset1.shape[0],1,offset1.shape[1],offset1.shape[2]))
        offset1 = torch.transpose(offset1,2,3)
        deconv1 = self.deconv1(act1, offset1)
        
            
        bn2 = self.bn2(deconv1)
        act2 = self.act(bn2)
        
        offset2 = self.offset2(act2)
        offset2 = torch.reshape(offset2,(offset2.shape[0],1,offset2.shape[1],offset2.shape[2]))
        offset2 = torch.transpose(offset2,2,3)
        deconv2 = self.deconv2(act2, offset2)
        
        output = deconv2 + x

        return output


    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
#             module.weight.data.normal_(mean=0.0, std=0.02)
#             nn.init.xavier_uniform_(module.weight , gain=torch.nn.init.calculate_gain('relu'))
    
    
    
            if isinstance(module, (nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
                
                

                

                
                
                
                
                
                
class SpliceAI_V_DCN(nn.Module):
    def __init__(self, out_channels, kernel_sizes, dilation_rates, tf_dict=None, offset_dict=None):
        # L, out_channels: Number of convolution kernels, scalar
        # W, kernel_sizes: Convolution window size in each residual unit, array of int
        # AR, dilation_rates: Atrous rate in each residual unit, array of int
        super().__init__()
        
        assert len(kernel_sizes) == len(dilation_rates)


        self.dict_index = {(1, 4, 16): 0, (1, 16, 16): 0, (11, 16, 16): 0, (21, 16, 16): 0, (41, 16, 16): 0, (1, 16, 3): 0}
        self.tf_dict = tf_dict
        
        self.need_index={(11, 16, 11): 0, (21, 16, 21): 0, (41, 16, 41): 0}
        self.need_dict = offset_dict
        
        self.all_key={}
        
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates

        self.CL = int(2 * np.sum(dilation_rates*(kernel_sizes-1)))
        self.conv = nn.Conv1d( 4 , out_channels, kernel_size=1)
        self.skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        residual_units = []
        dense_layers = []
        for i in range(len(kernel_sizes)):
            
                
            residual_unit = ResidualDCN(out_channels, kernel_sizes[i], dilation_rates[i])
            residual_units.append(residual_unit)
                
                
                
            if (((i+1) % 4 == 0) or ((i+1) == len(kernel_sizes))):
                dense_layer = nn.Conv1d(out_channels, out_channels, kernel_size=1)
                dense_layers.append(dense_layer)
                
        self.residual_units = nn.ModuleList(residual_units)
        self.dense_layers = nn.ModuleList(dense_layers)

        # Classification layer
        self.class_output_layer = nn.Conv1d(out_channels, 3, kernel_size=1) # 3 classes: acceptor, donor and none
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

        
    def forward(self, x , testing=None):
        x=torch.transpose(x , 2, 1)
        
        
        conv = self.conv(x)
        skip = self.skip(conv)
        
        dense_ind = 0
        for i in range(len(self.kernel_sizes)):
            residual_unit = self.residual_units[i]
            conv = residual_unit(conv)
            
            if (((i+1) % 4 == 0) or ((i+1) == len(self.kernel_sizes))):
                # Skip connections to the output after every 4 residual units
                dense_layer = self.dense_layers[dense_ind]
                dense = dense_layer(conv)
                skip = skip + dense
                dense_ind += 1

        # Remove flanking ends
        skip = skip[:, :, int(self.CL/2):int(-self.CL/2 )] # replaces Keras' Cropping1D(CL/2)(skip)
        
        site_hidden = skip

        # Compute classification logits and loss
        logits_class = self.class_output_layer(site_hidden)
        probs_class = self.softmax(logits_class)
        outputs = (logits_class,)
        
        if not testing: # training
            return logits_class
            
        else:
            probs_class=torch.transpose(probs_class , 2, 1)
            return probs_class # (loss_class), logits_class, (site_hidden)

    
    def load_tf_model(self , module):
        if isinstance(module, (nn.Conv1d)) or isinstance(module , (DeformConv1d)):
        
            
            key_now = ( module.weight.shape[2] , module.weight.shape[1] , module.weight.shape[0]  ) # 1 , 32, 3
            if key_now not in self.dict_index.keys():
                self.all_key[key_now]=0
            
            if module.weight.shape in [ torch.Size([32,4,1]) ]:
                index_now = self.dict_index[key_now]
                vals = self.tf_dict[key_now][index_now]
                tt=torch.reshape(vals,  ( key_now[1] , key_now[2] , key_now[0]  )  )  # 32 , 3 , 1
                tt=torch.transpose(tt,1,0)  #   3 , 32 , 1
                module.weight = torch.nn.Parameter(tt)
                self.dict_index[key_now]+=1
                
                
            elif key_now in self.dict_index.keys():
                index_now = self.dict_index[key_now]
                vals = np.array(self.tf_dict[key_now][index_now])
                tt=vals.transpose(2, 1, 0)
                module.weight = torch.nn.Parameter(torch.Tensor(tt))
                self.dict_index[key_now]+=1
                
                
                
            else:
                index_now = self.need_index[key_now]
                vals = self.need_dict[key_now][index_now]
                tt=torch.reshape(vals,  ( key_now[1] , key_now[2] , key_now[0]  )  )  # 32 , 3 , 1
                tt=torch.transpose(tt,1,0)  #   3 , 32 , 1
                module.weight = torch.nn.Parameter(tt)
                self.need_index[key_now]+=1
    
            
    
    
    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        if self.tf_dict is not None:
            print('loading tf model')
            self.apply(self._init_weights)
            self.apply(self.load_tf_model)
            
            
        else:
            self.apply(self._init_weights)
        
        
        
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.xavier_normal_(module.weight , gain=torch.nn.init.calculate_gain('relu'))
            
    
            if isinstance(module, (nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
                
                
                
                
                
                


class SpliceAINew(nn.Module):
    def __init__(self, out_channels, kernel_sizes, dilation_rates, tf_dict=None):
        # L, out_channels: Number of convolution kernels, scalar
        # W, kernel_sizes: Convolution window size in each residual unit, array of int
        # AR, dilation_rates: Atrous rate in each residual unit, array of int
        super().__init__()
        
        assert len(kernel_sizes) == len(dilation_rates)


        self.dict_index = {(1, 4, 32): 0, (11, 32, 32): 0, (21, 32, 32): 0, (41, 32, 32): 0, (1, 32, 32): 0, (1, 32, 3): 0}
        self.tf_dict = tf_dict
        
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates

        self.CL = int(2 * np.sum(dilation_rates*(kernel_sizes-1)))
        self.conv = nn.Conv1d( 4 , out_channels, kernel_size=1)
        self.skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        residual_units = []
        dense_layers = []
        for i in range(len(kernel_sizes)):
            residual_unit = ResidualUnit(out_channels, kernel_sizes[i], dilation_rates[i])
            residual_units.append(residual_unit)

            if (((i+1) % 4 == 0) or ((i+1) == len(kernel_sizes))):
                dense_layer = nn.Conv1d(out_channels, out_channels, kernel_size=1)
                dense_layers.append(dense_layer)
                
        self.residual_units = nn.ModuleList(residual_units)
        self.dense_layers = nn.ModuleList(dense_layers)

        # Classification layer
        self.class_output_layer = nn.Conv1d(out_channels, 3, kernel_size=1) # 3 classes: acceptor, donor and none
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

        
    def forward(self, x , testing=None):
        x=torch.transpose(x , 2, 1)
        
        
        conv = self.conv(x)
        skip = self.skip(conv)
        
        dense_ind = 0
        for i in range(len(self.kernel_sizes)):
            residual_unit = self.residual_units[i]
            conv = residual_unit(conv)
            
            if (((i+1) % 4 == 0) or ((i+1) == len(self.kernel_sizes))):
                # Skip connections to the output after every 4 residual units
                dense_layer = self.dense_layers[dense_ind]
                dense = dense_layer(conv)
                skip = skip + dense
                dense_ind += 1

        # Remove flanking ends
        skip = skip[:, :, int(self.CL/2):int(-self.CL/2 )] # replaces Keras' Cropping1D(CL/2)(skip)
        
        site_hidden = skip

        # Compute classification logits and loss
        logits_class = self.class_output_layer(site_hidden)
        probs_class = self.softmax(logits_class)
        outputs = (logits_class,)
        
        if not testing: # training
            return logits_class
            
        else:
            probs_class=torch.transpose(probs_class , 2, 1)
            return probs_class # (loss_class), logits_class, (site_hidden)

    
    def load_tf_model(self , module):
        if isinstance(module, (nn.Conv1d)):
            
            if module.weight.shape in [ torch.Size([32,4,1]) ]:
                key_now = ( module.weight.shape[2] , module.weight.shape[1] , module.weight.shape[0]  ) # 1 , 32, 3
                index_now = self.dict_index[key_now]
                vals = self.tf_dict[key_now][index_now]
                tt=torch.reshape(vals,  ( key_now[1] , key_now[2] , key_now[0]  )  )  # 32 , 3 , 1
                tt=torch.transpose(tt,1,0)  #   3 , 32 , 1
                module.weight = torch.nn.Parameter(tt)
                self.dict_index[key_now]+=1
            
            elif module.weight.shape in [ torch.Size([3,32,1]) ,torch.Size([32,32,1]) , torch.Size([32,32,11]) ,  torch.Size([32,32,21]) ,torch.Size([32,32,41]) ]:
                key_now = ( module.weight.shape[2] , module.weight.shape[1] , module.weight.shape[0]  ) # 1 , 32, 3
                index_now = self.dict_index[key_now]
                vals = np.array(self.tf_dict[key_now][index_now])
                tt=vals.transpose(2, 1, 0)
                module.weight = torch.nn.Parameter(torch.Tensor(tt))
                self.dict_index[key_now]+=1

                
            else:
                print('excpep')
    
            
    
    
    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        if self.tf_dict is not None:
            print('loading tf model')
            self.apply(self._init_weights)
            self.apply(self.load_tf_model)
            
            
        else:
            self.apply(self._init_weights)
        
        
        
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.xavier_normal_(module.weight , gain=torch.nn.init.calculate_gain('relu'))
            
    
            if isinstance(module, (nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()


                
                
                
class SpliceAIIncep(nn.Module):
    def __init__(self, out_channels, kernel_sizes, dilation_rates, tf_dict=None):
        # L, out_channels: Number of convolution kernels, scalar
        # W, kernel_sizes: Convolution window size in each residual unit, array of int
        # AR, dilation_rates: Atrous rate in each residual unit, array of int
        super().__init__()
        
        assert len(kernel_sizes) == len(dilation_rates)


        self.dict_index = {(1, 4, 32): 0, (32,): 0, (1, 32, 16): 0, (16,): 0, (11, 16, 16): 0, (7, 16, 16): 0, (1, 32, 8): 0, (8,): 0, (11, 8, 8): 0, (15, 8, 8): 0, (7, 8, 8): 0, (9, 8, 8): 0, (21, 8, 8): 0, (25, 8, 8): 0, (19, 8, 8): 0, (23, 8, 8): 0, (41, 8, 8): 0, (51, 8, 8): 0, (37, 8, 8): 0, (45, 8, 8): 0, (1, 32, 32): 0, (1, 32, 3): 0, (3,): 0}
        self.tf_dict = tf_dict
        
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.dilation_rates = dilation_rates

        self.CL = int(2 * np.sum(dilation_rates*(kernel_sizes-1)))
        self.conv = nn.Conv1d( 4 , out_channels, kernel_size=1)
        self.skip = nn.Conv1d(out_channels, out_channels, kernel_size=1)

        convs_1 = []
        convs_2 = []
        convs_3 = []
        convs_4 = []
        convs_5 = []
        res_1 = []
        res_2 = []
        res_3 = []
        res_4 = []
        res_5 = []
        dense_layers = []
        
        ######### -------   1    ------- #########
        
        for _ in range(4):
            conv_a = nn.Conv1d(32 , 16, kernel_size=1)
            conv_b = nn.Conv1d(32 , 16, kernel_size=1)
            
            Res_a = ResidualUnit(16, 11, 1)
            Res_b = ResidualUnit(16, 7, 2)
            
            convs_1.append(conv_a)
            convs_1.append(conv_b)
            
            res_1.append(Res_a)
            res_1.append(Res_b)
        
        dense_layers.append( nn.Conv1d(out_channels, out_channels, kernel_size=1) )
        
        
        
        
        ######### -------   2    ------- #########
        
        
        for _ in range(4):
            conv_a = nn.Conv1d(32 , 8, kernel_size=1)
            conv_b = nn.Conv1d(32 , 8, kernel_size=1)
            conv_c = nn.Conv1d(32 , 8, kernel_size=1)
            conv_d = nn.Conv1d(32 , 8, kernel_size=1)
            
            Res_a = ResidualUnit(8, 11, 4)
            Res_b = ResidualUnit(8, 15, 3)
            Res_c = ResidualUnit(8, 7, 7)
            Res_d = ResidualUnit(8, 9, 5)
            
            convs_2.append(conv_a)
            convs_2.append(conv_b)
            convs_2.append(conv_c)
            convs_2.append(conv_d)
            
            res_2.append(Res_a)
            res_2.append(Res_b)
            res_2.append(Res_c)
            res_2.append(Res_d)
        
        dense_layers.append( nn.Conv1d(out_channels, out_channels, kernel_size=1) )
        
        
        
        ######### -------   3    ------- #########
        
        
        for _ in range(4):
            conv_a = nn.Conv1d(32 , 8, kernel_size=1)
            conv_b = nn.Conv1d(32 , 8, kernel_size=1)
            conv_c = nn.Conv1d(32 , 8, kernel_size=1)
            conv_d = nn.Conv1d(32 , 8, kernel_size=1)
            
            Res_a = ResidualUnit(8, 21, 10)
            Res_b = ResidualUnit(8, 25, 8)
            Res_c = ResidualUnit(8, 19, 11)
            Res_d = ResidualUnit(8, 23, 9)
            
            convs_3.append(conv_a)
            convs_3.append(conv_b)
            convs_3.append(conv_c)
            convs_3.append(conv_d)
            
            res_3.append(Res_a)
            res_3.append(Res_b)
            res_3.append(Res_c)
            res_3.append(Res_d)
        
        dense_layers.append( nn.Conv1d(out_channels, out_channels, kernel_size=1) )
        
        
        
        
        ######### -------   4    ------- #########
        
        
        for _ in range(4):
            conv_a = nn.Conv1d(32 , 8, kernel_size=1)
            conv_b = nn.Conv1d(32 , 8, kernel_size=1)
            conv_c = nn.Conv1d(32 , 8, kernel_size=1)
            conv_d = nn.Conv1d(32 , 8, kernel_size=1)
            
            Res_a = ResidualUnit(8, 41, 25)
            Res_b = ResidualUnit(8, 51, 20)
            Res_c = ResidualUnit(8, 37, 27)
            Res_d = ResidualUnit(8, 45, 23)
            
            convs_4.append(conv_a)
            convs_4.append(conv_b)
            convs_4.append(conv_c)
            convs_4.append(conv_d)
            
            res_4.append(Res_a)
            res_4.append(Res_b)
            res_4.append(Res_c)
            res_4.append(Res_d)
        
        dense_layers.append( nn.Conv1d(out_channels, out_channels, kernel_size=1) )
        
        
        
        ######### -------   4    ------- #########
        
        
        for _ in range(4):
            conv_a = nn.Conv1d(32 , 8, kernel_size=1)
            conv_b = nn.Conv1d(32 , 8, kernel_size=1)
            conv_c = nn.Conv1d(32 , 8, kernel_size=1)
            conv_d = nn.Conv1d(32 , 8, kernel_size=1)
            
            Res_a = ResidualUnit(8, 11, 1)
            Res_b = ResidualUnit(8, 11, 4)
            Res_c = ResidualUnit(8, 21, 10)
            Res_d = ResidualUnit(8, 41, 25)
            
            convs_5.append(conv_a)
            convs_5.append(conv_b)
            convs_5.append(conv_c)
            convs_5.append(conv_d)
            
            res_5.append(Res_a)
            res_5.append(Res_b)
            res_5.append(Res_c)
            res_5.append(Res_d)
        
        dense_layers.append( nn.Conv1d(out_channels, out_channels, kernel_size=1) )
        
        
                
        self.convs_1 = nn.ModuleList(convs_1)
        self.convs_2 = nn.ModuleList(convs_2)
        self.convs_3 = nn.ModuleList(convs_3)
        self.convs_4 = nn.ModuleList(convs_4)
        self.convs_5 = nn.ModuleList(convs_5)
        self.res_1 = nn.ModuleList(res_1)
        self.res_2 = nn.ModuleList(res_2)
        self.res_3 = nn.ModuleList(res_3)
        self.res_4 = nn.ModuleList(res_4)
        self.res_5 = nn.ModuleList(res_5)
        self.dense_layers = nn.ModuleList(dense_layers)
        
        
        
        # Classification layer
        self.class_output_layer = nn.Conv1d(out_channels, 3, kernel_size=1) # 3 classes: acceptor, donor and none
        self.softmax = nn.Softmax(dim=1)

        self.init_weights()

        
    def forward(self, x , testing=None):
        x=torch.transpose(x , 2, 1)
        
        
        conv = self.conv(x)
        skip = self.skip(conv)
        
        
        ######### -------   1    ------- #########
        
        for i in range(4):
            conv_a = self.convs_1[2*i + 0](conv)
            conv_b = self.convs_1[2*i + 1](conv)

            conv_a = self.res_1[2*i + 0](conv_a) 
            conv_b = self.res_1[2*i + 1](conv_b) 
            conv = torch.cat((conv_a,conv_b), 1)

        dense = self.dense_layers[0](conv)
        skip = skip + dense
    
        
        
        ######### -------   2    ------- #########
        
        for i in range(4):
            conv_a = self.convs_2[4*i + 0](conv)
            conv_b = self.convs_2[4*i + 1](conv)
            conv_c = self.convs_2[4*i + 2](conv)
            conv_d = self.convs_2[4*i + 3](conv)

            conv_a = self.res_2[4*i + 0](conv_a) 
            conv_b = self.res_2[4*i + 1](conv_b) 
            conv_c = self.res_2[4*i + 2](conv_c) 
            conv_d = self.res_2[4*i + 3](conv_d) 
            conv = torch.cat((conv_a,conv_b,conv_c,conv_d), 1)

        dense = self.dense_layers[1](conv)
        skip = skip + dense
        
        
        ######### -------   3    ------- #########
        
        for i in range(4):
            conv_a = self.convs_3[4*i + 0](conv)
            conv_b = self.convs_3[4*i + 1](conv)
            conv_c = self.convs_3[4*i + 2](conv)
            conv_d = self.convs_3[4*i + 3](conv)

            conv_a = self.res_3[4*i + 0](conv_a) 
            conv_b = self.res_3[4*i + 1](conv_b) 
            conv_c = self.res_3[4*i + 2](conv_c) 
            conv_d = self.res_3[4*i + 3](conv_d) 
            conv = torch.cat((conv_a,conv_b,conv_c,conv_d), 1)

        dense = self.dense_layers[2](conv)
        skip = skip + dense
        
        
        ######### -------   4    ------- #########
        
        for i in range(4):
            conv_a = self.convs_4[4*i + 0](conv)
            conv_b = self.convs_4[4*i + 1](conv)
            conv_c = self.convs_4[4*i + 2](conv)
            conv_d = self.convs_4[4*i + 3](conv)

            conv_a = self.res_4[4*i + 0](conv_a) 
            conv_b = self.res_4[4*i + 1](conv_b) 
            conv_c = self.res_4[4*i + 2](conv_c) 
            conv_d = self.res_4[4*i + 3](conv_d) 
            conv = torch.cat((conv_a,conv_b,conv_c,conv_d), 1)

        dense = self.dense_layers[3](conv)
        skip = skip + dense
        
        
        ######### -------   5    ------- #########
        
        for i in range(4):
            conv_a = self.convs_5[4*i + 0](conv)
            conv_b = self.convs_5[4*i + 1](conv)
            conv_c = self.convs_5[4*i + 2](conv)
            conv_d = self.convs_5[4*i + 3](conv)

            conv_a = self.res_5[4*i + 0](conv_a) 
            conv_b = self.res_5[4*i + 1](conv_b) 
            conv_c = self.res_5[4*i + 2](conv_c) 
            conv_d = self.res_5[4*i + 3](conv_d) 
            conv = torch.cat((conv_a,conv_b,conv_c,conv_d), 1)

        dense = self.dense_layers[4](conv)
        skip = skip + dense
        
        
        
        skip = skip[:, :, int(self.CL/2):int(-self.CL/2 )] # replaces Keras' Cropping1D(CL/2)(skip)
        
        site_hidden = skip

        # Compute classification logits and loss
        logits_class = self.class_output_layer(site_hidden)
        probs_class = self.softmax(logits_class)
        outputs = (logits_class,)
        
        if not testing: # training
            return logits_class
            
        else:
            probs_class=torch.transpose(probs_class , 2, 1)
            return probs_class # (loss_class), logits_class, (site_hidden)

    
    def load_tf_model(self , module):
        
        
        if isinstance(module, (nn.Conv1d)):
            
            
            key_now = ( module.weight.shape[2] , module.weight.shape[1] , module.weight.shape[0]  ) # 1 , 32, 3
            
                
            
            if module.weight.shape in [ torch.Size([32,4,1]) ]:
                index_now = self.dict_index[key_now]
                vals = self.tf_dict[key_now][index_now]
                tt=torch.reshape(vals,  ( key_now[1] , key_now[2] , key_now[0]  )  )  # 32 , 3 , 1
                tt=torch.transpose(tt,1,0)  #   3 , 32 , 1
                module.weight = torch.nn.Parameter(tt)
                self.dict_index[key_now]+=1

            elif key_now in self.dict_index.keys():
                index_now = self.dict_index[key_now]
                vals = np.array(self.tf_dict[key_now][index_now])
                tt=vals.transpose(2, 1, 0)
                module.weight = torch.nn.Parameter(torch.Tensor(tt))
                self.dict_index[key_now]+=1
                
            else:
                print('except key'+str(key_now))
        
        try:
            if module.bias is not None:
                key_now = (module.bias.shape[0],)
                index_now = self.dict_index[key_now]

                vals = self.tf_dict[key_now][index_now]
                module.bias = torch.nn.Parameter(vals)
                self.dict_index[key_now]+=1
        except:
            a=3

    
    def init_weights(self):
        """ Initialize weights if needed. """
        # Initialize weights
        if self.tf_dict is not None:
            print('loading tf model')
            self.apply(self._init_weights)
            self.apply(self.load_tf_model)
            
            
        else:
            self.apply(self._init_weights)
        
        
        
        
    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Conv1d)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            nn.init.xavier_normal_(module.weight , gain=torch.nn.init.calculate_gain('relu'))
            
    
            if isinstance(module, (nn.Conv1d)) and module.bias is not None:
                module.bias.data.zero_()
