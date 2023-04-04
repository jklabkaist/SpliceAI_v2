import h5py
import time
import random
import torch
from spliceai_module_torch import *
from sklearn.metrics import average_precision_score
from keras.models import load_model

import numpy as np
import pickle
import sys
import time
import h5py

from torch.utils.data import TensorDataset, DataLoader

folder_name = sys.argv[1].split('/')[0]



print('GPU is')
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

kernel_sizes = np.asarray([11, 11, 11, 11, 11, 11, 11, 11,
                        21, 21, 21, 21, 41, 41, 41, 41])
dilation_rates = np.asarray([1, 1, 1, 1, 4, 4, 4, 4,
                        10, 10, 10, 10, 25, 25, 25, 25])





# w_key=load_obj('incep_baseline_weight')
# model = SpliceAIIncep(32, kernel_sizes, dilation_rates , w_key).cuda()
# w_key=''

model = SpliceAIIncep(32, kernel_sizes, dilation_rates)  ########## when loading
model.load_state_dict(torch.load('0314_fix_incep/save_model_12'))
model = model.cuda() ########## when loading






optimizer = torch.optim.Adam(model.parameters(), lr=0.00002) ############ CHANGED RL !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


print('param # ')
print(get_n_params(model))



filename = '../sp_test/gtex_10000_train_dataset.h5'
h5f = h5py.File(filename, 'r')


num_idx = len(h5f.keys())//2
idx_all = np.random.permutation(num_idx)

idx_train = [57, 112, 111, 26, 42, 82, 124, 109, 93, 0, 60, 27, 132, 123, 28, 14, 43, 68, 117, 77, 100, 128, 29, 80, 18, 75, 101, 17, 74, 86, 76, 118, 130, 5, 120, 131, 9, 13, 33, 70, 88, 91, 54, 10, 64, 20, 41, 106, 81, 69, 61, 110, 126, 24, 34, 63, 50, 7, 85, 98, 23, 103, 121, 114, 89, 37, 15, 127, 83, 31, 1, 59, 113, 6, 125, 78, 67, 44, 97, 95, 94, 36, 79, 35, 19, 115, 129, 25, 39, 12, 108, 48, 122, 4, 46, 11, 71, 22, 104, 102, 30, 51, 38, 49, 3, 65, 53, 92, 32, 40, 52, 58, 107, 16, 45, 73, 21, 87, 105]
idx_valid = []


for i in range(133):
    if i not in idx_train:
        idx_valid.append(i)
        
        
print()
print(idx_train)
print()
print(idx_valid)
print()


start_time = time.time()


EPOCH_NUM = 15
TRAIN_BATCH_SIZE=18

print('batch size : '+str(TRAIN_BATCH_SIZE))

start_time = time.time()

cc=0


loss_total=0.0


for cc in range(15,EPOCH_NUM+15):
        
    all_li = idx_train.copy()
    random.shuffle(all_li)
        
    for j,idx in enumerate(all_li):
        
#         if cc==0 and j==15:
#             break
        
        X = h5f[f'X{idx}'][:]
        Y = h5f[f'Y{idx}'][0, ...]


        ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y).float())
        loader = DataLoader(ds, batch_size=TRAIN_BATCH_SIZE, shuffle=True, drop_last=True, num_workers=8, pin_memory=True)  # TODO: Check whether 
        
        model.train()
        
        for batch in loader:

            X_input, Y_input = batch[0].cuda(), batch[1].cuda()
            optimizer.zero_grad()
            Y_pred = model.forward(X_input , testing=True)
            loss = custom_CE_loss(Y_pred , Y_input)

            loss.backward() ; optimizer.step()
            loss_total+= float(loss)
    
    
    
    
    
#     if epoch_num==15 or ((epoch_num+1) % len(idx_train) == 0):
    if True:
        
        
        
        
#         torch.save(model.state_dict(), 'test_spliceai_torch'+"/save_model_"+str(cc)  )
        torch.save(model.state_dict(), folder_name+"/small/save_model_"+str(cc)  )
        print('\n------------------------------\n')
        print('Done global step %i' %(cc+1           )) 
        print("--- %s seconds ---" % (time.time() - start_time))
        start_time = time.time()
        
        print('\ntotal loss : %.3f' %(loss_total))
        loss_total=0.0
        print("learning rate : %f" %(  optimizer.param_groups[0]['lr']  ))
        print('------------------------------\n')
        
        
        model.eval()
        with torch.no_grad():
        
            print("\nValidation set metrics:")

            Y_true_1 = [[] for t in range(1)]
            Y_true_2 = [[] for t in range(1)]
            Y_pred_1 = [[] for t in range(1)]
            Y_pred_2 = [[] for t in range(1)]

            for idx in idx_valid:

                X = h5f['X' + str(idx)][:]
                Y = h5f['Y' + str(idx)][:]
                X,Y = get_datapoint(X,Y)


                batch_num = X.shape[0]
                pq=[uuu for uuu in range(0,batch_num,TRAIN_BATCH_SIZE)]

                for ii in range(len(pq)):
                    start_index =  pq[ii]  
                    end_index = pq[ii] + TRAIN_BATCH_SIZE

                    X_input = X[start_index:end_index]
                    Y_input = Y[start_index:end_index]

                    X_input=torch.Tensor(X_input).cuda()
                    Y_input=torch.Tensor(Y_input).cuda()


                    with torch.no_grad():
                        Y_pred = model.forward(X_input, testing = True)

                    Y_input = Y_input.cpu().numpy()
                    Y_pred = Y_pred.cpu().numpy()


                    for t in range(1):
                        is_expr = (np.asarray(Y_input).sum(axis=(1,2)) >= 1)
                        Y_true_1[t].extend(np.asarray(Y_input)[is_expr, :, 1].flatten())
                        Y_pred_1[t].extend( np.asarray(Y_pred)[is_expr, :, 1].flatten())

                        Y_true_2[t].extend(np.asarray(Y_input)[is_expr, :, 2].flatten())
                        Y_pred_2[t].extend( np.asarray(Y_pred)[is_expr, :, 2].flatten())



            print("\nAcceptor:")
            for t in range(1):
                txx=print_topl_statistics(np.asarray(Y_true_1[t]),
                                      np.asarray(Y_pred_1[t]))

            if txx<0.01:
                break


            print("\nDonor:")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_2[t]),
                                      np.asarray(Y_pred_2[t]))






            print("\nTraining set metrics:")

            Y_true_1 = [[] for t in range(1)]
            Y_true_2 = [[] for t in range(1)]
            Y_pred_1 = [[] for t in range(1)]
            Y_pred_2 = [[] for t in range(1)]

            for idx in idx_train[:len(idx_valid)]:

                X = h5f['X' + str(idx)][:]
                Y = h5f['Y' + str(idx)][:]
                X,Y = get_datapoint(X,Y)


                batch_num = X.shape[0]
                pq=[uuu for uuu in range(0,batch_num,TRAIN_BATCH_SIZE)]

                for ii in range(len(pq)):
                    start_index =  pq[ii]  
                    end_index = pq[ii] + TRAIN_BATCH_SIZE

                    X_input = X[start_index:end_index]
                    Y_input = Y[start_index:end_index]

                    X_input=torch.Tensor(X_input).cuda()
                    Y_input=torch.Tensor(Y_input).cuda()


                    with torch.no_grad():
                        Y_pred = model.forward(X_input, testing = True)

                    Y_input = Y_input.cpu().numpy()
                    Y_pred = Y_pred.cpu().numpy()


                    for t in range(1):
                        is_expr = (np.asarray(Y_input).sum(axis=(1,2)) >= 1)
                        Y_true_1[t].extend(np.asarray(Y_input)[is_expr, :, 1].flatten())
                        Y_pred_1[t].extend( np.asarray(Y_pred)[is_expr, :, 1].flatten())

                        Y_true_2[t].extend(np.asarray(Y_input)[is_expr, :, 2].flatten())
                        Y_pred_2[t].extend( np.asarray(Y_pred)[is_expr, :, 2].flatten())



            print("\nAcceptor:")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_1[t]),
                                      np.asarray(Y_pred_1[t]))

            print("\nDonor:")
            for t in range(1):
                print_topl_statistics(np.asarray(Y_true_2[t]),
                                      np.asarray(Y_pred_2[t]))
    
    
    
        if cc >= 8 and cc<13:
            now_lr = optimizer.param_groups[0]['lr']

            optimizer.param_groups[0]['lr'] = now_lr*0.5
    
    

h5f.close()
    
    
