import numpy as np
import pickle
import os, glob
from scipy.interpolate import RegularGridInterpolator
import sys


def make_pickle(path, file,filetype, data):
    a = {}
    a[filetype] =  data
    c = dict(a)
    with open(os.path.join(path,file), "wb") as json_file:
        pickle.dump(c, json_file)
        json_file.close()
    return c



def load_pickle(path, file):
    file_dict = open(os.path.join(path,file),"rb")
    output = pickle.load(file_dict)
    return dict(output)


def update_pickle(path, file,filetype, data):
    old_data = load_pickle(path, file)
    old_data[filetype] = data
    with open(os.path.join(path,file), "wb") as json_file:
        pickle.dump(old_data, json_file)
        json_file.close()

def check_file(path, file):
    return os.path.isfile(os.path.join(path,file))

def runSaving(path,file,filetype,data):
    if check_file(path, file):
        update_pickle(path, file,filetype, data)
    else:
        make_pickle(path, file,filetype, data)



def BLUE(fantom, lstm, real, weight_input, relax_input,relaxation_alpha=0.3):
    fantom = fantom.reshape(fantom.shape[0], 1)
    lstm = lstm.reshape(fantom.shape[0], 1)
    real = real.reshape(fantom.shape[0], 1)
    X = np.hstack([fantom, lstm])
    Xt = X.T
    XtX = np.dot(Xt, X)

    XtX_inv = np.linalg.inv(XtX)
    Xty = np.dot(Xt, real)
    w = np.dot(XtX_inv, Xty)
    if weight_input.all() == 0:
        y_combined = np.dot(X, w)
        relaxation = relaxation_alpha * (real - y_combined)
    else:
        y_combined = np.dot(X, weight_input)
        relaxation = relaxation_alpha * (real - y_combined)
        y_combined = np.dot(X, weight_input) + relax_input
#############
    #w = np.array([[0.5],[0.5]])
    print(w.shape)

    return w,y_combined, relaxation



def combineDLFANTOM_updated(Fantom, DeepLearning, Realdata, step, counter,nameweight, namerelax, begin, end, relaxation_alpha=0.3,weight_updates = True, saveweights = True, nudge = False):

    minshape = min(Realdata.shape[1], Fantom.shape[1], DeepLearning.shape[1])
    nameweight = f"{nameweight}_{str(begin)}_{str(end)}_step_{step}.pkl"
    namerelax = f"{namerelax}_{str(begin)}_{str(end)}_step_{step}.pkl"
    ls_depth = [i for i in range(0,DeepLearning.shape[0]+1,step)]
    ls_depth[-1] = ls_depth[-1]+DeepLearning.shape[0]%step
    com_depth = [[ls_depth[i], ls_depth[i+1]] for i in range(len(ls_depth)-1)]
    Fantom_DeepLearning_all = np.zeros(DeepLearning.shape)
    weight = {}
    weight[str(counter)] = {}
    relax = {}
    relax[str(counter)] = {}
    for j in com_depth:
        depth_start = j[0]
        depth_end = j[1]
        weight[str(counter)][str(depth_start)+'_'+str(depth_end)] = {}
        weight[str(counter)][str(depth_start)+'_'+str(depth_end)]['input'] = np.zeros([2,1])
        weight[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'] = np.zeros([2,1])
        relax[str(counter)][str(depth_start)+'_'+str(depth_end)] = {}
        relax[str(counter)][str(depth_start)+'_'+str(depth_end)]['input'] = np.zeros([depth_end-depth_start,1])
        relax[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'] = np.zeros([depth_end-depth_start,1])


    if nudge:
        weight = load_pickle(".", nameweight)
        relax = load_pickle(".", namerelax)
        weight_updates = False
        saveweights = False

    for i in range(minshape):
        for j in com_depth:
            depth_start = j[0]
            depth_end = j[1]
            weight_new, Fantom_DeepLearning_com, relax_new = BLUE(Fantom[:,i][depth_start:depth_end],
                      DeepLearning[:,i][depth_start:depth_end],
                      Realdata[:,i][depth_start:depth_end],
                      weight[str(counter)][str(depth_start)+'_'+str(depth_end)]['input'],
                      relax[str(counter)][str(depth_start)+'_'+str(depth_end)]['input'],
                                                       relaxation_alpha = relaxation_alpha)
            if weight_updates:
                weight[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'] = np.hstack([weight[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'], weight_new])
                weight[str(counter)][str(depth_start)+'_'+str(depth_end)]['input'] = weight[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'][:,1:].mean(axis=1).reshape([2,1])
                relax[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'] = np.hstack([relax[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'], relax_new])
                relax[str(counter)][str(depth_start)+'_'+str(depth_end)]['input'] = relax[str(counter)][str(depth_start)+'_'+str(depth_end)]['avg'][:,1:].mean(axis=1).reshape([-1,1])

            Fantom_DeepLearning_all[depth_start:depth_end,i] = Fantom_DeepLearning_com.flatten()
        for k in range(len(Fantom_DeepLearning_all[:,i])-1):
            if Fantom_DeepLearning_all[:,i][k] <= Fantom_DeepLearning_all[:,i][k+1]:
                Fantom_DeepLearning_all[:,i][k+1] = Fantom_DeepLearning_all[:,i][k]
    if saveweights:
        runSaving('.',nameweight,str(counter), weight[str(counter)])
        runSaving('.',namerelax,str(counter), relax[str(counter)])
    return Fantom_DeepLearning_all

if __name__=='__main__':
    dt = int(20)
    counter = int(119*24*3600/dt)
    step = int(7)
    end = int(counter)
    begin = int(counter) - int(step*24*3600/dt)
    nudge = False
    saveweights = True
    weight_updates = True

    data_fetch = np.loadtxt(f'temperature_{begin}_{end}_data_fetch.dat')
    LSTM_output = np.loadtxt(f'temperature_{begin}_{end}_LSTM_output.dat')
    fantom_output = np.loadtxt(f'temperature_{begin}_{end}_fantom_output.dat')
    if nudge:
        timebegin = begin - int(step*24*3600/dt)
        timeend = end - int(step*24*3600/dt)
        counter = begin
    else:
        timebegin = begin
        timeend = end
        counter = counter

    DeepLearning_Fantom = combineDLFANTOM_updated(fantom_output, 
                                                  LSTM_output, 
                                                  data_fetch, 
                                                  step=60,
                                                  counter = counter,
                                                  nameweight="weights",
                                                  namerelax="relax",
                                                  begin = timebegin,
                                                  end = timeend,
                                                  relaxation_alpha = 0.3,
                                                  weight_updates = weight_updates, 
                                                  saveweights = saveweights,
                                                  nudge = nudge) 
    np.savetxt(f"temperature_{begin}_{end}_DeepLearning_Fantom.dat", DeepLearning_Fantom, fmt="%1.3f")

