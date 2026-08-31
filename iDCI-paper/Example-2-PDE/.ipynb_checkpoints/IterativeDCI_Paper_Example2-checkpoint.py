#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# The libraries we will use
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import norm, beta 
import scipy.io as sio
from scipy.stats import entropy
from tqdm import tqdm
import matplotlib as mpl


# In[ ]:

from ipywidgets import interact, IntSlider
import ipywidgets as widgets


# In[ ]:


def iterative_DCI(Q_pred_vals, observed_densities, num_epochs=1, r_init=1, 
                  pred_tol=1e-12, ratio_tol=0.1, KL_tol=1e-7, KL_update_rel_tol = 1e-2,
                  verbose=False, QoI_spaces=[]):
    '''
    Assume Q_pred_vals is a (num_samples x QoI_num) array of QoI vals 
    associated with initial parameter samples and observed_densities 
    is a list of observed QoI densities for each QoI map (coming from a
    data generating distribution)
    '''
    num_samples, num_QoI = np.shape(Q_pred_vals)

    if len(QoI_spaces) == 0:
        QoI_spaces = list(np.arange(num_QoI))
    
    true_obs_probs = []
    for d in range(len(QoI_spaces)):
        true_obs_probs.append(observed_densities[d](Q_pred_vals[:,QoI_spaces[d]].T))
    
    with tqdm(total=100) as pbar:    
        
        if np.size(r_init)==1:  # No initial weighting (possibly from prior iterations)
            rs = np.ones((num_samples, len(QoI_spaces), num_epochs))
            r_current = np.ones(num_samples)
        else:
            r_current = r_init
            rs = np.ones((num_samples, len(QoI_spaces), num_epochs))
            
        kl_from_observed_marginal = np.zeros((len(QoI_spaces), num_epochs))
        
        for k in range(num_epochs):
            
            if verbose is True:
                print('Starting epoch #', k)
            
            for d in range(len(QoI_spaces)):
                r_current /= np.mean(r_current)
                if verbose is True: 
                    print('Iteration #', d)
                    print('Some current r-values:', r_current[0:3])
                pred_dens = GKDE(Q_pred_vals[:,QoI_spaces[d]].T,
                                 weights=r_current)
                rs_update = np.where(pred_dens(Q_pred_vals[:,QoI_spaces[d]].T)>pred_tol,
                                        np.divide(observed_densities[d](Q_pred_vals[:,QoI_spaces[d]].T),
                                        pred_dens(Q_pred_vals[:,QoI_spaces[d]].T)), 0)
                r_current *= rs_update
                rs[:,d,k] = r_current
                if np.abs(np.mean(r_current)-1.0)>ratio_tol:
                    print('Outside of ratio tolerance at epoch #', k+1, ' iteration #', d+1)
                    return rs[:,:d+1,:k+1], k+1, d+1, kl_from_observed_marginal[:,k]
                pbar.update(100/(len(QoI_spaces)*num_epochs))
            
            for d in range(len(QoI_spaces)):
                pred_dens = GKDE(Q_pred_vals[:, QoI_spaces[d]].T, 
                                 weights=r_current)
                current_probs = pred_dens(Q_pred_vals[:,QoI_spaces[d]].T)
                kl_from_observed_marginal[d, k] = entropy(true_obs_probs[d], current_probs) / num_samples
            
            if verbose is True:
                print('Epoch #', k)
                print('KL divergences from observed marginals', kl_from_observed_marginal[:,k])
            
            if all( kl < KL_tol for kl in kl_from_observed_marginal[:,k]):
                print('KL divergences from observed marginals all within tolerance at epoch #', k+1)
                return rs[:,:,:k+1], k+1, d+1, kl_from_observed_marginal
            
            if k>0:
                kl_relative_update = np.abs(kl_from_observed_marginal[:,k-1] - kl_from_observed_marginal[:,k]) /\
                                    kl_from_observed_marginal[:,k-1]
                if all(kl_rel < KL_update_rel_tol for kl_rel in kl_relative_update):
                    print('KL divergences from observed marginals are not sufficiently updated at epoch #', k+1)
                    return rs[:,:,:k+1], k+1, d+1, kl_from_observed_marginal
    return rs, k+1, d+1, kl_from_observed_marginal


# In[ ]:


# Initial and predicted samples
# Assume the data files are named as follows: initial_inputs.dat and predicted_output.dat

num_init_samples = int(1e4)

params_init = np.loadtxt('initial_input.dat') # param samples, samples x dim

params_init = params_init[0:num_init_samples,:]

param_dim = params_init.shape[1]

Q_pred_vals = np.loadtxt('predicted_output.dat') # QoI samples, samples x dim

Q_pred_vals = Q_pred_vals[0:num_init_samples,:]

# Data generating and observed samples
# Assume the data files are named as follows: datagen_input.dat and datagen_output.dat

num_dg_samples = int(1e4)

params_dg = np.loadtxt('datagen_input.dat')

params_dg = params_dg[0:num_dg_samples,:]

Q_obs_vals = np.loadtxt('datagen_output.dat')

Q_obs_vals = Q_obs_vals[0:num_dg_samples,:]


# In[ ]:


Q_coordinates = np.loadtxt('sensor_points.dat')


# In[ ]:


which_QoI = []
for i in range(8):  # First 8 locations for pressure
    which_QoI.append(i)

for i in range(1000,1004):  # First 4 locations also give velocity
    which_QoI.append(i)


# In[ ]:


fig = plt.figure(num=0)
fig.clf()

num_QoI = len(which_QoI)

for i in which_QoI:
    if i >= 1000:
        plt.scatter(Q_coordinates[i-1000,0], Q_coordinates[i-1000,1])
        plt.annotate(r', $v$', (Q_coordinates[i-1000,0], Q_coordinates[i-1000,1]), textcoords="offset points", 
                     xytext=(5, -15), ha='center', fontsize=12)
    else:
        plt.scatter(Q_coordinates[i,0], Q_coordinates[i,1])
        plt.annotate(r'$p$', (Q_coordinates[i,0], Q_coordinates[i,1]), textcoords="offset points", 
                     xytext=(-5, -15), ha='center', fontsize=12)
plt.title('Pressure ($p$) and Velocity ($v$) Sensor Locations')
plt.show()


# In[ ]:


# Apply a standard scaler to the QoI data

from sklearn.preprocessing import StandardScaler

scaler_predict = StandardScaler()

scaler_predict.fit(Q_pred_vals)

Q_pred_vals_scaled = scaler_predict.transform(Q_pred_vals)

Q_obs_vals_scaled = scaler_predict.transform(Q_obs_vals)


# In[ ]:


num_epochs = 50

QoI_spaces = [0, 1, 2, 3, 4, 5, 6, 7, 1000, 1001, 1002, 1003]

print('QoI_spaces are ', QoI_spaces)

observed_densities = []
for d in range(len(QoI_spaces)):
    observed_densities.append(GKDE(Q_obs_vals_scaled[:,QoI_spaces[d]].T))

rs, last_epoch, last_iter, kl_epochs = iterative_DCI(Q_pred_vals_scaled, observed_densities, 
                                           num_epochs = num_epochs, r_init = 1, ratio_tol=0.1,
                                                    QoI_spaces = QoI_spaces, KL_tol=1e-6) 


# In[ ]:


# Only use this if all QoI spaces are 1-d

def plot_Qs(fignum, iter_num):

    fig = plt.figure(num=fignum, figsize=(10, 6))
    fig.clf()

    num_QoI = len(QoI_spaces)

    gs = fig.add_gridspec(2*int(num_QoI/3),6)

    row = -1
    for d in range(num_QoI):
        if (d)%3 == 0:
            row += 1
        ax = fig.add_subplot(gs[0+2*row:2+2*row:,0+2*(d%3):2+2*(d%3)])
        pred_dens = GKDE(Q_pred_vals_scaled[:,QoI_spaces[d]], weights=rs[:, (iter_num-1) % num_QoI, (iter_num-1) // num_QoI])
        plt_Qs = np.linspace(-2.5, 2.5, 101)
        ax.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', label='PF of Update')
        ax.plot(plt_Qs, observed_densities[d](plt_Qs), label='Observed')
        ax.legend()
        if d == 1:
            title_str = 'Epoch: ' + str((iter_num-1) // num_QoI +1) +\
                        ', Iter: ' + str((iter_num-1) % num_QoI + 1)
            ax.set_title(title_str)
    plt.tight_layout()
    plt.show()


# In[ ]:


fignum = 1

total_iters = len(QoI_spaces) * (last_epoch-1) + last_iter-1

interact(plot_Qs, 
         fignum = widgets.fixed(fignum),
         iter_num = IntSlider(min=1, max=total_iters+1, steps=1))

total_iters = len(QoI_spaces) * (last_epoch-1) + last_iter-1

interact(plot_Qs, 
         fignum = widgets.fixed(fignum),
         iter_num = IntSlider(min=1, max=total_iters+1, steps=1, value=12))

interact(plot_Qs, 
         fignum = widgets.fixed(fignum),
         iter_num = IntSlider(min=1, max=total_iters+1, steps=1, value=36))


# In[ ]:


plt.figure(num=2)
plt.clf()
params_to_plot = 16
param = np.arange(params_to_plot)+1

plt.barh(param, np.average(params_dg[:,0:params_to_plot], axis=0), color='blue', alpha=0.5)

epoch_num = 1
iter_num = 1

plt.barh(param, np.average(params_init[:,0:params_to_plot], axis=0, 
                           weights=rs[:,iter_num-1,epoch_num-1]), color='black')

plt.xlim([-0.3,0.3])
plt.xlabel("Mean of updated marginals")
plt.ylabel("KL Mode")
plt.title('Epoch #' + str(epoch_num) + ', Iteration #' + str(iter_num))
plt.show()


# In[ ]:


plt.figure(num=3)
plt.clf()

plt.barh(param, np.average(params_dg[:,0:params_to_plot], axis=0), color='blue', alpha=0.5)

epoch_num = 1
iter_num = 6

plt.barh(param, np.average(params_init[:,0:params_to_plot], axis=0, 
                           weights=rs[:,iter_num-1,epoch_num-1]), color='black')

plt.xlim([-0.3,0.3])
plt.xlabel("Mean of updated marginals")
plt.ylabel("KL Mode")
plt.title('Epoch #' + str(epoch_num) + ', Iteration #' + str(iter_num))
plt.show()


# In[ ]:


plt.figure(num=4)
plt.clf()

plt.barh(param, np.average(params_dg[:,0:params_to_plot], axis=0), color='blue', alpha=0.5)

epoch_num = 1
iter_num = 12

plt.barh(param, np.average(params_init[:,0:params_to_plot], axis=0, 
                           weights=rs[:,iter_num-1,epoch_num-1]), color='black')

plt.xlim([-0.3,0.3])
plt.xlabel("Mean of updated marginals")
plt.ylabel("KL Mode")
plt.title('Epoch #' + str(epoch_num) + ', Iteration #' + str(iter_num))
plt.show()


# In[ ]:


plt.figure(num=5)
plt.clf()

plt.barh(param, np.average(params_dg[:,0:params_to_plot], axis=0), color='blue', alpha=0.5)

epoch_num = last_epoch
iter_num = last_iter

plt.barh(param, np.average(params_init[:,0:params_to_plot], axis=0, 
                           weights=rs[:,iter_num-1,epoch_num-1]), color='black')

plt.xlim([-0.3,0.3])
plt.xlabel("Mean of updated marginals")
plt.ylabel("KL Mode")
plt.title('Epoch #' + str(epoch_num) + ', Iteration #' + str(iter_num))
plt.show()


# In[ ]:


total_iters = len(QoI_spaces) * (last_epoch-1) + last_iter-1
updated_means = np.zeros((param_dim, total_iters))

epoch_count = 0
for i in range(total_iters):
    if i % num_QoI == 0:
        epoch_count += 1
    current_rs = rs[:, i % len(QoI_spaces), epoch_count-1]
    updated_means[:, i] = np.average(params_init, axis=0, weights=current_rs)


# In[ ]:


# If all QoI used individually

plt.figure(num=6)
plt.clf()
for i in range(5):
    plt.plot(np.arange(1, total_iters+1), updated_means[i,:].T) #, label='KL Mode #' + str(i+1))
    if i != 4:
        plt.annotate('KL Mode #' + str(i+1), 
                     xy=(int(total_iters*0.8**(i+1)), updated_means[i, int(total_iters*0.8**(i+1))]), 
                     xytext=(int(total_iters*0.8**(i+1)) + 1, updated_means[i, int(total_iters*0.8**(i+1))]+0.2),
                     arrowprops=dict(facecolor='black', shrink=0.005),
                     horizontalalignment='left', verticalalignment='bottom')
    else:
        plt.annotate('KL Mode #' + str(i+1), 
                     xy=(int(total_iters*0.8**(3)), updated_means[i, int(total_iters*0.8**(3))]), 
                     xytext=(int(total_iters*0.8**(3)) - 1, updated_means[i, int(total_iters*0.8**(3))]-0.25),
                     arrowprops=dict(facecolor='black', shrink=0.005),
                     horizontalalignment='right', verticalalignment='bottom')

plt.title('Means of KL modes')
plt.xlabel('iteration')
plt.ylim([-.75, .75])
plt.show()


# In[ ]:


# Now use joint pressure & velocity data at 4 locations that provide both

QoI_spaces = [[0, 1000], [1, 1001], [2, 1002], [3, 1003], 4, 5, 6, 7] 

print('QoI_spaces are ', QoI_spaces)

observed_densities = []
for d in range(len(QoI_spaces)):
    observed_densities.append(GKDE(Q_obs_vals_scaled[:,QoI_spaces[d]].T))

rs, last_epoch, last_iter, kl_epochs = iterative_DCI(Q_pred_vals_scaled, observed_densities, 
                                           num_epochs = num_epochs, r_init = 1, ratio_tol=0.1,
                                                    QoI_spaces = QoI_spaces, KL_tol=1e-6) 


# In[ ]:


plt.figure(num=7)
plt.clf()

plt.barh(param, np.average(params_dg[:,0:params_to_plot], axis=0), color='blue', alpha=0.5)

epoch_num = last_epoch
iter_num = last_iter

plt.barh(param, np.average(params_init[:,0:params_to_plot], axis=0, 
                           weights=rs[:,iter_num-1,epoch_num-1]), color='black')

plt.xlim([-0.3,0.3])
plt.xlabel("Mean of updated marginals")
plt.ylabel("KL Mode")
plt.title('Epoch #' + str(epoch_num) + ', Iteration #' + str(iter_num))
plt.show()


# In[ ]:


total_iters = len(QoI_spaces) * (epoch_num-1) + last_iter-1
updated_means = np.zeros((param_dim, total_iters))

epoch_count = 0
for i in range(total_iters):
    if i % num_QoI == 0:
        epoch_count += 1
    current_rs = rs[:, i % len(QoI_spaces), epoch_count-1]
    updated_means[:, i] = np.average(params_init, axis=0, weights=current_rs)


# In[ ]:


# If joint pressure and velocity data are used

plt.figure(num=8)
plt.clf()
for i in range(5):
    plt.plot(np.arange(1, total_iters+1), updated_means[i,:].T) #, label='KL Mode #' + str(i+1))
    if i != 4:
        plt.annotate('KL Mode #' + str(i+1), 
                     xy=(int(total_iters*0.8**(i+1)), updated_means[i, int(total_iters*0.8**(i+1))]), 
                     xytext=(int(total_iters*0.8**(i+1)) + 10, updated_means[i, int(total_iters*0.8**(i+1))]+0.2),
                     arrowprops=dict(facecolor='black', shrink=0.005),
                     horizontalalignment='left', verticalalignment='bottom')
    else:
        plt.annotate('KL Mode #' + str(i+1), 
                     xy=(int(total_iters*0.8**(i+1)), updated_means[i, int(total_iters*0.8**(i+1))]), 
                     xytext=(int(total_iters*0.8**(i+1)) - 10, updated_means[i, int(total_iters*0.8**(i+1))]-0.25),
                     arrowprops=dict(facecolor='black', shrink=0.005),
                     horizontalalignment='right', verticalalignment='bottom')
# plt.legend(loc='center')
plt.title('Means of KL modes')
plt.xlabel('iteration')
plt.ylim([-.75, .75])
plt.show()

