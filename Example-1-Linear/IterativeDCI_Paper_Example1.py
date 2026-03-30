#!/usr/bin/env python
# coding: utf-8

# In[1]:


# The libraries we will use
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde as GKDE
from scipy.stats import norm, beta 
import scipy.io as sio
from scipy.stats import entropy
from tqdm import tqdm


# In[2]:


from ipywidgets import interact, IntSlider
import ipywidgets as widgets


# In[3]:


# Not likely that we use since we prefer weighted KDEs, but maybe we will
def rejection_sampling(r):
    # Perform accept/reject sampling on a set of proposal samples using
    # the weights r associated with the set of samples and return
    # the indices idx of the proposal sample set that are accepted.
    N = r.size # size of proposal sample set
    check = np.random.uniform(low=0,high=1,size=N) # create random uniform weights to check r against
    M = np.max(r)
    new_r = r/M # normalize weights 
    idx = np.where(new_r>=check)[0] # rejection criterion
    return idx


# In[4]:


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


# In[5]:


# Initialize random seed for reproducibility
np.random.seed(121212)

num_init_samples = int(1e3)
param_dim = 2

gaussian_example=False  # False for example used in paper

if gaussian_example:
    params_init = np.random.multivariate_normal(mean=(0,0), cov=np.array([[2, -1],[-1, 2]]), size=(num_init_samples))
else:
    params_init = np.random.uniform(0, 1, (num_init_samples, param_dim))
    


# In[6]:


# For plotting density estimates on parameter space
num_points = 100

if gaussian_example:
    x = np.linspace(-3, 3, num_points)
    y = np.linspace(-3, 3, num_points)
    
else:
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    
X, Y = np.meshgrid(x, y)
plotting_params = np.vstack([X.ravel(), Y.ravel()]).T


# In[7]:


num_dg_samples = int(1e3)

if gaussian_example:
    params_dg = np.random.normal(loc=(.75,1), scale=0.25, size=(num_dg_samples, param_dim))
else:
    Beta1 = beta(3, 9)
    Beta2 = beta(8, 2)

    params_dg_1 = Beta1.rvs(size=num_dg_samples)
    params_dg_2 = Beta2.rvs(size=num_dg_samples)
    params_dg = np.vstack([params_dg_1, params_dg_2]).T


# In[8]:


plt.figure(num=1, figsize=(3,3))
plt.clf()
plt.scatter(params_init[:,0], params_init[:,1], label='Initial')
plt.scatter(params_dg[:,0], params_dg[:,1], marker='+',label='DG')
plt.xlabel(r'$\lambda_1$')
plt.ylabel(r'$\lambda_2$')
plt.legend()
plt.tight_layout(pad=.25)
plt.show()


# In[9]:


# For first part of the example

# 2D linear map
Q_maps = [lambda x: 2*x[:,0] + x[:,1], lambda x: 2.5*x[:,0]+0.5*x[:,1]]


# In[10]:


num_QoI = len(Q_maps)
Q_pred_vals = np.zeros((num_init_samples, num_QoI))
Q_obs_vals = np.zeros((num_dg_samples, num_QoI))

for d in range(num_QoI):
    Q_pred_vals[:, d] = Q_maps[d](params_init)
    Q_obs_vals[:, d] = Q_maps[d](params_dg)

x = np.linspace(Q_pred_vals[:,0].min(), Q_pred_vals[:,0].max(), num_points)
y = np.linspace(Q_pred_vals[:,1].min(), Q_pred_vals[:,1].max(), num_points)
    
X, Y = np.meshgrid(x, y)
plotting_QoI = np.vstack([X.ravel(), Y.ravel()]).T


# In[11]:


fig = plt.figure(num=2, figsize=(3,3))
fig.clf()
plt.scatter(Q_pred_vals[:,0], Q_pred_vals[:,1], label='Predicted')
plt.scatter(Q_obs_vals[:,0], Q_obs_vals[:,1], marker='+', label='Observed')
plt.xlabel(r'$Q_1$')
plt.ylabel(r'$Q_2$')
plt.legend()
plt.tight_layout(pad=.25)
plt.show()


# In[12]:


observed_densities = []
for d in range(num_QoI):
    observed_densities.append(GKDE(Q_obs_vals[:,d]))


# In[13]:


num_epochs = 100

rs, last_epoch, last_iter, kl_epochs = iterative_DCI(Q_pred_vals, observed_densities, 
                                           num_epochs = num_epochs, r_init = 1, ratio_tol=0.1)


# In[14]:


print(np.mean(rs[:,1,:], axis=0))  # looking at diagnostic at end of each epoch


# In[15]:


def plot_epoch(fignum, epoch_num, observed_densities, fix_cbar_ranges=False):

    # Construct updated densities for particular iterations of epoch
    update_dens_iter_1 = GKDE(params_init.T,
                              weights = rs[:,0,epoch_num-1])
    update_dens_iter_2 = GKDE(params_init.T,
                              weights = rs[:,1,epoch_num-1])
    if num_QoI==3:
        update_dens_iter_3 = GKDE(params_init.T, 
                              weights = rs[:,2,epoch_num-1])
    
    kde_truth = GKDE(params_dg.T)

    # Plot settings defined as dict
    plot_dict = {'marker' : 'o', 
                 'cmap' : 'hot_r', 
                 's': 10}
    if fix_cbar_ranges is True:
        true_vals = kde_truth(params_init.T)
        true_max = np.max(true_vals)
        true_min = np.min(true_vals)
        plot_dict['vmin'] = true_min
        plot_dict['vmax'] = true_max

    # Create plots
    fig = plt.figure(num=fignum, figsize=(9, 6), tight_layout=True)
    fig.clf()
    gs = fig.add_gridspec(2,6)
    
    ax1 = fig.add_subplot(gs[0,0:2])
    scatter1 = ax1.scatter(plotting_params[:,0], plotting_params[:,1],
                c = update_dens_iter_1(plotting_params.T),
                **plot_dict)
    ax1.set_aspect('equal', adjustable='box')
    title_str = '1st iter. of epoch ' + str(epoch_num)
    ax1.set_title(title_str)
    ax1.set_xlabel(r'$\lambda_1$')
    ax1.set_ylabel(r'$\lambda_2$')
    plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)
    
    
    ax2 = fig.add_subplot(gs[0,2:4])
    scatter2 = ax2.scatter(plotting_params[:,0], plotting_params[:,1],
                c = update_dens_iter_2(plotting_params.T), 
                **plot_dict)
    ax2.set_aspect('equal', adjustable='box')
    title_str = '2nd iter. of epoch ' + str(epoch_num)
    ax2.set_title(title_str)
    ax2.set_xlabel(r'$\lambda_1$')
    ax2.set_ylabel(r'$\lambda_2$')
    plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)
    
    if num_QoI==3:
        ax3 = fig.add_subplot(gs[0,4:])
        scatter3 = ax3.scatter(plotting_params[:,0], plotting_params[:,1],
                    c = update_dens_iter_3(plotting_params.T),
                    **plot_dict)
        ax3.set_aspect('equal', adjustable='box')
        ax3.set_title('3rd iter. of epoch ' + str(epoch_num))
        ax3.set_xlabel(r'$\lambda_1$')
        ax3.set_ylabel(r'$\lambda_2$')
        plt.colorbar(scatter3, ax=ax3, fraction=0.046, pad=0.04)

        ax5 = fig.add_subplot(gs[1,0:2])
        pred_dens = GKDE(Q_pred_vals[:,0],weights=rs[:,1,epoch_num-1])
        min_Q = np.min(Q_pred_vals[:,0])
        max_Q = np.max(Q_pred_vals[:,0])
        plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
        ax5.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
        ax5.plot(plt_Qs, observed_densities[0](plt_Qs), label='Observed')
        ax5.set_xlabel(r'$Q_1$')
        ax5.set_title('$Q_1$ densities after epoch ' + str(epoch_num))
        ax5.legend()
        
        ax6 = fig.add_subplot(gs[1,2:4])
        pred_dens = GKDE(Q_pred_vals[:,1],weights=rs[:,1,epoch_num-1])
        min_Q = np.min(Q_pred_vals[:,1])
        max_Q = np.max(Q_pred_vals[:,1])
        plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
        ax6.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
        ax6.plot(plt_Qs, observed_densities[1](plt_Qs), label='Observed')
        ax6.set_xlabel(r'$Q_2$')
        ax6.set_title('$Q_2$ densities after epoch ' + str(epoch_num))
        ax6.legend()

        ax7 = fig.add_subplot(gs[1,4:])
        pred_dens = GKDE(Q_pred_vals[:,2],weights=rs[:,2,epoch_num-1])
        min_Q = np.min(Q_pred_vals[:,2])
        max_Q = np.max(Q_pred_vals[:,2])
        plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
        ax7.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
        ax7.plot(plt_Qs, observed_densities[2](plt_Qs), label='Observed')
        ax7.set_xlabel(r'$Q_3$')
        ax7.set_title('$Q_3$ densities after epoch ' + str(epoch_num))
        ax7.legend()
    
    else: 
        ax4 = fig.add_subplot(gs[0,4:])
        scatter4 = ax4.scatter(plotting_params[:,0], plotting_params[:,1],
                    c = kde_truth(plotting_params.T),
                    **plot_dict)
        ax4.set_aspect('equal', adjustable='box')
        ax4.set_title('KDE of DG')
        ax4.set_xlabel(r'$\lambda_1$')
        ax4.set_ylabel(r'$\lambda_2$')
        plt.colorbar(scatter4, ax=ax4, fraction=0.046, pad=0.04)
    
        ax5 = fig.add_subplot(gs[1,0:3])
        pred_dens = GKDE(Q_pred_vals[:,0],weights=rs[:,1,epoch_num-1])
        min_Q = np.min(Q_pred_vals[:,0])
        max_Q = np.max(Q_pred_vals[:,0])
        plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
        ax5.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
        ax5.plot(plt_Qs, observed_densities[0](plt_Qs), label='Observed')
        ax5.set_xlabel(r'$Q_1$')
        ax5.set_title('$Q_1$ densities after epoch ' + str(epoch_num))
        ax5.legend()
        
        ax6 = fig.add_subplot(gs[1,3:])
        pred_dens = GKDE(Q_pred_vals[:,1],weights=rs[:,1,epoch_num-1])
        min_Q = np.min(Q_pred_vals[:,1])
        max_Q = np.max(Q_pred_vals[:,1])
        plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
        ax6.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
        ax6.plot(plt_Qs, observed_densities[1](plt_Qs), label='Observed')
        ax6.set_xlabel(r'$Q_2$')
        ax6.set_title('$Q_2$ densities after epoch ' + str(epoch_num))
        ax6.legend()

    plt.tight_layout(pad=0.25)
    plt.show()


# In[16]:


fignum = 3

interact(plot_epoch, 
         fignum = widgets.fixed(fignum),
         epoch_num=IntSlider(min=1, max=last_epoch, steps=1),
         observed_densities=widgets.fixed(observed_densities),
         fix_cbar_ranges=widgets.Checkbox(value=True, 
                                          description='KDE of DG defines all colorbars',
                                          width='auto',
                                          disabled=False))

fignum = 3

interact(plot_epoch, 
         fignum = widgets.fixed(fignum),
         epoch_num=IntSlider(min=1, max=last_epoch, steps=1, value=5),
         observed_densities=widgets.fixed(observed_densities),
         fix_cbar_ranges=widgets.Checkbox(value=True, 
                                          description='KDE of DG defines all colorbars',
                                          width='auto',
                                          disabled=False))

fignum = 3

interact(plot_epoch, 
         fignum = widgets.fixed(fignum),
         epoch_num=IntSlider(min=1, max=last_epoch, steps=1, value=last_epoch),
         observed_densities=widgets.fixed(observed_densities),
         fix_cbar_ranges=widgets.Checkbox(value=True, 
                                          description='KDE of DG defines all colorbars',
                                          width='auto',
                                          disabled=False))


# In[17]:


total_iters = num_QoI * (last_epoch-1) + last_iter
kl_from_truth = np.zeros(total_iters)

epoch_count = 0

kde_truth = GKDE(params_dg.T)

true_probs = kde_truth(params_init.T)

for i in range(total_iters):
    if i % num_QoI == 0:
        epoch_count += 1
    current_rs = rs[:, i % num_QoI, epoch_count-1]
    update_dens = GKDE(params_init.T, weights=current_rs)
    current_probs = update_dens(params_init.T)
    kl_from_truth[i] = entropy(true_probs, current_probs) / num_init_samples


# In[18]:


fig4 = plt.figure(num=4)
fig4.clf()
plt.plot(np.arange(1, total_iters+1), kl_from_truth)
plt.title('KL divergence in parameter space')
plt.xlabel('Iteration #')
plt.show()


# In[19]:


epoch_count = 0

true_obs_probs = []
for j in range(num_QoI):
    true_obs_probs.append(observed_densities[j](Q_pred_vals[:,j]))

kl_from_observed_marginal = np.zeros((total_iters,num_QoI))

for i in range(total_iters):
    if i % num_QoI == 0:
        epoch_count += 1
    current_rs = rs[:, i % num_QoI, epoch_count-1]
    for j in range(num_QoI):
        pred_dens = GKDE(Q_pred_vals[:, j], weights=current_rs)
        current_probs = pred_dens(Q_pred_vals[:,j])
        kl_from_observed_marginal[i, j] = entropy(true_obs_probs[j], current_probs) / num_init_samples


# In[20]:


fig5 = plt.figure(num=5)
fig5.clf()
plt.semilogy(np.arange(1, total_iters+1), kl_from_observed_marginal[:,0], label='QoI #1')
plt.semilogy(np.arange(1, total_iters+1), kl_from_observed_marginal[:,1], c='k',
           ls=':',label='QoI #2')
plt.title('KL divergence in data spaces')
plt.xlabel('Iteration #')
plt.show()
plt.legend()
plt.tight_layout(pad=0.25)


# In[21]:


# Joint DCI computation

observed_density = GKDE(Q_obs_vals.T)
predicted_density = GKDE(Q_pred_vals.T)
r_joint = np.divide(observed_density(Q_pred_vals.T), predicted_density(Q_pred_vals.T))
print(np.mean(r_joint))
updated_density = GKDE(params_init.T, weights=r_joint)

# DCI of incorrect observed computation

observed_density_wrong = lambda x: observed_densities[0](x[0,:]) * observed_densities[1](x[1,:])
r_joint_wrong = np.divide(observed_density_wrong(Q_pred_vals.T), predicted_density(Q_pred_vals.T))
print(np.mean(r_joint_wrong))
updated_density_wrong = GKDE(params_init.T, weights=r_joint_wrong)


# In[22]:


# Joint DCI results

fig = plt.figure(num=6, figsize=(9,6), tight_layout=True)
fig.clf()
gs = fig.add_gridspec(2,6)

plot_dict = {'marker' : 'o', 
             'cmap' : 'hot_r', 
             's': 10}
true_vals = kde_truth(params_init.T)
true_max = np.max(true_vals)
true_min = np.min(true_vals)
plot_dict['vmin'] = true_min
plot_dict['vmax'] = true_max

ax1 = fig.add_subplot(gs[0,1:3])
scatter1 = ax1.scatter(plotting_params[:,0], plotting_params[:,1],
            c = updated_density(plotting_params.T),
            **plot_dict)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title('Joint DCI results')
ax1.set_xlabel(r'$\lambda_1$')
ax1.set_ylabel(r'$\lambda_2$')
plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(gs[0,3:5])
scatter2 = ax2.scatter(plotting_params[:,0], plotting_params[:,1],
            c = kde_truth(plotting_params.T),
            **plot_dict)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title('KDE of DG')
ax2.set_xlabel(r'$\lambda_1$')
ax2.set_ylabel(r'$\lambda_2$')
plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)

ax5 = fig.add_subplot(gs[1,0:3])
pred_dens = GKDE(Q_pred_vals[:,0],weights=r_joint)
min_Q = np.min(Q_pred_vals[:,0])
max_Q = np.max(Q_pred_vals[:,0])
plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
ax5.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
ax5.plot(plt_Qs, observed_densities[0](plt_Qs), label='Observed')
ax5.set_xlabel(r'$Q_1$')
ax5.set_title('$Q_1$ densities for joint DCI')
ax5.legend()

ax6 = fig.add_subplot(gs[1,3:])
pred_dens = GKDE(Q_pred_vals[:,1],weights=r_joint)
min_Q = np.min(Q_pred_vals[:,1])
max_Q = np.max(Q_pred_vals[:,1])
plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
ax6.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
ax6.plot(plt_Qs, observed_densities[1](plt_Qs), label='Observed')
ax6.set_xlabel(r'$Q_2$')
ax6.set_title('$Q_2$ densities for joint DCI')
ax6.legend()

plt.tight_layout(pad=0.25)
plt.show()


# In[23]:


# Wrong DCI results

fig = plt.figure(num=7, figsize=(9,6), tight_layout=True)
fig.clf()
gs = fig.add_gridspec(2,6)

plot_dict = {'marker' : 'o', 
             'cmap' : 'hot_r', 
             's': 10}
true_vals = kde_truth(params_init.T)
true_max = np.max(true_vals)
true_min = np.min(true_vals)
plot_dict['vmin'] = true_min
plot_dict['vmax'] = true_max

ax1 = fig.add_subplot(gs[0,1:3])
scatter1 = ax1.scatter(plotting_params[:,0], plotting_params[:,1],
            c = updated_density_wrong(plotting_params.T),
            **plot_dict)
ax1.set_aspect('equal', adjustable='box')
ax1.set_title('Incorrect joint DCI results')
ax1.set_xlabel(r'$\lambda_1$')
ax1.set_ylabel(r'$\lambda_2$')
plt.colorbar(scatter1, ax=ax1, fraction=0.046, pad=0.04)

ax2 = fig.add_subplot(gs[0,3:5])
scatter2 = ax2.scatter(plotting_params[:,0], plotting_params[:,1],
            c = kde_truth(plotting_params.T),
            **plot_dict)
ax2.set_aspect('equal', adjustable='box')
ax2.set_title('KDE of DG')
ax2.set_xlabel(r'$\lambda_1$')
ax2.set_ylabel(r'$\lambda_2$')
plt.colorbar(scatter2, ax=ax2, fraction=0.046, pad=0.04)

ax5 = fig.add_subplot(gs[1,0:3])
pred_dens = GKDE(Q_pred_vals[:,0],weights=r_joint_wrong)
min_Q = np.min(Q_pred_vals[:,0])
max_Q = np.max(Q_pred_vals[:,0])
plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
ax5.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
ax5.plot(plt_Qs, observed_densities[0](plt_Qs), label='Observed')
ax5.set_xlabel(r'$Q_1$')
ax5.set_title('$Q_1$ densities for incorrect joint DCI')
ax5.legend()

ax6 = fig.add_subplot(gs[1,3:])
pred_dens = GKDE(Q_pred_vals[:,1],weights=r_joint_wrong)
min_Q = np.min(Q_pred_vals[:,1])
max_Q = np.max(Q_pred_vals[:,1])
plt_Qs = np.linspace(min_Q-0.05*(max_Q-min_Q), max_Q+0.05*(max_Q-min_Q), 101)
ax6.plot(plt_Qs, pred_dens(plt_Qs), linestyle=':', linewidth=2, label='PF of Update')
ax6.plot(plt_Qs, observed_densities[1](plt_Qs), label='Observed')
ax6.set_xlabel(r'$Q_2$')
ax6.set_title('$Q_2$ densities for incorrect joint DCI')
ax6.legend()

plt.tight_layout(pad=0.25)
plt.show()


# In[24]:


# For last part of example

# 3D linear map
Q_maps = [lambda x: 2*x[:,0] + x[:,1], lambda x: 2.5*x[:,0]+0.5*x[:,1], lambda x: -x[:,0] + x[:,1]]


# In[ ]:





# In[25]:


num_QoI = len(Q_maps)
Q_pred_vals = np.zeros((num_init_samples, num_QoI))
Q_obs_vals = np.zeros((num_dg_samples, num_QoI))

for d in range(num_QoI):
    Q_pred_vals[:, d] = Q_maps[d](params_init)
    Q_obs_vals[:, d] = Q_maps[d](params_dg)


# In[26]:


fig = plt.figure(num=8, figsize=(3,3))
fig.clf()
ax = fig.add_subplot(projection='3d')
ax.scatter(Q_pred_vals[:,0], Q_pred_vals[:,1], Q_pred_vals[:,2], label='Predicted', alpha=0.125)
ax.scatter(Q_obs_vals[:,0], Q_obs_vals[:,1], Q_obs_vals[:,2], marker='+', label='Observed')
ax.set_xlabel(r'$Q_1$')
ax.set_ylabel(r'$Q_2$')
ax.set_zlabel(r'$Q_3$')
ax.legend()
plt.tight_layout(pad=1)
plt.show()


# In[27]:


observed_densities = []
for d in range(num_QoI):
    observed_densities.append(GKDE(Q_obs_vals[:,d]))


# In[28]:


num_epochs = 100

rs, last_epoch, last_iter, kl_epochs = iterative_DCI(Q_pred_vals, observed_densities, 
                                           num_epochs = num_epochs, r_init = 1, ratio_tol=0.1)


# In[29]:


fignum = 9

interact(plot_epoch, 
         fignum = widgets.fixed(fignum),
         epoch_num=IntSlider(min=1, max=last_epoch, steps=1, value=last_epoch),
         observed_densities=widgets.fixed(observed_densities),
         fix_cbar_ranges=widgets.Checkbox(value=True, 
                                          description='KDE of DG defines all colorbars',
                                          width='auto',
                                          disabled=False))


# In[ ]:




