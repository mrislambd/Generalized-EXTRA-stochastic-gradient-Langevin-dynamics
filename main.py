# %%
from sklearn import preprocessing
from sklearn.datasets import load_breast_cancer
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pickle
from networks import Network
from extrasgld import GeneralizedExtraSGLD
from evaluation import ClassificationAccuracy, Wasserstein2Distance
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# %% [markdown]
# # Bayesian Logistic Regression with Synthetic Data
# ## Parameters

# %%
size_w = 6
N = 100
sigma = 1
eta = 0.0005
T = 20
dim = 3
b = 32
lam = 10
total_data = 1000
hv = np.linspace(0.001, 0.5, 10)

# %% [markdown]
# ## Data Generation and Preparation

# %%
x = []
np.random.seed(10)
for i in range(total_data):
    x.append([-20 + (20 + 20) * np.random.normal(), -10 + np.random.normal()])
np.random.seed(11)
y = [1 / (1 + np.exp(-item[0] * 1 - 1 * item[1] + 10)) for item in x]
for i in range(len(y)):
    temp = np.random.uniform(0, 1)
    if temp <= y[i]:
        y[i] = 1
    else:
        y[i] = 0

x_all = np.array(x)
y_all = np.array(y)
x_all = x
y_all = y
x_all = np.insert(x_all, 0, 1, axis=1)
x = x_all

'''
    Data splitting
'''

X_train1, x_trainRemain, y_train1, y_trainRemain = train_test_split(
    x, y, test_size=0.83333, random_state=42
)
X_train2, x_trainRemain, y_train2, y_trainRemain = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.8, random_state=42
)
X_train3, x_trainRemain, y_train3, y_trainRemain = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.75, random_state=42
)
X_train4, x_trainRemain, y_train4, y_trainRemain = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.66666666, random_state=42
)
X_train5, X_train6, y_train5, y_train6 = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.5, random_state=42
)
x = [X_train1, X_train2, X_train3, X_train4, X_train5, X_train6]
y = [y_train1, y_train2, y_train3, y_train4, y_train5, y_train6]

# %% [markdown]
# ## Communication Network

# %%
net = Network(size_w)
wf = net.fully_connected_network()
wc = net.circular_network()
ws = net.star_network()
wd = net.disconnected_network()

nets = np.array([wf, wc, ws, wd])

# %% [markdown]
# ## SGLD Method

# %%
four_net_combined = []
for i in tqdm(range(len(nets))):
    method = GeneralizedExtraSGLD(
        size_w, N, sigma, eta, T, dim, b, lam, x, y, nets[i], hv, 'logreg')
    sgld, _ = method.sgld()
    four_net_combined.append(sgld)

# %% [markdown]
# ## SGLD Accuracy

# %%
sgld_acc = []
sgld_std = []
for sgld in tqdm(four_net_combined):
    sgld_accuracy = ClassificationAccuracy(x_all, y_all, sgld, T)
    d_acc, d_std = sgld_accuracy.compute_accuracy()
    sgld_acc.append(d_acc)
    sgld_std.append(d_std)
sgld_acc = np.array(sgld_acc)
sgld_std = np.array(sgld_std)

# %% [markdown]
# ## SGLD Accuracy Plot

# %%
fig, axs = plt.subplots(1, len(sgld_acc), figsize=(36, 8))
mpl.rcParams['font.size'] = 24
index = list(range(T + 1))
titles = ["Fully Connected", "Circular", "Star", "Disconnected"]
for i, (d_acc, d_std, title) in enumerate(zip(sgld_acc, sgld_std, titles)):
    axs[i].plot(d_acc, 'b-', linewidth=3)
    axs[i].fill_between(index, d_acc + d_std, d_acc - d_std, alpha=0.5)
    axs[i].set_title(title, fontsize=24)
    axs[i].legend(['Mean of accuracy', r'Accuracy $\pm$ Std'],
                  loc='lower right', fontsize=24)
    axs[i].set_xlabel(r'Iterations $k$', fontsize=20)
    axs[i].set_ylabel('Accuracy', fontsize=24)
    axs[i].set_ylim(0, 1)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()
plt.tight_layout()
plt.savefig('otherplots/fig1.png')
plt.show()

# %% [markdown]
# ## EXTRA SGLD Method

# %%
four_net_combined = []
for i in tqdm(range(len(nets))):
    method = GeneralizedExtraSGLD(
        size_w, N, sigma, eta, T, dim, b, lam, x, y, nets[i], hv, 'logreg')
    extra_sgld, _ = method.extra_sgld()
    four_net_combined.append(extra_sgld)

# %% [markdown]
# ## EXTRA SGLD Accuracy

# %%
extra_sgld_acc = []
extra_sgld_std = []
for i in tqdm(range(len(nets))):
    exacc = []
    exstd = []
    for j in range(len(hv)):
        extra_sgld_accuracy = ClassificationAccuracy(
            x_all, y_all, four_net_combined[i][j], T
        )
        acc, std = extra_sgld_accuracy.compute_accuracy()
        exacc.append(acc)
        exstd.append(std)
    extra_sgld_acc.append(exacc)
    extra_sgld_std.append(exstd)

extra_sgld_acc = np.array(extra_sgld_acc)
extra_sgld_std = np.array(extra_sgld_std)

# %% [markdown]
# ## Plot EXTRA SGLD for each $h$ value

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)
mpl.rcParams['font.size'] = 24
index = list(range(T + 1))


titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for j, title in enumerate(titles):
    for i, h_value in enumerate(hv):
        axs[j].plot(index, extra_sgld_acc[j][i], label=f'h = {h_value:.3f}')
    axs[j].set_title(title, fontsize=24)
    axs[j].legend(loc='lower right')
    axs[j].set_xlabel(r'Iterations $k$', fontsize=20)
    axs[j].set_ylabel('Accuracy', fontsize=24)
    axs[j].set_ylim(0, 1)
    axs[j].tick_params(labelsize=24)
    axs[j].grid()
plt.tight_layout()
plt.savefig('otherplots/fig2.png')
plt.show()

# %% [markdown]
# ## Optimal $h$ for maximum accuracy

# %%
max_acc_arrays = []
max_acc_indices = []

for i in range(len(nets)):
    max_acc_array = extra_sgld_acc[i][np.argmax(
        np.max(extra_sgld_acc[i], axis=1))]
    max_acc_index = np.argmax(np.max(extra_sgld_acc[i], axis=1))
    max_acc_arrays.append(max_acc_array)
    max_acc_indices.append(max_acc_index)

max_acc_arrays = np.array(max_acc_arrays)
max_acc_indices = np.array(max_acc_indices)
max_hv_values = hv[max_acc_indices]
print('LogReg: Optimal h for maximum accuracy on synthetic data\n')
print(max_hv_values)

max_acc_stds = np.zeros((len(nets), (T+1)))
for i in range(len(nets)):
    max_acc_stds[i] = extra_sgld_std[i, max_acc_indices[i]]

# %% [markdown]
# ## Plot the maximum accuracy

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)

mpl.rcParams['font.size'] = 24
index = list(range(T + 1))


titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for i, title in enumerate(titles):
    axs[i].plot(max_acc_arrays[i], 'r-', linewidth=3,
                label=f'h = {max_hv_values[i]:.3f}')
    axs[i].fill_between(
        index, max_acc_arrays[i]+max_acc_stds[i],
        max_acc_arrays[i]-max_acc_stds[i], alpha=0.25, color='red'
    )
    axs[i].set_title(title, fontsize=24)
    axs[i].legend(loc='lower right')
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel('Accuracy', fontsize=24)
    axs[i].set_ylim(0, 1)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()
plt.tight_layout()
plt.savefig('otherplots/fig3.png')
plt.show()

# %% [markdown]
# ## SGLD vs EXTRA SGLD comparison

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)

mpl.rcParams['font.size'] = 24
index = list(range(T + 1))

titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for i, title in enumerate(titles):
    axs[i].plot(sgld_acc[i], 'b-', linewidth=3, label='DE-SGLD')
    axs[i].fill_between(
        index, sgld_acc[i] + sgld_std[i],
        sgld_acc[i] - sgld_std[i], alpha=0.25, color='blue'
    )

    axs[i].plot(max_acc_arrays[i], 'r-', linewidth=3, label='EXTRA SGLD')
    axs[i].fill_between(
        index, max_acc_arrays[i] + max_acc_stds[i],
        max_acc_arrays[i] - max_acc_stds[i], alpha=0.25, color='red'
    )
    axs[i].set_title(title, fontsize=24)
    axs[i].legend(loc='lower right')
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel('Accuracy', fontsize=24)
    axs[i].set_ylim(0, 1)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()

plt.tight_layout()
plt.savefig('plots/fig5.png', dpi=600)
plt.show()

# %% [markdown]
# # Bayesian Logistic Regression with Real Data
# ## Parameters

# %%
size_w = 6
N = 50
sigma = 1
eta = 0.0008
T = 100
dim = 31
b = 32
lam = 10
total_data = 1000
hv = np.linspace(0.001, 0.5, 10)

# %% [markdown]
# ## Data

# %%

cancer = load_breast_cancer()
x = cancer.data
y = cancer.target

lam = 1*len(x)

x_all = x
y_all = y

x_all = np.insert(x_all, 0, 1, axis=1)
x_all = preprocessing.scale(x_all)
x = x_all

'''
    Data splitting
'''

X_train1, x_trainRemain, y_train1, y_trainRemain = train_test_split(
    x, y, test_size=0.83333, random_state=42
)
X_train2, x_trainRemain, y_train2, y_trainRemain = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.8, random_state=42
)
X_train3, x_trainRemain, y_train3, y_trainRemain = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.75, random_state=42
)
X_train4, x_trainRemain, y_train4, y_trainRemain = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.66666666, random_state=42
)
X_train5, X_train6, y_train5, y_train6 = train_test_split(
    x_trainRemain, y_trainRemain, test_size=0.5, random_state=42
)
x = [X_train1, X_train2, X_train3, X_train4, X_train5, X_train6]
y = [y_train1, y_train2, y_train3, y_train4, y_train5, y_train6]

# %% [markdown]
# ## Communication Network

# %%
net = Network(size_w)
wf = net.fully_connected_network()
wc = net.circular_network()
ws = net.star_network()
wd = net.disconnected_network()

nets = np.array([wf, wc, ws, wd])

# %% [markdown]
# ## SGLD Method

# %%
four_net_combined = []
for i in tqdm(range(len(nets))):
    method = GeneralizedExtraSGLD(
        size_w=size_w, N=N, sigma=sigma, eta=eta, T=T,
        dim=dim, b=b, lam=lam, x=x, y=y, w=nets[i],
        hv=hv, reg_type='logreg'
    )
    sgld, _ = method.sgld()
    four_net_combined.append(sgld)

# %% [markdown]
# ## SGLD Accuracy

# %%
sgld_acc = []
sgld_std = []
for sgld in tqdm(four_net_combined):
    sgld_accuracy = ClassificationAccuracy(x_all, y_all, sgld, T)
    d_acc, d_std = sgld_accuracy.compute_accuracy()
    sgld_acc.append(d_acc)
    sgld_std.append(d_std)
sgld_acc = np.array(sgld_acc)
sgld_std = np.array(sgld_std)

# %% [markdown]
# ## SGLD Accuracy Plot

# %%
fig, axs = plt.subplots(1, len(sgld_acc), figsize=(36, 8))
mpl.rcParams['font.size'] = 24
index = list(range(T + 1))
titles = ["Fully Connected", "Circular", "Star", "Disconnected"]
for i, (d_acc, d_std, title) in enumerate(zip(sgld_acc, sgld_std, titles)):
    axs[i].plot(d_acc, 'b-', linewidth=3)
    axs[i].fill_between(index, d_acc + d_std, d_acc - d_std, alpha=0.5)
    axs[i].set_title(title, fontsize=24)
    axs[i].legend(['Mean of accuracy', r'Accuracy $\pm$ Std'],
                  loc='lower right', fontsize=24)
    axs[i].set_xlabel(r'Iterations $k$', fontsize=20)
    axs[i].set_ylabel('Accuracy', fontsize=24)
    axs[i].set_ylim(0, 1)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()
plt.tight_layout()
plt.savefig('otherplots/fig4.png')
plt.show()

# %% [markdown]
# ## EXTRA SGLD Method

# %%
four_net_combined = []
for i in tqdm(range(len(nets))):
    method = GeneralizedExtraSGLD(
        size_w=size_w, N=N, sigma=sigma, eta=eta, T=T,
        dim=dim, b=b, lam=lam, x=x, y=y, w=nets[i],
        hv=hv, reg_type='logreg'
    )
    extra_sgld, _ = method.extra_sgld()
    four_net_combined.append(extra_sgld)

# %% [markdown]
# ## EXTRA SGLD Accuracy

# %%
extra_sgld_acc = []
extra_sgld_std = []
for i in tqdm(range(len(nets))):
    exacc = []
    exstd = []
    for j in range(len(hv)):
        extra_sgld_accuracy = ClassificationAccuracy(
            x_all, y_all, four_net_combined[i][j], T
        )
        acc, std = extra_sgld_accuracy.compute_accuracy()
        exacc.append(acc)
        exstd.append(std)
    extra_sgld_acc.append(exacc)
    extra_sgld_std.append(exstd)

extra_sgld_acc = np.array(extra_sgld_acc)
extra_sgld_std = np.array(extra_sgld_std)

# %% [markdown]
# ## Plot Accuracy for EXTRA SGLD for each $h$ value

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)
mpl.rcParams['font.size'] = 24
index = list(range(T + 1))


titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for j, title in enumerate(titles):
    for i, h_value in enumerate(hv):
        axs[j].plot(index, extra_sgld_acc[j][i], label=f'h = {h_value:.3f}')
    axs[j].set_title(title, fontsize=24)
    axs[j].legend(loc='lower right')
    axs[j].set_xlabel(r'Iterations $k$', fontsize=20)
    axs[j].set_ylabel('Accuracy', fontsize=24)
    axs[j].set_ylim(0, 1)
    axs[j].tick_params(labelsize=24)
    axs[j].grid()
plt.tight_layout()
plt.savefig('otherplots/fig5.png')
plt.show()

# %% [markdown]
# ## Optimal $h$ for maximum accuracy

# %%
max_acc_arrays = []
max_acc_indices = []

for i in range(len(nets)):
    max_acc_array = extra_sgld_acc[i][np.argmax(
        np.max(extra_sgld_acc[i], axis=1))]
    max_acc_index = np.argmax(np.max(extra_sgld_acc[i], axis=1))
    max_acc_arrays.append(max_acc_array)
    max_acc_indices.append(max_acc_index)

max_acc_arrays = np.array(max_acc_arrays)
max_acc_indices = np.array(max_acc_indices)
max_hv_values = hv[max_acc_indices]
print('LogReg: Optimal h for maximum accuracy on real data\n')
print(max_hv_values)

max_acc_stds = np.zeros((len(nets), (T+1)))
for i in range(len(nets)):
    max_acc_stds[i] = extra_sgld_std[i, max_acc_indices[i]]

# %% [markdown]
# ## Plot the maximum accuracy

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)

mpl.rcParams['font.size'] = 24
index = list(range(T + 1))


titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for i, title in enumerate(titles):
    axs[i].plot(max_acc_arrays[i], 'r-', linewidth=3,
                label=f'h = {max_hv_values[i]:.3f}')
    axs[i].fill_between(
        index, max_acc_arrays[i]+max_acc_stds[i],
        max_acc_arrays[i]-max_acc_stds[i], alpha=0.25, color='red'
    )
    axs[i].set_title(title, fontsize=24)
    axs[i].legend(loc='lower right')
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel('Accuracy', fontsize=24)
    axs[i].set_ylim(0, 1)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()
plt.tight_layout()
plt.savefig('otherplots/fig6.png')
plt.show()

# %% [markdown]
# ## SGLD vs EXTRA SGLD Comparison

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)

mpl.rcParams['font.size'] = 24
index = list(range(T + 1))

titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for i, title in enumerate(titles):
    axs[i].plot(sgld_acc[i], 'b-', linewidth=3, label='DE-SGLD')
    axs[i].fill_between(
        index, sgld_acc[i] + sgld_std[i],
        sgld_acc[i] - sgld_std[i], alpha=0.25, color='blue'
    )

    axs[i].plot(max_acc_arrays[i], 'r-', linewidth=3, label='EXTRA SGLD')
    axs[i].fill_between(
        index, max_acc_arrays[i] + max_acc_stds[i],
        max_acc_arrays[i] - max_acc_stds[i], alpha=0.25, color='red'
    )
    axs[i].set_title(title, fontsize=24)
    axs[i].legend(loc='lower right')
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel('Accuracy', fontsize=24)
    axs[i].set_ylim(0, 1)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()

plt.tight_layout()
plt.savefig('plots/fig6.png', dpi=600)
plt.show()

# %% [markdown]
# # Bayesian Linear Regression with Synthetic Data
#
# ## Parameters

# %%
size_w = 20
N = 100
dim = 2
sigma = np.eye(dim)
sigma_sample = 1
eta = 0.009
T = 200
lam = 10
b = 50
hv = np.linspace(0.001, 0.5, 5)

# %% [markdown]
# ## Data

# %%
x = []

for i in range(5000):
    x.append([np.random.random()*1])

y = [item[0]*3-0.5+np.random.random() for item in x]

x_all = np.array(x)
y_all = np.array(y)
x_all = np.insert(x_all, 0, 1, axis=1)
cov_pri = lam*sigma

avg_post = np.dot(
    np.linalg.inv(np.linalg.inv(cov_pri) +
                  np.dot(np.transpose(x_all), x_all)/(sigma_sample**2)),
    (np.dot(np.transpose(x_all), y_all)/(sigma_sample**2))
)
cov_post = np.linalg.inv(np.linalg.inv(
    cov_pri)+np.dot(np.transpose(x_all), x_all)/(sigma_sample**2))

x = np.split(x_all, 100)
y = np.split(y_all, 100)

# %% [markdown]
# ## Communication Network

# %%
net = Network(size_w)
wf = net.fully_connected_network()
wc = net.circular_network()
ws = net.star_network()
wd = net.disconnected_network()

nets = np.array([wf, wc, ws, wd])

# %% [markdown]
# ## SGLD Method

# %%
four_net_combined_agents = []
four_net_combined_magents = []

for i in tqdm(range(len(nets))):
    method = GeneralizedExtraSGLD(
        size_w=size_w, N=N, sigma=sigma, eta=eta, T=T,
        dim=dim, b=b, lam=lam, x=x, y=y, w=nets[i],
        hv=hv, reg_type='linreg'
    )
    sgld_agents, sgld_magents = method.sgld()
    four_net_combined_agents.append(sgld_agents)
    four_net_combined_magents.append(sgld_magents)

# %% [markdown]
# ## Compute the $\mathcal{W}_2$ Distances

# %%
sgld_four_net_combined_dist = []

for dis_agent, dis_magent in tqdm(zip(four_net_combined_agents,
                                      four_net_combined_magents)):
    distance = Wasserstein2Distance(
        size_w, T, avg_post, cov_post, dis_agent, dis_magent)
    dis = distance.W2_dist()
    sgld_four_net_combined_dist.append(dis)

with open('sgld_four_net_combined_dist.pkl', 'wb') as f:
    pickle.dump(sgld_four_net_combined_dist, f)

# %% [markdown]
# ## Plot

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8))
mpl.rcParams['font.size'] = 24
index = list(range(T + 1))
titles = ["Fully Connected", "Circular", "Star", "Disconnected"]

for i in range(len(sgld_four_net_combined_dist)):
    axs[i].plot(index, sgld_four_net_combined_dist[i][0],
                linewidth=3, label=r'Agent 1 $x_1^{(k)}$')
    axs[i].plot(sgld_four_net_combined_dist[i][1],
                linewidth=3, label=r'Agent 2 $x_2^{(k)}$')
    axs[i].plot(sgld_four_net_combined_dist[i][2],
                linewidth=3, label=r'Agent 3 $x_3^{(k)}$')
    axs[i].plot(sgld_four_net_combined_dist[i][3],
                linewidth=3, label=r'Agent 4 $x_4^{(k)}$')
    axs[i].plot(sgld_four_net_combined_dist[i][-1], linewidth=3,
                label=r'Mean of Agents $\bar{x}^{(k)}$')
    axs[i].set_title(titles[i], fontsize=24)
    axs[i].legend(loc='upper right', fontsize=17)
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel(r'$\mathcal{W}_2$ Distance ', fontsize=24)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()
plt.tight_layout()
plt.savefig('otherplots/fig7.png')
plt.show()

# %% [markdown]
# ## EXTRA SGLD Method

# %%
four_net_combined_agents = []
four_net_combined_magents = []
for i in tqdm(range(len(nets))):
    method = GeneralizedExtraSGLD(
        size_w, N, sigma, eta, T, dim, b, lam,
        x, y, nets[i], hv, reg_type="linreg"
    )
    extra_sgld_agents, extra_sgld_magents = method.extra_sgld()
    four_net_combined_agents.append(extra_sgld_agents)
    four_net_combined_magents.append(extra_sgld_magents)

# %% [markdown]
# ## Compute the $\mathcal{W}_2$ Distances

# %%
extra_sgld_four_net_combined_dist = []

for i in tqdm(range(len(nets))):
    ex = []
    for j in range(len(hv)):
        extra_distance = Wasserstein2Distance(
            size_w, T, avg_post, cov_post,
            four_net_combined_agents[i][j],
            four_net_combined_magents[i][j]
        )
        # Compute Wasserstein 2 distance
        ex.append(extra_distance.W2_dist())
    extra_sgld_four_net_combined_dist.append(ex)

extra_sgld_four_net_combined_dist = np.array(extra_sgld_four_net_combined_dist)

with open('extra_sgld_four_net_combined_dist.pkl', 'wb') as f:
    pickle.dump(extra_sgld_four_net_combined_dist, f)

# %% [markdown]
# ## Work with the $\mathcal{W}_2$ distances of mean agents

# %%
mean_dist = np.empty((len(nets), len(hv), T+1))
for i in tqdm(range(len(nets))):
    for j in range(len(hv)):
        mean_dist[i, j, :] = extra_sgld_four_net_combined_dist[i, j, -1, :]

# %% [markdown]
# ## Plot the $\mathcal{W}_2$ distance of mean agents for each $h$ values

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)
mpl.rcParams['font.size'] = 24

titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for j, title in enumerate(titles):
    for i, h_value in enumerate(hv):
        axs[j].plot(mean_dist[j][i], linewidth=3, label=f'h={h_value:.3f}')
    axs[j].set_title(title, fontsize=24)
    axs[j].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[j].set_ylabel(r'$\mathcal{W}_2$ Dist. of mean agents', fontsize=24)
    axs[j].tick_params(labelsize=24)
    axs[j].legend(loc='upper right')
    axs[j].grid()
plt.tight_layout()
plt.savefig('otherplots/fig8.png')
plt.show()

# %% [markdown]
# ## Find the minimum distance for optimal $h$ values

# %%
min_dist_arrays = []
min_dist_indices = []

for i in range(len(nets)):
    min_dist_array = mean_dist[i][np.argmin(np.min(mean_dist[i], axis=1))]
    min_dist_index = np.argmin(np.min(mean_dist[i], axis=1))
    min_dist_arrays.append(min_dist_array)
    min_dist_indices.append(min_dist_index)

min_dist_arrays = np.array(min_dist_arrays)
min_dist_indices = np.array(min_dist_indices)
min_hv_values = hv[min_dist_indices]

with open('min_dist_arrays.pkl', 'wb') as f:
    pickle.dump(min_dist_arrays, f)

print('LinReg: Optimal h for minimum distances\n')
print(min_hv_values)

# %% [markdown]
# ## Plot the Optimal EXTRA SGLD Mean Distance for the optimal $h$ value

# %%
fig, axs = plt.subplots(1, len(min_dist_arrays),
                        figsize=(36, 8), sharex=True, sharey=True)
mpl.rcParams['font.size'] = 24

titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for i, title in enumerate(titles):
    axs[i].plot(min_dist_arrays[i], 'r-', linewidth=3,
                label=f'h={min_hv_values[i]:.3f}')
    axs[i].set_title(title, fontsize=24)
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel(r'$\mathcal{W}_2$ Dist. of mean agents', fontsize=24)
    axs[i].tick_params(labelsize=24)
    axs[i].legend(loc='upper right')
    axs[i].grid()

plt.tight_layout()
plt.savefig('otherplots/fig9.png')
plt.show()

# %%
# EXTRA DESGLD performance for the optimal $h$ values
fig, axs = plt.subplots(1, 4, figsize=(36, 8))
mpl.rcParams['font.size'] = 24
index = list(range(T + 1))
titles = ['Fully Connected', 'Circular', "Star", 'Disconnected']
positions = np.searchsorted(hv, min_hv_values)
for i in range(len(extra_sgld_four_net_combined_dist)):
    axs[i].plot(index, extra_sgld_four_net_combined_dist[i]
                [positions[i]][0], linewidth=3, label=r'Agent 1 $x_1^{(k)}$')
    axs[i].plot(extra_sgld_four_net_combined_dist[i][positions[i]]
                [1], linewidth=3, label=r'Agent 2 $x_2^{(k)}$')
    axs[i].plot(extra_sgld_four_net_combined_dist[i][positions[i]]
                [2], linewidth=3, label=r'Agent 3 $x_3^{(k)}$')
    axs[i].plot(extra_sgld_four_net_combined_dist[i][positions[i]]
                [3], linewidth=3, label=r'Agent 4 $x_4^{(k)}$')
    axs[i].plot(extra_sgld_four_net_combined_dist[i][positions[i]]
                [-1], linewidth=3, label=r'Mean of Agents $\bar{x}^{(k)}$')
    axs[i].set_title(titles[i], fontsize=24)
    axs[i].legend(loc='upper right', fontsize=17)
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel(r'$\mathcal{W}_2$ Distance ', fontsize=24)
    axs[i].tick_params(labelsize=24)
    axs[i].grid()
plt.tight_layout()
plt.savefig('plots/fig2.png', dpi=600)
plt.show()

# %% [markdown]
# ## Finally, we can compare the performances from both of the algorithms

# %%
fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)
mpl.rcParams['font.size'] = 24

titles = ['Fully Connected', 'Circular', 'Star', 'Disconnected']

for i, title in enumerate(titles):
    axs[i].plot(min_dist_arrays[i], 'r-', linewidth=3,
                label='EXTRA SGLD')
    axs[i].plot(sgld_four_net_combined_dist[i][-1], linewidth=3, label='DE-SGLD')
    axs[i].set_title(title, fontsize=24)
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel(r'$\mathcal{W}_2$ Dist. of mean agents', fontsize=24)
    axs[i].tick_params(labelsize=24)
    axs[i].legend(loc='upper right')
    axs[i].grid()
plt.tight_layout()
plt.savefig('plots/fig3.png', dpi=600)
plt.show()

# %%
with open('sgld_four_net_combined_dist.pkl', 'rb') as f:
    sgld_four_net_combined_dist = pickle.load(f)
with open('min_dist_arrays.pkl', 'rb') as f:
    min_dist_arrays = pickle.load(f)
with open('extra_sgld_four_net_combined_dist.pkl', 'rb') as f:
    extra_sgld_four_net_combined_dist = pickle.load(f)


fig, axs = plt.subplots(1, 4, figsize=(36, 8), sharex=True, sharey=True)
mpl.rcParams['font.size'] = 24

titles = ['Fully Connected', 'Circular', "Star", 'Disconnected']

for i, title in enumerate(titles):
    sns.histplot(
        min_dist_arrays[i], kde=True, color='blue', alpha=0.3,
        ax=axs[i], label='EXTRA SGLD', log_scale=(False, True),
        line_kws={'linewidth': 3},
        bins=int(np.sqrt(len(min_dist_arrays[i])))
    )
    sns.histplot(
        sgld_four_net_combined_dist[i][-1], kde=True, color='green', alpha=0.3,
        ax=axs[i], label='DE-SGLD', log_scale=(False, True),
        line_kws={'linewidth': 3},
        bins=int(np.sqrt(len(sgld_four_net_combined_dist[i][-1])))
    )
    axs[i].set_title(title, fontsize=24)
    axs[i].set_xlabel(r'Iterations $k$', fontsize=24)
    axs[i].set_ylabel(r'$\mathcal{W}_2$ Dist. of mean agents', fontsize=24)
    axs[i].tick_params(labelsize=24)
    axs[i].legend(loc='upper right')
    axs[i].grid()
plt.xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], [
           '0', '50', '100', '125', '150', '175', '200'])
plt.tight_layout()
plt.savefig('plots/fig4.png', dpi=600)
plt.show()
