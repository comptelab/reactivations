
from __future__ import division
from matplotlib.pylab import *
import scipy.io as io
import glob
import sys
from scipy.stats import *
import seaborn as sns
from scikits import bootstrap
from scipy.signal import detrend
from scipy.io import loadmat
from scipy.stats import sem
from scipy import stats
from scipy.ndimage.filters import gaussian_filter
import rpy2.robjects as robjects
from rpy2.robjects import r, pandas2ri
from rpy2.robjects.packages import importr
pandas2ri.activate()

# import the BayesFactor package
BayesFactor = importr('BayesFactor')
sns.set_style("ticks")
sns.set_context("talk", font_scale=0.8)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})


cmap = sns.color_palette("vlag", as_cmap=True)
 
 #sns.color_palette("magma", as_cmap=True)
nboots = 50
#### EXPERIMENT 1
### load data
imp_i = 1125
time_exp1 = np.linspace(-.1,2.646,1374)

# TEMPORARY: load cued
root_dir = "../baseline_decoders/single_trials2017/exp1/"
#root_dir = "../baseline_decoders/single_trials2017/exp1/alpha_"

all_cued = []
for sub in range(1,30):
    print(sub)
    # load single-trial decoders
    files=glob.glob(root_dir+"cued_exp1_single%i.mat" % (sub))[0]
    dec_cued=loadmat(files)["cued_trials"]
    dec_cued = np.mean(dec_cued[:,800:imp_i],-1)
    all_cued.append(dec_cued)

all_cued = np.array(all_cued)

# load uncued
root_dir = "/Users/jbarbosa/Dropbox/Neuro/papers/Reactivation/Dynamic_hidden_states/scripts/decoders_smoothed_EEG/"
dec_uncued = loadmat(root_dir + "uncued_smoothed_2factor_single_trials.mat")

dec_uncued = dec_uncued["dec_uncued"][:,0]
all_uncued= []
for i,d in enumerate(dec_uncued):
    s2 = np.mean(d[:,imp_i-100:imp_i],-1)
    all_uncued.append(s2)

all_uncued = np.array(all_uncued)[range(1,30)]

subjects1 = np.arange(5,len(all_cued),1) 
trials1 = np.arange(50,980,980//20) 
T = np.zeros((2,len(trials1),len(subjects1)))

idx_subjects = list(range(len(all_cued)))

for nt,n_trials in enumerate(trials1):
    print(nt)
    for _ in range(nboots):
        np.random.shuffle(idx_subjects)
        for ns,n_subjects in enumerate(subjects1):

            cued_subjects = all_cued[idx_subjects[:n_subjects]]
            uncued_subjects = all_uncued[idx_subjects[:n_subjects]]

            # shuffle across trials
            [np.random.shuffle(c) for c in cued_subjects]
            [np.random.shuffle(c) for c in uncued_subjects]

            # cued_trials = [np.mean(l[:n_trials]) for l in cued_subjects]
            # uncued_trials = [np.mean(l[:n_trials]) for l in uncued_subjects]

            cued_trials = np.concatenate([l[:n_trials] for l in cued_subjects])
            uncued_trials = np.concatenate([l[:n_trials] for l in uncued_subjects])

            # Bayes Factor
            robjects.globalenv["cued_trials"] = np.array(cued_trials)
            bf = r('ttestBF(cued_trials)[1]@bayesFactor$bf')
            T[0,nt,ns] += np.exp(bf[0])/nboots

            robjects.globalenv["uncued_trials"] = np.array(uncued_trials)
            bf = r('ttestBF(uncued_trials)[1]@bayesFactor$bf')
            T[1,nt,ns] += np.exp(bf[0])/nboots

T_s = T.copy()

T[0] = gaussian_filter(T_s[0], sigma=(2, 2))
T[1] = gaussian_filter(T_s[1], sigma=(2, 2))

T1 = T.copy()

### EXPERIMENT 2

time = np.linspace(-0.1,1.1980,650)
imp_i = 650
data = loadmat(root_dir + "early_late_smoothed_2factor_single_trials.mat")


dec_early1_sub = data['dec_early1_sub'][0]
dec_early2_sub = data['dec_early2_sub'][0]
dec_late1_sub = data['dec_late1_sub'][0]
dec_late2_sub = data['dec_late2_sub'][0]

all_early = []
all_late= []

for i,d in enumerate(dec_early2_sub):
    s1= np.mean(dec_early1_sub[i][:,imp_i-100:imp_i],1)
    s2 = np.mean(dec_early2_sub[i][:,imp_i-100:imp_i],1)
    #all_early.append(np.concatenate([s1,s2]))
    all_early.append(s1); all_early.append(s2)

    s1= np.mean(dec_late1_sub[i][:,imp_i-100:imp_i],1)
    s2 = np.mean(dec_late2_sub[i][:,imp_i-100:imp_i],1)
    #all_late.append(np.concatenate([s1,s2]))
    all_late.append(s1); all_late.append(s2)


all_early = np.array(all_early)
all_late = np.array(all_late)

all_early_ori = all_early.copy()
all_late_ori = all_late.copy()


subjects = range(5,38,1) 
trials = range(50,750,750//20) # //20
T = np.zeros((2,len(trials),len(subjects)))

idx_subjects = list(range(19))
for nt,n_trials in enumerate(trials):
    print(nt)
    for _ in range(nboots):
        np.random.shuffle(idx_subjects)
        for ns,n_subjects in enumerate(subjects):

            early_subjects = all_early[idx_subjects[:n_subjects]]
            late_subjects = all_late[idx_subjects[:n_subjects]]

            # shuffle across trials
            [np.random.shuffle(c) for c in early_subjects]
            [np.random.shuffle(c) for c in late_subjects]

            # early_trials = [np.mean(l[:n_trials]) for l in early_subjects]
            # late_trials = [np.mean(l[:n_trials]) for l in late_subjects]

            early_trials = np.concatenate([l[:n_trials]for l in early_subjects])
            late_trials = np.concatenate([l[:n_trials] for l in late_subjects])

            # Bayes factor
            robjects.globalenv["early_trials"] = np.array(early_trials)
            bf = r('ttestBF(early_trials)[1]@bayesFactor$bf')
            T[0,nt,ns] += np.exp(bf[0])/nboots

            robjects.globalenv["late_trials"] = np.array(late_trials)
            bf = r('ttestBF(late_trials)[1]@bayesFactor$bf')
            T[1,nt,ns] += np.exp(bf[0])/nboots

T_s = T.copy()

T[0] = gaussian_filter(T_s[0], sigma=(2, 2))
T[1] = gaussian_filter(T_s[1], sigma=(2, 2))


plt.figure(figsize=(11,2))

plt.subplot(1,4,1)
plt.title("attended") # [0.7-1.2 s]
plt.imshow(1/T[0],aspect="auto", 
               vmin=-.5, vmax=5.5,origin='lower',extent=[subjects[0],subjects[-1],trials[0],trials[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()           
cbar.set_label('Bayes Factor', rotation=270)  
cbar.set_ticks([0,3])
#cbar.ax.set_yticklabels(["No evidence", "Evidence for absense of code"])  # vertically oriented colorbar
#contours = plt.contour(subjects1, trials1,1/T[0], [1], colors='black',linewidths=1)

plt.xlabel("number of subjects")
plt.ylabel("erp decoding\nnumber of trials")
plt.tick_params(left = False,bottom = False)

plt.subplot(1,4,2)
plt.title("unattended")
plt.imshow(1/T[1],aspect="auto", 
                vmin=-.5, vmax=5.5,origin='lower',extent=[subjects[0],subjects[-1],trials[0],trials[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()           
cbar.set_label('Bayes Factor', rotation=270)  
cbar.set_ticks([1,3])
cbar.ax.set_yticklabels(["No evidence", "Evidence for absense of code"])  # vertically oriented colorbar
contours = plt.contour(subjects, trials,1/T[1], [1,1.5,2,2.5,3], colors='black',linewidths=1)

plt.xlabel("number of subjects")
plt.tick_params(left = False,bottom = False)
plt.yticks([])


plt.subplot(1,4,3)
plt.title("attended (cued) ") #[1.65-2.15 s]
plt.imshow(1/T1[0],aspect="auto", 
               vmin=-.5, vmax=5.5,origin='lower',extent=[subjects1[0],subjects1[-1],trials1[0],trials1[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()     
cbar.set_ticks([0,3])
#cbar.ax.set_yticklabels(["No evidence", "Evidence for absense of code"])  # vertically oriented colorbar
#contours = plt.contour(subjects1, trials1,1/T1[0], [1], colors='black',linewidths=1)

plt.xlabel("number of subjects")
plt.tick_params(left = False,bottom = False)
plt.yticks(range(300,1000,200))
plt.xticks(range(subjects[0],30,10))

plt.subplot(1,4,4)
plt.title("discarded")
plt.imshow(1/T1[1],aspect="auto", 
                vmin=-.5, vmax=5.5,origin='lower',extent=[subjects1[0],subjects1[-1],trials1[0],trials1[-1]],interpolation="nearest",cmap=cmap)
cbar = plt.colorbar()           
cbar.set_ticks([0,3])
#contours = plt.contour(subjects1, trials1,1/T1[1], [3], colors='black',linewidths=1)
contours = plt.contour(subjects1, trials,1/T[1], [1,1.5,2,2.5,3], colors='black',linewidths=1)

plt.xlabel("number of subjects")
plt.tick_params(left = False,bottom = False)
plt.yticks([])
plt.xticks(range(subjects[0],30,10))

plt.savefig("figures/BF_erp_power_analyses.svg")
plt.show()


# ## time-resolved bayes factor
# time2 = loadmat('../decoders/exp2_dec_mem_late_time.mat')["time"][0]
# time1 = np.linspace(-.1,2.646,1374)
# ## load full time course early/late
# root_dir = "../baseline_decoders/single_trials2017/"

# all_early = []
# all_late= []

# for sub in range(1,20):
#     print(sub)
#     d_early=[]
#     d_late=[]
#     for session in [1,2]:
#         # load single-trial decoders, early
#         files=glob.glob(root_dir+"early_d_%i_single_dec_%i.mat" % (session, sub))[0]
#         dec_early=loadmat(files)["dec_early%i" % session]
#         d_early+=list(dec_early)

#         # load single-trial decoders, late
#         files=glob.glob(root_dir+"late_d_%i_single_dec_%i.mat" % (session, sub))[0]
#         dec_late=loadmat(files)["dec_late%i" % session]
#         d_late+=list(dec_late)
    
#     all_late.append(np.mean(d_late,0))
#     all_early.append(np.mean(d_early,0))

# all_early = np.array(all_early)
# all_late = np.array(all_late)



# ## load full time course cued/uncued
# all_cued = []
# all_uncued= []
# root_dir = "../baseline_decoders/single_trials2017/exp1/"

# for sub in range(1,30):
#     print(sub)

#     # load single-trial decoders
#     files=glob.glob(root_dir+"cued_exp1_single%i.mat" % (sub))[0]
#     dec_cued=loadmat(files)["cued_trials"]
#     all_cued.append(np.mean(dec_cued,0))

#     # load single-trial decoders
#     files=glob.glob(root_dir+"uncued_exp1_single%i.mat" % (sub))[0]
#     dec_uncued=loadmat(files)["uncued_trials"]
#     all_uncued.append(np.mean(dec_uncued,0))

# all_cued = np.array(all_cued)
# all_uncued = np.array(all_uncued)

# w=400

# def get_BF(decs):
# 	for i in range(len(decs)):
# 		decs[i,:] = np.mean(decs[i:i+w,:],0)

# 	BF = []
# 	for z in decs:
# 		robjects.globalenv["z"] = z
# 		bf = 1/np.exp(r('ttestBF(z)[1]@bayesFactor$bf'))
# 		BF.append(bf[0])
# 	BF = np.array(BF)
# 	return BF

# data = all_early,all_late,all_cued,all_uncued
# BF_data = []
# for i,d in enumerate(data):
# 	print(i)
# 	BF_data.append(get_BF(d.T))


# subplot(2,1,1)
# plt.plot(time1,np.ones_like(time1)*3)
# plt.plot(time2,BF_data[0],"black",label="attended")
# plt.plot(time1,BF_data[3],"blue",label="uncued")
# plt.plot(time2,BF_data[1],"red",label="unattended")
# plt.xlim(0,1.2)
# plt.legend()


# subplot(2,1,2)
# plt.plot(time1,np.zeros_like(time1))
# plt.plot(time2,np.mean(all_early,0),"black",label="attended")
# plt.plot(time2,np.mean(all_late,0),"red",label="unattended")
# plt.xlim(0,1.2)
# plt.legend()
