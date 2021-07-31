from __future__ import division
from matplotlib.pylab import *
import scipy.io as io
import glob
import sys
from scipy.stats import *
from scipy.stats.mstats_basic import pearsonr, spearmanr
import seaborn as sns
from scikits import bootstrap
from scipy.signal import detrend
import mat73
from scipy.io import loadmat

find = lambda x: where(x)[0]
def sig_bar(sigs,axis,y,color):
	w=diff(axis)[0]
	for s in sigs:
		beg =axis[s]
		end = beg+w
		fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)


sns.set_style("ticks")
sns.set_context("talk", font_scale=1.5)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})

## WOLFF DATA

imponset =0.25 + 0.95; 
all_files = sort(glob.glob("../Dynamic_hidden_states/Data/concat/exp2_d1*.mat"))
all_files2 = sort(glob.glob("../Dynamic_hidden_states/Data/concat/exp2_d2*.mat"))

all_files1_alpha = sort(glob.glob("../Dynamic_hidden_states/Data/concat/alpha_exp2_d1*.mat"))
all_files2_alpha= sort(glob.glob("../Dynamic_hidden_states/Data/concat/alpha_exp2_d2*.mat"))


all_files1_stim= sort(glob.glob("../Dynamic_hidden_states/Data/concat/stim_exp2_d1*.mat"))
all_files2_stim= sort(glob.glob("../Dynamic_hidden_states/Data/concat/stim_exp2_d2*.mat"))

f=io.loadmat(all_files[0])

time = f["data1"]["time"][0][0][0]

imp_i = argmin(abs(time-imponset))

all_avg_early = []
all_avg_late = []
all_corrs = []
all_ve = []
all_vl = []
all_shuffle_corrs = []

for f,file in enumerate(all_files):
    print(f)

    #laod stims 
    f1=io.loadmat(all_files1_stim[f])
    f2=io.loadmat(all_files2_stim[f])

    stims1 = io.loadmat(all_files1_stim[f])["mem_angles1"]
    stims2 = io.loadmat(all_files2_stim[f])["mem_angles2"]

    # load ERP
    f1=io.loadmat(file)
    f2=io.loadmat(all_files2[f])
    data1_alpha = f1["data1"]["trial"][0][0]
    data1_alpha = data1_alpha - np.mean(data1_alpha,1)[:,None,:]

    data2_alpha = f2["data2"]["trial"][0][0]
    data2_alpha = data2_alpha - np.mean(data2_alpha,1)[:,None,:]

    #load alpha - comment to analyse ERPs instead
    f1=io.loadmat(all_files1_alpha[f])
    f2=io.loadmat(all_files2_alpha[f])
    data1_alpha = f1["pow1"] 
    data1_alpha -= np.mean(data1_alpha,1)[:,None,:]

    data2_alpha = f2["pow2"]
    data2_alpha -= np.mean(data2_alpha,1)[:,None,:]

    # session to use
    d1 = np.mean(data1_alpha[:,:,(time<1.2) & (time>0.7)],-1)
    d2 = np.mean(data2_alpha[:,:,(time<1.2) & (time>0.7)],-1)

    data_alpha = np.concatenate((d1,d2))
    data_alpha = d1

    stims = np.concatenate((stims1,stims2))
    stims = stims1
    _,_,idx1 = binned_statistic(stims[:,0],stims[:,1],bins=8)
    _,_,idx2 = binned_statistic(stims[:,1],stims[:,1],bins=8)

    def one_corr():
        avg_alpha_early = []
        avg_alpha_late = []
        for i in np.unique(idx1):
            avg1 = np.mean(data_alpha[idx1 == idx1[i]],0)
            avg2 = np.mean(data_alpha[idx2 == idx2[i]],0)
            avg_alpha_early.append(avg1)
            avg_alpha_late.append(avg2)

        early = np.array(avg_alpha_early - np.mean(avg_alpha_early,0))
        late = np.array(avg_alpha_late- np.mean(avg_alpha_late,0))

        # r_early = [roll(early[:,i],argmax(early[:,i])+4) for i in range(17)]
        # r_late = [roll(late[:,i],argmax(early[:,i])+4) for i in range(17)]

        #return early, late
        corr_tuning = [spearmanr(early[:,i],late[:,i]) for i in range(17)]
        corr_elec = spearmanr(np.std(avg_alpha_early,0),np.std(avg_alpha_late,0))[0]
        
        return [np.mean(corr_tuning),corr_elec]

    corrs = one_corr()
    shuffle_corrs = []
    for _ in range(500):
        shuffle(idx1)
        shuffle(idx2)
        shuffle_corrs.append(one_corr())

    all_corrs.append(corrs)
    all_shuffle_corrs.append(shuffle_corrs)




z_sub = (np.array(all_corrs)[:,0] - np.mean(np.array(all_shuffle_corrs)[:,:,0],1)) / np.std(np.array(all_shuffle_corrs)[:,:,0],1)
print("tuning ",ttest_1samp(z_sub,0))


z_sub = (np.array(all_corrs)[:,1] - np.mean(np.array(all_shuffle_corrs)[:,:,1],1)) / np.std(np.array(all_shuffle_corrs)[:,:,1],1)
print("electrodes ",ttest_1samp(z_sub,0))


plt.subplot(2,1,1)
plt.imshow(np.mean(all_corrs,0)[0].T)

plt.subplot(2,1,2)
plt.imshow(np.mean(all_corrs,0)[1].T)
