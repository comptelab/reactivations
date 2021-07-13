
from __future__ import division
from matplotlib.pylab import *
import scipy.io as io
import glob
import sys
from scipy.stats import *
import seaborn as sns
from scikits import bootstrap
sys.path.insert(0, '/home/joaob/Dropbox/Neuro/mypytools/')
sys.path.insert(0, '/Users/jbarbosa/Dropbox/Neuro/mypytools/')
from helpers import *
from scipy.signal import detrend
#import mat73
from scipy.io import loadmat
from scipy.stats import sem


sns.set_style("ticks")
sns.set_context("talk", font_scale=1)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})


# wolff 2015 trial by trial analyses
root_dir = "Wolff2015/Data/trial_by_trial/"

time2015 = io.loadmat("Wolff2015/time.mat")['t'][0]
time2015_delay = io.loadmat("Wolff2015/time_delay.mat")['time'][0]

decs_imp = []
eegs_imp=[]
decs_delay = []
eegs_delay=[]
high_low=[]

T= 6
idx_imp=range(147-T,147+T)

idx_delay=range(75,155)
idx_delay=range(80,160)
idx_delay=range(88-T,88+T)
all_decs=[]
all_eeg=[]

for sub in range(1,25):
	dec_bins_imp=[]
	eeg_bins_imp=[]
	dec_bins_delay=[]
	eeg_bins_delay=[]
	all_decs_bins=[]
	all_eeg_bins=[]
	for bin_n in [1,2]:
		dec_pairs_imp = []
		eeg_pairs_imp = []
		dec_pairs_delay = []
		eeg_pairs_delay = []
		all_decs_pairs=[]
		all_eeg_pairs=[]
		for pair in [1,2]:

			# Delay
			# load single-trial decoders
			files=glob.glob(root_dir+"maha*pair%ibin%i*Delay%i.mat" % (pair, bin_n,sub))[0]
			dec_delay=loadmat(files)["d"]

			# select point where to make the split
			m_dec_delay = mean(dec_delay[:,idx_delay],1)
			#m_dec_delay = amax(dec_delay,1)
	
			# do median split
			l_idx = m_dec_delay < percentile(m_dec_delay,50)
			h_idx = m_dec_delay > percentile(m_dec_delay,50)

			# load trial-by-trial voltage traces			
			files=glob.glob(root_dir+"eeg*pair%ibin%i*Delay%i.mat" % (pair, bin_n,sub))[0]
			eeg_delay=loadmat(files)["dat1_dat2"]

			# compute variability for each median split
			eeg_delay_low_dec=std(eeg_delay[l_idx],0)
			eeg_delay_high_dec=std(eeg_delay[h_idx],0)
			# compute the difference
			diff_delay = mean(eeg_delay_low_dec,0) -  mean(eeg_delay_high_dec,0)

			# Impulse, same as delay but for impulse period
			# load single-trial decoders
			files=glob.glob(root_dir+"maha*pair%ibin%i*Impulse%i.mat" % (pair, bin_n,sub))[0]
			dec_imp=loadmat(files)["d"]
			
			# select point where to make the split
			m_dec_imp = mean(dec_imp[:,idx_imp],1)
			#m_dec_imp = amax(dec_imp[:,:200],1)
			
			# do median split
			l_idx = m_dec_imp < percentile(m_dec_imp,50)
			h_idx = m_dec_imp > percentile(m_dec_imp,50)

			# load trial-by-trial voltage traces
			files=glob.glob(root_dir+"eeg*pair%ibin%i*Impulse%i.mat" % (pair, bin_n,sub))[0]
			eeg_imp=loadmat(files)["dat1_dat2"]

			# compute variability for each median split
			eeg_imp_low_dec=std(eeg_imp[l_idx],0)
			eeg_imp_high_dec=std(eeg_imp[h_idx],0)			
			# compute the difference
			diff_imp = mean(eeg_imp_low_dec,0) -  mean(eeg_imp_high_dec,0)

			dec_pairs_imp.append(dec_imp)
			eeg_pairs_imp.append(diff_imp)

			dec_pairs_delay.append(dec_delay)
			eeg_pairs_delay.append(diff_delay)
			all_decs_pairs.append([m_dec_delay, m_dec_imp])

			m_dl = mean(eeg_delay,1) # average across sensors
			m_imp = mean(eeg_imp,1) # average across sensors
			
			# least squares distance, for all time points. single trial proxy of variance. if averaged, its actually variance
			dl_sq_dist = (m_dl - mean(m_dl,0))**2 
			imp_sq_dist = (m_imp - mean(m_imp,0))**2
			all_eeg_pairs.append([mean(dl_sq_dist[:,idx_delay],1), mean(imp_sq_dist[:,idx_imp],1)])

		all_decs_bins.append(all_decs_pairs)
		all_eeg_bins.append(all_eeg_pairs)

		dec_bins_imp=concatenate(dec_pairs_imp)
		eeg_bins_imp.append(mean(eeg_pairs_imp,0))

		dec_bins_delay=concatenate(dec_pairs_delay)
		eeg_bins_delay.append(mean(eeg_pairs_delay,0))

	decs_imp.append(mean(dec_bins_imp,0))
	eegs_imp.append(mean(eeg_bins_imp,0))

	decs_delay.append(mean(dec_bins_delay,0))
	eegs_delay.append(mean(eeg_bins_delay,0))
	all_decs.append(all_decs_bins)
	all_eeg.append(all_eeg_bins)


for i in [0,1]:
	eegs = [eegs_delay,eegs_imp][i]
	time = [time2015_delay,time2015][i]
	decs = [decs_delay,decs_imp][i]
	idx = [idx_delay,idx_imp][i]
	figure(figsize=(5,5))
	subplot(2,1,1)
	errorbar(time, mean(eegs,0),2*sem(eegs,0))
	fill_between([time[idx[0]],time[idx[-1]]],[-0.2,-0.2],[0,0],alpha=0.2,color="green")
	plot([time[0],time[-1]],[0,0])
	ylim(-0.5, 0.5)
	subplot(2,1,2)
	errorbar(time,mean(decs,0),2*sem(decs,0))
	fill_between([time[idx[0]],time[idx[-1]]],[0,0],[0.01,0.01],alpha=0.2,color="green")
	xlim(-0.2,0.8)


all_eeg = array(all_eeg)
all_decs = array(all_decs)

delay_decs = all_decs[:,:,:,0]
imp_decs = all_decs[:,:,:,1]

delay_eeg = all_eeg[:,:,:,0]
imp_eeg = all_eeg[:,:,:,1]



d_decs = [zscore(concatenate((concatenate(sub))))  for sub in delay_decs]
d_eegs = [zscore(concatenate((concatenate(sub))))  for sub in delay_eeg]
i_decs = [zscore(concatenate((concatenate(sub))))  for sub in imp_decs]
i_eegs = [zscore(concatenate((concatenate(sub))))  for sub in imp_eeg]

ttest_1samp([spearmanr(i_decs[s],i_eegs[s])[0] for s in range(24)],0)
ttest_1samp([spearmanr(d_decs[s],d_eegs[s])[0] for s in range(24)],0)

all_ieeg, all_idec = concatenate(i_eegs),concatenate(i_decs)
all_deeg, all_ddec = concatenate(d_eegs),concatenate(d_decs)


pearsonr(all_ieeg, all_idec)
pearsonr(all_deeg, all_ddec)

def perm_test(n_perms):
	r1=pearsonr(all_ieeg, all_idec)[0]
	r2=pearsonr(all_deeg, all_ddec)[0]
	r_diff = r1-r2
	n_trials = len(all_idec)
	p_diff = []
	idx = range(n_trials*2)
	for _ in range(n_perms):
		both_eeg = concatenate([all_ieeg,all_deeg])
		both_dec = concatenate([all_idec,all_ddec])
		shuffle(idx)
		r1=pearsonr(both_eeg[idx[:n_trials]],both_dec[idx[:n_trials]])[0]
		r1=pearsonr(both_eeg[idx[n_trials:]],both_dec[idx[n_trials:]])[0]
		p_diff.append(r1-r2)
	return r_diff,p_diff

a=perm_test(1000)
print(mean(a[0]>array(a[1])))



idx = bootstrap.bootstrap_indexes(all_ieeg,n_samples=1000)

corr_i = [pearsonr(all_ieeg[i], all_idec[i])[0] for i in idx]
corr_d = [pearsonr(all_deeg[i], all_ddec[i])[0] for i in idx]

ci_i = percentile(corr_i,[2.5,97.5])
ci_d = percentile(corr_d,[2.5,97.5])

yerr = [[ci_d[0]-mean(corr_d),ci_d[1]+mean(corr_d)],[ci_i[0]-mean(corr_i),ci_i[1]+mean(corr_i)]]
errorbar([0,0.1],[mean(corr_d),mean(corr_i)],yerr)
