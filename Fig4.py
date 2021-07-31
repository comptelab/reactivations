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

find = lambda x: where(x)[0]
def sig_bar(sigs,axis,y,color):
	w=diff(axis)[0]
	for s in sigs:
		beg =axis[s]
		end = beg+w
		fill_between([beg,end],[y[0],y[0]],[y[1],y[1]],color=color)



sns.set_style("ticks")
sns.set_context("talk", font_scale=1.25)
sns.set_style({"ytick.direction": "in"})
sns.set_style({"xtick.direction": "in"})

## WOLFF DATA

imponset =0.25 + 0.95; 
all_files = sort(glob.glob("../Dynamic_hidden_states/Data/concat/exp2_d1*.mat"))
all_files2 = sort(glob.glob("../Dynamic_hidden_states/Data/concat/exp2_d2*.mat"))
f=io.loadmat(all_files[0])

time = f["data1"]["time"][0][0][0]

imp_i = argmin(abs(time-imponset))

s_vars = []

for f,file in enumerate(all_files):
	f1=io.loadmat(file)
	f2=io.loadmat(all_files2[f])
	data1 = f1["data1"]["trial"][0][0]
	data1 = data1 - np.mean(data1,1)[:,None,:]
	data2 = f2["data2"]["trial"][0][0]
	data2 = data2 - np.mean(data2,1)[:,None,:]
	data = concatenate((data1,data2))
	var=std(data,0)**2
	s_vars.append(mean(var,0))


s_vars_dt = detrend(s_vars,1)
base_line_dt = mean(s_vars_dt[:,imp_i-100:imp_i],1)
base_line = mean(array(s_vars)[:,imp_i-100:imp_i],1)

s_vars_dt_rel = [(s_vars_dt[:,t] - base_line_dt)/base_line * 100 for t in range(len(time))]


m_var_st = mean(s_vars_dt_rel,1)
stderr_st = 2*std(s_vars_dt_rel,1)/sqrt(len(s_vars_dt_rel[0]))
low_st = m_var_st-stderr_st
high_st = m_var_st+stderr_st
ps_st = array([ttest_1samp(v,0)[1] for v in s_vars_dt_rel])


# Wolff 2015 data

all_files = glob.glob("../Wolff2015/eeg_*.mat")

f=io.loadmat(all_files[0])

time2015 = io.loadmat("../Wolff2015/time.mat")['t'][0]

s_vars2015 = []
for file in all_files:
	f=io.loadmat(file)
	data2015 = f['eeg']
	data2015 = data2015 - np.mean(data2015,1)[:,None,:]
	var2015=std(data2015,0)**2
	s_vars2015.append(mean(var2015,0))

s_vars_2015_dt = detrend(s_vars2015,1)
base_line2015 = mean(array(s_vars2015)[:,25:50],1)
base_line2015_dt = mean(s_vars_2015_dt[:,25:50],1)

s_vars_2015_rel = [(s_vars_2015_dt[:,t] - base_line2015_dt)/base_line2015 * 100 for t in range(len(time2015))]

m_var2015 = mean(s_vars_2015_rel,1)
stderr2015 = std(s_vars_2015_rel,1)/sqrt(len(s_vars_2015_rel[0]))
low2015 = m_var2015-2*stderr2015
high2015 = m_var2015+2*stderr2015


ps_st2015 = array([ttest_1samp(v,0)[1] for v in s_vars_2015_rel])

## ROSE DATA
file_path = "../Rose et al/TMS_Data/Exp2"
all_files = glob.glob(file_path+"/*.mat")
#subjs =  unique([int(a.split("/")[4].split("_")[0]) for a in all_files])
subjs = [104, 105, 106, 107, 108, 111]

all_vars = []
for sub in subjs:
	files = glob.glob("%s/%s*.mat" % (file_path,sub))
	s_vars = []
	s_w_vars = []
	for file in files:
		f=io.loadmat(file)
		good_chan = f["goodchannels"][0]-1
		data = f["data"]
		times = f["times"][0]
		d=data[good_chan]
		m_sens = mean(d,0)
		d = [d1-m_sens for d1 in d]
		var = std(d,-1)**2
		baseline_tms = mean(var[:,:200:250],1)
		norm_var = [(var[:,t] - baseline_tms) / baseline_tms * 100 for t in range(len(times))]
		s_vars.append(mean(norm_var,1))
	all_vars.append(mean(s_vars,0))

ps_rose = array([ttest_1samp(v,0)[1] for v in array(all_vars).T])
m_var =mean(all_vars,0)
stderr = 2*std(all_vars,0)/sqrt(len(all_vars))
low = m_var-stderr
high = m_var+stderr



m_fr = loadtxt("simulations_for_plot/wolff_sims_fr.txt")
time_stk = loadtxt("simulations_for_plot/wolff_sims_time.txt")
time_stk -= 0.3/2
m_fr = m_fr[mean(m_fr,1)>0]


baseline = mean(std(m_fr[:,:5],0)/mean(m_fr[:,:5],0))
idx = bootstrap.bootstrap_indexes(m_fr)


diff_ff_nostp = loadtxt("simulations_for_plot/diff_ff_sim_nostp.txt")
time_nostp = loadtxt("simulations_for_plot/time_sim_nostp.txt")


boot = [(std(m_fr[i],0)/mean(m_fr[i],0) - baseline) for i in idx]

#### var split

# wolff 2015 trial by trial analyses
root_dir = "../Wolff2015/Data/trial_by_trial/"

time2015 = io.loadmat("../Wolff2015/time.mat")['t'][0]

eegs_imp=[]

T = 6
idx_imp=range(147-T,147+T)


for sub in range(1,25):
	dec_bins_imp=[]
	eeg_bins_imp=[]
	for bin_n in [1,2]:
		dec_pairs_imp = []
		eeg_pairs_imp = []
		for pair in [1,2]:


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
			eeg_pairs_imp.append(diff_imp)


		eeg_bins_imp.append(mean(eeg_pairs_imp,0))


	eegs_imp.append(mean(eeg_bins_imp,0))

ci_split = np.array([bootstrap.ci(e,n_samples=1000) for e in np.array(eegs_imp).T])
stderr_split = np.array([bootstrap.ci(e,n_samples=1000,alpha=1-0.69) for e in np.array(eegs_imp).T])

plt.figure(figsize=(10,10))

subplot(2,2,1)
title("Wolff et al (2017)\nWolff et al (2015)",pad=10)

fill_between(time-imponset,low_st,high_st,alpha=0.5,color="gray")
#plot(time-imponset,(array(s_vars_dt)),"k",alpha=0.15)
plot(time-imponset,m_var_st,"gray",lw=2)
plot(time-imponset,zeros(len(time)),"--",color="gray")

fill_between(time2015,low2015,high2015,alpha=0.5,color="black")
plot(time2015, m_var2015, "black")


sig_bar(find(ps_st<0.005),time-imponset,[40*0.95,40],"gray")
sig_bar(find(ps_st2015<0.005),time2015,[40*0.94*0.94,40*0.94],"black")


xlim(-0.2,.4)
ylim(-40,40)
yticks([-40,-20, 0,20,40])
ylabel(r"$\Delta$" "variance (%)")
xlim(-0.2,0.5)
xticks([0,0.5],[])

subplot(2,2,2)
title("Rose et al (2018)",pad=10)
fill_between(times/1000,low,high,alpha=0.5,color="k")
#plot(times/1000,(array(all_vars)).T,"k",alpha=0.2)
plot(times/1000,m_var,"k",lw=2)
plot(times/1000,zeros(len(times)),"k--")
sig_bar(find(ps_rose<0.005),times/1000,[40*0.95,40],"k")
ylim(-40,40)
#ylabel("% of " r"$\Delta$" "variance")
yticks([-40,-20,0,20,40])
xlim(-0.2,0.5)
xticks([0,0.5],[])

subplot(2,2,3)
title("variability median-split by \n reactivation decoding (2015)",pad=10)
sig_bar(find(ci_split[:,0]>0),time2015,[0.5*0.94,0.5],"black")
fill_between(time2015,ci_split[:,0],ci_split[:,1],alpha=0.5,color="black")
plot(time2015, mean(eegs_imp,0),color="black")
plot([time2015[0],time2015[-1]],[0,0],"k--")
plot(0.38799999999999996,0,"*")
xlim(-0.2, 0.5)
ylim(-0.5, 0.5)
xticks([0,0.5],["0", ".5"])
yticks([-0.5, 0,0.5])
xlabel("time from impulse (s)")


ax1 = subplot(2,2,4)
title("Model",pad=10)
fill_between(time_stk,mean(boot,0)-std(boot,0)*2,mean(boot,0)+std(boot,0)*2,alpha=0.5,color="k")
plot(time_stk,mean(boot,0),"k",lw=2,label="reactivations")
ylabel(r"$\Delta$""fano-factor")
ylim([-2,2])
yticks([-2,-1, 0, 1, 2])
plot(time_stk,zeros(len(time_stk)),"k--")
xlim(-0.2,0.5)
xticks([0,0.5],["0", ".5"])
xlabel("time from impulse (s)")

ax2 = ax1.twinx() 
ax2.plot(time_nostp-2,np.percentile(diff_ff_nostp,50,0),color="gray",label="no reactivations")
ax2.fill_between(time_nostp-2,np.percentile(diff_ff_nostp,5,0),np.percentile(diff_ff_nostp,95,0),alpha=0.5,color="gray")
ax2.set_ylim([-.2,.2])
ax2.tick_params(axis='y', labelcolor="gray")

sns.despine(right=False)
plt.tight_layout()
#ax1.legend(frameon=False)
#ax2.legend(frameon=False)

savefig("figures/var_col.png",dpi=300)
savefig("figures/var_col.svg",dpi=300)
plt.show()


