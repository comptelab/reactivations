from __future__ import division
from numpy import *
from numpy.random import randn
from matplotlib.pylab import *
import seaborn as sns
import pylustrator

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Bitstream Vera Sans']

sns.set_context("talk", font_scale=1)
sns.set_style("ticks")


sigma = 0.1
mu = 0
n_trials = 1000
n_time = 100

slope1=1/40
slope2=1/80
slope_flash=1/20

trials1 = sigma*randn(n_trials,int(n_time/2),2)+mu
trials2 = 0.5*sigma*randn(n_trials,int(n_time/4),2)+mu
trials2_no_drop = sigma*randn(n_trials,int(n_time/4),2)+mu
trials3 = sigma*randn(n_trials,int(n_time/4),2)+mu

code1 = arange(n_time)[::-1]*slope1
code2 = arange(n_time)[::-1]*slope2

trials = concatenate([trials1,trials2,trials3],1)
trials[:,:,0]+=code1
trials[:,:,1]+=code2

trials2_no_drop[:,:,0] += code1[:25]*0.25
trials_no_drop = concatenate([trials1,trials2_no_drop,trials3],1)
trials_no_drop[:,:,0]+=code1
trials_no_drop[:,:,1]+=code2

def snr(trials):
	m=mean(trials,0)
	s=std(trials,0)
	sr = abs(m[:,0]-m[:,1])/(sqrt(s[:,0]*s[:,1]))
	return sr

plt.figure(figsize=(9,6))



plt.subplot(2,3,1)
plt.title("signal in\nneural activity")
plt.ylabel(r"$\mu$",rotation=0,fontsize=25)
plt.plot(trials[1,:,0],"k")
plt.plot(trials[1,:,1],"gray")
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,2)
plt.title("noise")
plt.ylabel(r"$\sigma$",rotation=0,fontsize=25)
plt.plot(sqrt(std(trials,0)[:,0]*std(trials,0)[:,1]),"k")
plt.ylim(0,.2)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,3)
plt.title("signal-to-noise")
plt.ylabel(r"$\frac{\mu}{\sigma}$",rotation=0,fontsize=25)
plt.plot(snr(trials),"k",ms=5,)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,4)
plt.ylabel(r"$\mu$",rotation=0,fontsize=25)
plt.plot(trials_no_drop[1,:,0],"k")
plt.plot(trials_no_drop[1,:,1],"gray")
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,5)
plt.plot(sqrt(std(trials_no_drop,0)[:,0]*std(trials_no_drop,0)[:,1]), "k")
plt.ylabel(r"$\sigma$",rotation=0,fontsize=25)
plt.xlabel("time")
plt.ylim(0,.2)
plt.xticks([])
plt.yticks([])

plt.subplot(2,3,6)
plt.plot(snr(trials_no_drop),"k",ms=5)
plt.ylabel(r"$\frac{\mu}{\sigma}$",rotation=0,fontsize=25)
plt.xticks([])
plt.yticks([])

sns.despine()

#plt.tight_layout()

plt.savefig("figures/simulation_drop_variance.svg")

show()