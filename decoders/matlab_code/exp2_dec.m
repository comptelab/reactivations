% Script containing analyses for Figure 5a of "Dynamic hidden
% states underlying working-memory-guided behavior", Nature Neuroscience,
% 2017

% Input is preprocessed (see article) and baselined (-200 to 0 ms relative
% to stimulus onset) EEG data.
addpath /home/joaob/Dropbox/Neuro/papers/Reactivation/DragonBalls/fieldtrip
ft_defaults

close all
clear all

main_dir='../../../Dynamic_hidden_states/'; % path to main folder containing all data, functions, toolboxes, scripts
addpath(genpath(main_dir))

dat_dir=[main_dir 'Data/'];
angspace=(-pi:pi/6:pi)'; % angular space for tuning curve (in radians)
angspace(end)=[];
bin_width=pi/6; % width of each angle bin of the tuning curve (in radians)

s_factor=8; % determines the SD of the smoothing kernel for the decoding time-courses,
            % depends on resolution of data (500 hz = 2ms, 2ms times 8 is thus an SD of 16 ms)
%%

for sub=1:19,
    fprintf(['Doing ' num2str(sub) '\n'])
    
    load(fullfile(dat_dir,['Dynamic_hidden_states_exp2_' num2str(sub) '.mat']));
    Results1=exp2_data.Results_sess1;
    Results2=exp2_data.Results_sess2;
    
    load(fullfile(dat_dir,['concat/exp2_d1_' num2str(sub) '.mat']));
    load(fullfile(dat_dir,['concat/exp2_d2_' num2str(sub) '.mat']));
    
    EEG_dat1 = data1;
    EEG_dat2 = data2;
    time1=EEG_dat1.time;
    time2=EEG_dat2.time;
    
    EEG1_dat = EEG_dat1;
    EEG2_dat = EEG_dat2;
   
    time=EEG_dat1.time;
    
    clear exp2_data
    
    incl1=not(ismember(1:size(EEG_dat1.trial,1),EEG_dat1.bad_trials))'; % logical array of trials to be included
    data1 = EEG_dat1.trial(incl1,:,:); 
    
    incl2=not(ismember(1:size(EEG_dat2.trial,1),EEG_dat2.bad_trials))'; % logical array of trials to be included
    data2 = EEG_dat2.trial(incl2,:,:); 
    
    %clear EEG_dat1 EEG_dat2
    
    data1= bsxfun(@minus, data1, mean(data1,2)); % mean center voltage across channels to normalize
    mem_angles1=Results1(incl1,1:2)*2; % extract memory item angles and rescale 
    mem_angles1=Results1(:,1:2)*2; % extract memory item angles and rescale 

    data2= bsxfun(@minus, data2, mean(data2,2)); % mean center voltage across channels to normalize
    mem_angles2=Results2(incl2,1:2)*2; % extract memory item angles and rescale
    mem_angles2=Results2(:,1:2)*2; % extract memory item angles and rescale

    cfg           = [];
    cfg.output    = 'fourier';
    cfg.channel   = 'all';
    cfg.method    = 'hilbert';
    cfg.foi       = [10];
    cfg.toi       = 'all';
    cfg.filttype  = 'fir';
    cfg.filtorder = (3/8)*512;
    cfg.filtdir   = 'twopass';
    cfg.width     = 2;
    cfg.pad       = 3;
    %cfg.trials    = incl1;
    freq          = ft_freqanalysis(cfg,EEG_dat1);

    pow1 = abs(squeeze(freq.fourierspctrm)).^2;

   
    %cfg.trials    = incl2;
    freq          = ft_freqanalysis(cfg,EEG_dat2);

    pow2 = abs(squeeze(freq.fourierspctrm)).^2;
    
        
    pow1= bsxfun(@minus, pow1, mean(pow1,2)); % mean center voltage across channels to normalize
    
    pow2= bsxfun(@minus, pow2, mean(pow2,2)); % mean center voltage across channels to normalize
    
%% FOR ERP
%     pow1=data1;
%     pow2=data2;
    
%    pow1=bsxfun(@minus,pow1,mean(pow1(:,:,650-100:650),3));        
%    pow2=bsxfun(@minus,pow2,mean(pow2(:,:,650-100:650),3));        
    
%    save EEG traces
%    save(['baseline_decoders/single_trials2017/eeg_single_trial_d1' num2str(sub) '.mat'],'pow1');
%    save(['baseline_decoders/single_trials2017/eeg_single_trial_d2' num2str(sub) '.mat'],'pow2');   
%     save(fullfile(dat_dir,['concat/alpha_exp2_d1_' num2str(sub) '.mat']),'pow1');
%     save(fullfile(dat_dir,['concat/alpha_exp2_d2_' num2str(sub) '.mat']),'pow2');
%     save(fullfile(dat_dir,['concat/stim_exp2_d1_' num2str(sub) '.mat']),'mem_angles1');
%     save(fullfile(dat_dir,['concat/stim_exp2_d2_' num2str(sub) '.mat']),'mem_angles2');


    %%run decoder
    dec_early1 = mahalTune_func(pow1,mem_angles1(:,1),angspace,bin_width); % decode early-tested item in session 1
    dec_late1 = mahalTune_func(pow1,mem_angles1(:,2),angspace,bin_width); % decode late-tested item in session 1
    
    dec_early2 = mahalTune_func(pow2,mem_angles2(:,1),angspace,bin_width); % decode early-tested item in session 2
    dec_late2 = mahalTune_func(pow2,mem_angles2(:,2),angspace,bin_width); % decode late-tested item in session 2 
    
    dec_mem_early(sub,:)=(mean(dec_early1,1)+mean(dec_early2,1))/2; % average over decoding values of each item and all trials and smooth the time-course
    dec_mem_late(sub,:)=(mean(dec_late1,1)+mean(dec_late2,1))/2;    
    
end


save('../exp2_dec_mem_late_theta.mat', 'dec_mem_late')
save('../exp2_dec_mem_early_theta.mat', 'dec_mem_early')