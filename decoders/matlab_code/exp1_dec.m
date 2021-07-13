% Script containing analyses for Figures 2f, 3a, and 3b of "Dynamic hidden
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
save_dir = '/Users/jbarbosa/Dropbox/Neuro/papers/Reactivation/baseline_decoders/single_trials2017';
%
for sub=1:30

    memtic;
    tic;
    fprintf(['Doing ' num2str(sub) '\n'])
    
    load(fullfile(dat_dir,['Dynamic_hidden_states_exp1_' num2str(sub) '.mat']));
    
    cueonset  = exp1_data.EEG_mem_items.time(end)+1/exp1_data.EEG_mem_items.fsample;
    imponset  = exp1_data.EEG_mem_items.time(end)+1/exp1_data.EEG_mem_items.fsample+exp1_data.EEG_cue.time(end)+1/exp1_data.EEG_mem_items.fsample;

    data = exp1_data.EEG_mem_items;
    Results = exp1_data.Results;
        
    [out, dc1, good, bad] = catWolff(exp1_data.EEG_mem_items.trial,exp1_data.EEG_cue.trial,1:50);
    [out2, dc2, good2, bad2] = catWolff(out,exp1_data.EEG_impulse.trial,1:50);
    
    % here we recorver one sample
    data.fsample = exp1_data.EEG_mem_items.fsample;
    data.trial = zeros(size(out2,1),size(out2,2),size(out2,3)+1);
    data.trial(good2,:,:) = cat(3,out(good2,:,1), out2(good2,:,:));
    data.trial(bad2,:,:)  = cat(3,out(bad2,:,2),  out2(bad2,:,:));
    clear out out2 dc1 dc2;
    
    data.time = -0.1:1/data.fsample:(size(data.trial,3)/data.fsample)-0.1-1/data.fsample;
    time = data.time;
    badtrl = unique([exp1_data.EEG_mem_items.bad_trials exp1_data.EEG_cue.bad_trials' exp1_data.EEG_impulse.bad_trials']);
    trl = ones(1,size(data.trial,1));
    trl(badtrl) = 0;
    incl = find(trl==1);
    clear exp1_data;
    
    fsample = data.fsample;
    
%     cfg = [];
%     cfg.trials = incl;
%    data = ft_selectdata(cfg,data);
    
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
    cfg.trials    = incl;
    freq          = ft_freqanalysis(cfg,data);
    
    
    data.trial = abs(squeeze(freq.fourierspctrm)).^2;
    
    data.trial= bsxfun(@minus, data.trial, mean(data.trial,2)); % mean center voltage across channels to normalize
    
    mem_angles=Results(incl,1:2)*2; % extract memory item angles and rescale
    cue=Results(incl,3);
    acc=Results(incl,5);
    probe_rotation=Results(incl,4);
    
    cued_mem_left=mem_angles(cue==1,1); %orientations of cued items on the left
    cued_mem_right=mem_angles(cue==2,2); %orientations of cued items on the right
    
    uncued_mem_left=mem_angles(cue==2,1); %orientations of uncued items on the left
    uncued_mem_right=mem_angles(cue==1,2); %orientations of uncued items on the right
    
    dec_cued_left = mahalTune_func(data.trial(cue==1,:,:),cued_mem_left,angspace,bin_width); % decode cued items on the left
    dec_cued_right = mahalTune_func(data.trial(cue==2,:,:),cued_mem_right,angspace,bin_width); % decode cued items on the right
    
    %dec_imp_cued(sub,:)=mean(cat(1,dec_cued_left,dec_cued_right),1); % average over trials and cued locations and smooth over time
    
    dec_uncued_left = mahalTune_func(double(data.trial(cue==2,:,:)),uncued_mem_left,angspace,bin_width); % decode uncued items on the left
    dec_uncued_right = mahalTune_func(double(data.trial(cue==1,:,:)),uncued_mem_right,angspace,bin_width); % decode uncued items on the right
    %dec_imp_uncued(sub,:)=mean(cat(1,dec_uncued_left,dec_uncued_right),1);
    
    cued_trials = cat(1,dec_cued_left,dec_cued_right);
    uncued_trials = cat(1,dec_uncued_left,dec_uncued_right);
    
    save(fullfile(save_dir,['exp1/alpha_cued_exp1_single' num2str(sub) '.mat']),'cued_trials');
    save(fullfile(save_dir,['exp1/alpha_uncued_exp1_single' num2str(sub) '.mat']),'uncued_trials');
    
end



