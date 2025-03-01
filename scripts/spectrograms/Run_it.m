% import
data = load('C:\noam\Ben_Gurion\year_3\pre_laplacian_data.mat');
labels = load('C:\noam\Ben_Gurion\year_3\pre_laplacian_labels.mat');
data = data.a;
labels = labels.a;

all = 1:length(labels);
right_ind = all(labels==0);
left_ind = all(labels==1);
idle_ind = all(labels==2);
% data idel
chanlocs =  {'C3', 'C4', 'Cz', 'FC1', 'FC2', 'FC5', 'FC6', 'CP1', 'CP2', 'CP5', 'CP6'};
chan1 = 1;
chan2 = 2;
data_cell{1,1}=data(right_ind,chan1,:); %Right
data_cell{1,2}=data(left_ind,chan1,:); %Left
data_cell{2,1}=data(right_ind,chan2,:);  %Right
data_cell{2,2}=data(left_ind,chan2,:);  %Left

Titles.RnL = {'right','left'};
Titles.Channels = {cell2mat(chanlocs(chan1)), cell2mat(chanlocs(chan2))};
Titles.BestFeat='Visualizition of the best features';
Titles.AllFeat ='ALL feature Visualizition';
Titles.BandPower ='Band Power Visualizition';
Titles.diff = 'The Difference Spectrograms between right-left';

FontSize.XL =18;
FontSize.L = 15;
FontSize.M = 13;
FontSize.S = 11;
FontSize.XS = 8;

SR = 125;
window=0.1*round(SR);%[samples]
noverlap=round(0.5*window);%[samples]

create_spectograms(data_cell, window, noverlap, SR, Titles,FontSize)

