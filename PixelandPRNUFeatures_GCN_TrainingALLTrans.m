clc
close all
clear

cd('S:\WVU_Multimodal\CODES')
addpath('Functions/')
addpath('Filter/')

qmf = MakeONFilter('Daubechies',8);
L = 4;

%% IPT1
 
imageDir = 'S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\IPT1';

cd(imageDir)
images = dir('*.bmp');
cd('S:\WVU_Multimodal\CODES')
for i=1:length(images)
    i
    filename = images(i).name;
    
    img = imresize(double(imread(fullfile(imageDir,filename))),[96,96]);
    img = img(:,:,1);
    
    PixelFeatures(i,:) = img(:);
    Noisex_fft = PhaseNoiseExtractFromImage_SUD_Enhanced(img,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);
    PRNUFeatures(i,:) = Noiseresidual_testimage(:);
    FileName{i} = filename;
end
    
cd('S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\FeatsIPT1')
save('PixelFeatures_IPT1_Res.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT1_Res.mat','PRNUFeatures')

clear images PixelFeatures PRNUFeatures FileName
%% IPT2
imageDir = 'S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\IPT2';

cd(imageDir)
images = dir('*.bmp');
cd('S:\WVU_Multimodal\CODES')
for i=1:length(images)
    i
    filename = images(i).name;
    
    img = imresize(double(imread(fullfile(imageDir,filename))),[96,96]);
    img = img(:,:,1);
    
    PixelFeatures(i,:) = img(:);
    Noisex_fft = PhaseNoiseExtractFromImage_SUD_Enhanced(img,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);
    PRNUFeatures(i,:) = Noiseresidual_testimage(:);
    FileName{i} = filename;
end
    
cd('S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\FeatsIPT2')
save('PixelFeatures_IPT2_Res.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT2_Res.mat','PRNUFeatures')

clear images PixelFeatures PRNUFeatures FileName
%% IPT3
 
imageDir = 'S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\IPT3';

cd(imageDir)
images = dir('*.bmp');
cd('S:\WVU_Multimodal\CODES')
for i=1:length(images)
    i
    filename = images(i).name;
    
    img = imresize(double(imread(fullfile(imageDir,filename))),[96,96]);
    img = img(:,:,1);
    
    PixelFeatures(i,:) = img(:);
    Noisex_fft = PhaseNoiseExtractFromImage_SUD_Enhanced(img,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);
    PRNUFeatures(i,:) = Noiseresidual_testimage(:);
    FileName{i} = filename;
end
    
cd('S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\FeatsIPT3')
save('PixelFeatures_IPT3_Res.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT3_Res.mat','PRNUFeatures')

clear images PixelFeatures PRNUFeatures FileName
%% IPT4
imageDir = 'S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\IPT4';

cd(imageDir)
images = dir('*.bmp');
cd('S:\WVU_Multimodal\CODES')
for i=1:length(images)
    i
    filename = images(i).name;
    
    img = imresize(double(imread(fullfile(imageDir,filename))),[96,96]);
    img = img(:,:,1);
    
    PixelFeatures(i,:) = img(:);
    Noisex_fft = PhaseNoiseExtractFromImage_SUD_Enhanced(img,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);
    PRNUFeatures(i,:) = Noiseresidual_testimage(:);
    FileName{i} = filename;
end
    
cd('S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\FeatsIPT4')
save('PixelFeatures_IPT4_Res.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT4_Res.mat','PRNUFeatures')

clear images PixelFeatures PRNUFeatures FileName
%% IPT5
 
imageDir = 'S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\IPT5';

cd(imageDir)
images = dir('*.bmp');
cd('S:\WVU_Multimodal\CODES')
for i=1:length(images)
    i
    filename = images(i).name;
    
    img = imresize(double(imread(fullfile(imageDir,filename))),[96,96]);
    img = img(:,:,1);
    
    PixelFeatures(i,:) = img(:);
    Noisex_fft = PhaseNoiseExtractFromImage_SUD_Enhanced(img,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);
    PRNUFeatures(i,:) = Noiseresidual_testimage(:);
    FileName{i} = filename;
end
    
cd('S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\FeatsIPT5')
save('PixelFeatures_IPT5_Res.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT5_Res.mat','PRNUFeatures')

clear images PixelFeatures PRNUFeatures FileName
%% IPT6
imageDir = 'S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\IPT6';

cd(imageDir)
images = dir('*.bmp');
cd('S:\WVU_Multimodal\CODES')
for i=1:length(images)
    i
    filename = images(i).name;
    
    img = imresize(double(imread(fullfile(imageDir,filename))),[96,96]);
    img = img(:,:,1);
    
    PixelFeatures(i,:) = img(:);
    Noisex_fft = PhaseNoiseExtractFromImage_SUD_Enhanced(img,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);
    PRNUFeatures(i,:) = Noiseresidual_testimage(:);
    FileName{i} = filename;
end
    
cd('S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\WVUMultimodal_GCN\TRAININGSET\FeatsIPT6')
save('PixelFeatures_IPT6_Res.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT6_Res.mat','PRNUFeatures')

clear images PixelFeatures PRNUFeatures FileName