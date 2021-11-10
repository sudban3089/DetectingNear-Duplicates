clc
close all
clear

%% This script extracts the pixel intesnity fetaures from resized images of 96x96 and the sensor pattern nosie (PRNU) features

% Make sure the Functions and Filter folder is added to the working directory
addpath('Functions/')
addpath('Filter/')

qmf = MakeONFilter('Daubechies',8);
L = 4;

%% Process LFW images to craete training set of 6 IPT configurations.

 % IPT1
 imageDir = 'TRAININGSET\IPT1';

cd(imageDir)
images = dir('*.bmp');
cd('Directory where Filter\ and Functions\ are located')
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
    
cd('Results\TRAININGSET\FeatsIPT1')
save('PixelFeatures_IPT1.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT1.mat','PRNUFeatures')
clear images PixelFeatures PRNUFeatures FileName

% IPT2
imageDir = 'TRAININGSET\IPT2';

cd(imageDir)
images = dir('*.bmp');
cd('Directory where Filter\ and Functions\ are located')
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
    
cd('Results\TRAININGSET\FeatsIPT2')
save('PixelFeatures_IPT2.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT2.mat','PRNUFeatures')
clear images PixelFeatures PRNUFeatures FileName

% IPT3
 
imageDir = 'TRAININGSET\IPT3';

cd(imageDir)
images = dir('*.bmp');
cd('Directory where Filter\ and Functions\ are located')
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
    
cd('Results\TRAININGSET\FeatsIPT3')
save('PixelFeatures_IPT3_Res.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT3_Res.mat','PRNUFeatures')
clear images PixelFeatures PRNUFeatures FileName

% IPT4
imageDir = 'TRAININGSET\IPT4';

cd(imageDir)
images = dir('*.bmp');
cd('Directory where Filter\ and Functions\ are located')
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
    
cd('Results\TRAININGSET\FeatsIPT4')
save('PixelFeatures_IPT4.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT4.mat','PRNUFeatures')
clear images PixelFeatures PRNUFeatures FileName

% IPT5
 
imageDir = 'TRAININGSET\IPT5';

cd(imageDir)
images = dir('*.bmp');
cd('Directory where Filter\ and Functions\ are located')
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
    
cd('Results\TRAININGSET\FeatsIPT5')
save('PixelFeatures_IPT5.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT5.mat','PRNUFeatures')
clear images PixelFeatures PRNUFeatures FileName

% IPT6
imageDir = 'TRAININGSET\IPT6';

cd(imageDir)
images = dir('*.bmp');
cd('Directory where Filter\ and Functions\ are located')
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
    
cd('Results\FeatsIPT6')
save('PixelFeatures_IPT6.mat','FileName','PixelFeatures')
save('PRNUFeatures_IPT6.mat','PRNUFeatures')

clear images PixelFeatures PRNUFeatures FileName
