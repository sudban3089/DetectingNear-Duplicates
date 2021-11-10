clc
close all
clear

%% This script extracts the pixel intensity features and sensor pattern noise (PRNU) features from the images of the NDFI test set

% Make sure the folders Functions/ and Filter/ are loctaed in the current working diretcory for PRNU extraction
addpath('Functions/')
addpath('Filter/')
qmf = MakeONFilter('Daubechies',8);
L = 4;


imageDir = 'Path to NDFI SET I imagesa';
cd(imageDir)
images = dir('*.bmp');
cd('Path to where /Filter and /Functions folders are located')
for i=1:length(images)
    i
    filename = images(i).name;
    image = double(imread(fullfile(imageDir,filename)));
        
    PixelFeatures(i,:) = image(:);
    
    Noisex_fft = PhaseNoiseExtractFromImage_Enhanced(image,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN 
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);    
    PRNU_Features(i,:) = Noiseresidual_testimage(:);
    
    FileName{i} = filename;
end
    
cd('Results\FaceFeats_NDFI')
save('PixelFeatures.mat','FileName','PixelFeatures')
save('PRNUFeats.mat','PRNU_Features')
clear images PixelFeatures FileName PRNU_Features

