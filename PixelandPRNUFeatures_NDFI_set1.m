clc
close all
clear

addpath('Functions/')
addpath('Filter/')

qmf = MakeONFilter('Daubechies',8);
L = 4;


imageDir = 'S:\DependenceGraph_FaceImages\Dataset\SET I';
cd(imageDir)
images = dir('*.bmp');
cd('S:\WVU_Multimodal\CODES')
for i=1:length(images)
    i
    filename = images(i).name;
    image = double(imread(fullfile(imageDir,filename)));
        
    PixelFeatures(i,:) = image(:);
    
    Noisex_fft = PhaseNoiseExtractFromImage_SUD_Enhanced(image,qmf,2,L);
    %% use this for Basic SPN, enhanced Basic SPN 
    Noiseresidual_spatial = single(Noisex_fft);
    Noiseresidual_testimage = double(Noiseresidual_spatial);    
    PRNU_Features(i,:) = Noiseresidual_testimage(:);
    
    FileName{i} = filename;
end
    
cd('S:\PGM Image Phylogeny\Codes\HGNN-master\datasets\FaceFeats_NDFI\SetI')
save('PixelFeatures_Resized.mat','FileName','PixelFeatures')
save('PRNUFeats_Resized.mat','PRNU_Features')
clear images PixelFeatures FileName PRNU_Features

