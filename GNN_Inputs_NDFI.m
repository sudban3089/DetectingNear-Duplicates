clc
close all
clear

%% Process the pixel features and the sensor pattern noise feraures such that they can be fed to the GNN. 
%% Use this script to provide flattened pixel intensity values of near-duplicate sets, adjacency graph using PRNU, labels and partitions


%% PixelFeatures  (Use pixel intensity features from LFW dataset, we used a subset of features)
                                                                   
cd('Path to pixel intensity features of train IPT-1 set');
F1 = load('PixelFeatures_IPT1_Ori.mat');
train_1 = F1.PixelFeatures(3651:4900,:);
val_1 = F1.PixelFeatures(7451:7950,:);

cd('Path to pixel intensity features of train IPT-2 set');
F2 = load('PixelFeatures_IPT2.mat');
train_2 = F2.PixelFeatures(3651:4900,:);
val_2 = F2.PixelFeatures(7451:7950,:);

cd('Path to pixel intensity features of train IPT-3 set');
F3 = load('PixelFeatures_IPT3.mat');
train_3 = F3.PixelFeatures(4951:6200,:);
val_3 = F3.PixelFeatures(7981:8480,:);

cd('Path to pixel intensity features of train IPT-4 set');
F4 = load('PixelFeatures_IPT4.mat');
train_4 = F4.PixelFeatures(4951:6200,:);
val_4 = F4.PixelFeatures(7981:8480,:);

cd('Path to pixel intensity features of train IPT-5 set');
F5 = load('PixelFeatures_IPT5.mat');
train_5 = F5.PixelFeatures(6201:7450,:);
val_5 = F5.PixelFeatures(8486:8985,:);

cd('Path to pixel intensity features of train IPT-6 set');
F6 = load('PixelFeatures_IPT6.mat');
train_6 = F6.PixelFeatures(6201:7450,:);
val_6 = F6.PixelFeatures(8486:8985,:);


train=[train_1;train_2;train_3;train_4;train_5;train_6];
val=[val_1;val_2;val_3;val_4;val_5;val_6];

%% NDFI test images

cd('Path to pixel intensity features of NDFI test set')
F7 =  load('PixelFeatures.mat');
test = F7. PixelFeatures;

Features_PIXEL = [train; val; test];
Features_PIXEL= (double(Features_PIXEL));

%% Adjacency 

ADJ_1 =eye(5);


ADJ_1(1,2)=1;
ADJ_1(1,3)=1;
ADJ_1(2,4)=1;
ADJ_1(3,5)=1;

ADJ_1(1,4)=1;
ADJ_1(1,5)=1;

ADJ_2 =eye(5);



ADJ_2(1,2)=1;
ADJ_2(2,3)=1;
ADJ_2(3,4)=1;
ADJ_2(4,5)=1;

ADJ_2(1,3)=1;
ADJ_2(1,4)=1;
ADJ_2(1,5)=1;

ADJ_2(2,4)=1;
ADJ_2(2,5)=1;

ADJ_2(3,5)=1;


ADJ_3 =eye(5);



ADJ_3(1,2)=1;
ADJ_3(1,3)=1;
ADJ_3(2,4)=1;
ADJ_3(2,5)=1;


ADJ_3(1,4)=1;
ADJ_3(1,5)=1;

ADJ_4=eye(5);


ADJ_4(1,2)=1;
ADJ_4(1,3)=1;
ADJ_4(1,4)=1;
ADJ_4(1,5)=1;

ADJ_5 = eye(5);

ADJ_5(1,2)=1;
ADJ_5(1,3)=1;
ADJ_5(1,4)=1;
ADJ_5(2,5)=1;

ADJ_5(1,5)=1;

ADJ_6 = eye(5);

ADJ_6(1,2)=1;
ADJ_6(2,3)=1;
ADJ_6(3,4)=1;
ADJ_6(3,5)=1;


ADJ_6(1,3)=1;
ADJ_6(1,4)=1;
ADJ_6(1,5)=1;
ADJ_6(2,4)=1;
ADJ_6(2,5)=1;

%% TR
AR= eye(250);

ADJ_TR_1 = kron(AR,ADJ_1);
ADJ_TR_2 = kron(AR,ADJ_2);
ADJ_TR_3 = kron(AR,ADJ_3);
ADJ_TR_4 = kron(AR,ADJ_4);
ADJ_TR_5 = kron(AR,ADJ_5);
ADJ_TR_6 = kron(AR,ADJ_6);

ADJ_TR_ALL = blkdiag(ADJ_TR_1,ADJ_TR_2,ADJ_TR_3,ADJ_TR_4,ADJ_TR_5,ADJ_TR_6);

%% VAL

BR= eye(100);

ADJ_VA_1 = kron(BR,ADJ_1);
ADJ_VA_2 = kron(BR,ADJ_2);
ADJ_VA_3 = kron(BR,ADJ_3);
ADJ_VA_4 = kron(BR,ADJ_4);
ADJ_VA_5 = kron(BR,ADJ_5);
ADJ_VA_6 = kron(BR,ADJ_6);

ADJ_VAL_ALL = blkdiag(ADJ_VA_1,ADJ_VA_2,ADJ_VA_3,ADJ_VA_4,ADJ_VA_5,ADJ_VA_6);


%% TEST

cd('Path to NDFI PRNU features')
F8 = load('PRNUFeats.mat');
PRNUtest_2 = F8.PRNU_Features;

D_2=pdist(PRNUtest_2,'seuclidean');
Z_2=squareform(D_2);

n=10; j=size(PRNUtest_2,1)/n;

M_2=zeros(n,n,j);
for i=1:j
    M_2(:,:,i)= Z_2(10*(i-1)+1:10*i,10*(i-1)+1:10*i);
   
    mat_2 = M_2(:,:,i);

    mean_elem_2 = mean2(mat_2);

    for t=1:10
        meanind_2(t)=numel(find(mat_2(t,:)<mean_elem_2));
    end
    [sorted_elem_2, sorted_ind_2] = sort(meanind_2,'descend');
    UT_2(i,:)=sorted_ind_2;
end

ADJ_TE_2 = eye(n*j);
 for i=1:j
     ad_2 = eye(10);
     vect_2 = UT_2(i,:);
     for k=1:10
         if (k+11-(find(vect_2==k)))>=10
         
             ad_2(k,k:10)=ones(1,length(k:10));
         else
             
             ad_2(k,k:(11-find(vect_2==k)))=ones(1,length(11-(find(vect_2==k))));
         end
     end
     ADJ_TE_2(10*(i-1)+1:10*i,10*(i-1)+1:10*i)=ad_2;
 end
 
ADJ_TE_ALL = ADJ_TE_2;
%% Complete adjacency matrix
ADJ_ALL =blkdiag(ADJ_TR_ALL,ADJ_VAL_ALL,ADJ_TE_ALL);

ADJ_ALL_sp = sparse(ADJ_ALL);

 %% Create mask vectors for training, validation and test sets (30/70) 

numImages = (1500*5+600*5+size(ADJ_TE_2,1));

% Training

train_mask=int32(zeros(1, numImages));
train_mask(1:7500)=1;

% Validation 

val_mask=int32(zeros(1, numImages));
val_mask(7501:10500)=1;

% Testing

test_mask=int32(zeros(1, numImages));
test_mask(10501:end)=1;


%% Create label matrices for training, validation (30/70)  (depth as label)


IPT1_label = [1 0 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 1 0 0];
IPT2_label = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 0 1];
IPT3_label = [1 0 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 1 0 0];
IPT4_label = [1 0 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0];
IPT5_label = [1 0 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 0 1 0 0];
IPT6_label = [1 0 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 1 0];

% TRAIN
IPT1_ALL = repmat(IPT1_label,250,1);
IPT2_ALL = repmat(IPT2_label,250,1);
IPT3_ALL = repmat(IPT3_label,250,1);
IPT4_ALL = repmat(IPT4_label,250,1);
IPT5_ALL = repmat(IPT5_label,250,1);
IPT6_ALL = repmat(IPT6_label,250,1);

IPT_label_TR = [IPT1_ALL;IPT2_ALL;IPT3_ALL;IPT4_ALL;IPT5_ALL;IPT6_ALL];

clear IPT1_ALL IPT2_ALL IPT3_ALL IPT4_ALL IPT5_ALL IPT6_ALL


% VAL
IPT1_ALL = repmat(IPT1_label,100,1);
IPT2_ALL = repmat(IPT2_label,100,1);
IPT3_ALL = repmat(IPT3_label,100,1);
IPT4_ALL = repmat(IPT4_label,100,1);
IPT5_ALL = repmat(IPT5_label,100,1);
IPT6_ALL = repmat(IPT6_label,100,1);

IPT_label_VAL = [IPT1_ALL;IPT2_ALL;IPT3_ALL;IPT4_ALL;IPT5_ALL;IPT6_ALL];

clear IPT1_ALL IPT2_ALL IPT3_ALL IPT4_ALL IPT5_ALL IPT6_ALL


% TEST
IPT_TE2_label = [1 0 0 0 0; 0 1 0 0 0; 0 1 0 0 0; 0 0 1 0 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 1 0; 0 0 1 0 0; 0 0 0 1 0; 0 0 0 1 0];
IPT_label_TE = repmat(IPT_TE2_label,1229,1);

% Training

y_train=zeros(numImages,5);

y_train(1:7500,:) = IPT_label_TR;

% Validation 

y_val=zeros(numImages,5);

y_val(7501:10500,:) = IPT_label_VAL;

% Testing

y_test=zeros(numImages,5);

y_test(10501:end,:) = IPT_label_TE;

%% SAVE (save the inputs to the GNN in matfiles, Create a directory 'INPUTMATFILES_XXX' where XXX is the name of the test set)

cd('INPUTMATFILES_NDFI\')
save('adj.mat','ADJ_ALL_sp')
save('PixelFeatures.mat','Features_PIXEL')
save('y_train.mat','y_train')
save('y_val.mat','y_val')
save('y_test.mat','y_test')
save('train_mask.mat','train_mask')
save('val_mask.mat','val_mask')
save('test_mask.mat','test_mask')

