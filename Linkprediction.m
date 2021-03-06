clc
close all
clear

%% This script perform link prediction by using pairwise analysis. Uee the depth labels produced by GNN and the PRNU features of the test set to construct the IPT.
%% This script also evaluates the IPT reconstruction in terms of root node identification and IPT reconstruction accuracy

cd('GNNPredictions');
load('Depthlabels.mat');
PRED_TEST = depth_labels(10501:end); % test set satrts from 10501 onwards
PRED_TEST_NDFI=PRED_TEST+1; % the depth labels start from '0' so we added a '1'
save('PRED_test_NDFI.mat','PRED_TEST_NDFI');

cd('INPUTMATFILES_NDFI')
load('y_test.mat');
tmp = y_test(10501:end,:);
for i=1:length(tmp)
    GT_TEST(i)=find(tmp(i,:)==1);
end
cd('GNNPredictions');
save('GT_test.mat','GT_TEST');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
cd('GNNPredictions');
load('GT_test.mat')
load('PRED_test_NDFI.mat')

% Ground truth IPT configuration for NDFI
ORIGINAL_EDGES_NDFI = [[[2:10]', ones(9,1),]; 4,2;5,2;6,2;7,2; 8,3;9,3;10,3; 6,4; 7,5; 9,8;10,8];

% PRNU Features
cd('Results\FaceFeats_NDFI')
load('PRNUFeatures.mat')
testPRNU = PRNU_Features;
rowmin = min(testPRNU,[],2);
rowmax = max(testPRNU,[],2);
testPRNU_norm = rescale(testPRNU,'InputMin',rowmin,'InputMax',rowmax);
dist = pdist(testPRNU_norm,'seuclidean');

correlate = squareform(dist);
numNodes=10;

for i=1:size(testPRNU_norm,1)/numNodes
    i
    tic
    row = numNodes*(i-1)+1;
    col = numNodes*i;
    cc = correlate(row:col,row:col);
      
    vect_old = PRED_TEST_NDFI(1,row:col);
   
            multroots = find(vect_old==1);
            vect = vect_old;
            if length(multroots)>1
                for rt=1:length(multroots)
                    remind = setdiff([1:length(multroots)],rt);%setdiff([1:numNodes],rt);
                    for rind =1:length(remind)
                        multroots_temp(rind) = cc(multroots(rt),multroots(remind(rind)));%cc(multroots(rt),remind(rind));
                    end
                    multroots_cc(rt) = mean(multroots_temp);
                    clear remind multroots_temp
                end
                [~, sortrootsi] = sort(multroots_cc,'descend');
                if length(sortrootsi)>1
                    vect(multroots(sortrootsi(1)))=1;
                    remroots = setdiff(multroots, multroots(sortrootsi(1)));
                    
                    for y=1:length(remroots)
                        vect(remroots(y)) = 2;
                    end
                    clear remroots sortrootsi multroots_cc multroots
                end
            end
            
            node = 1:numNodes;
           depth = vect;
    
    parent =[];
    child = [];
    t=1;
    for j=1:length(node)
        n= node(j);
        
        cc_temp_node = [];
        pot_links = [];
        
        pot_links = find(depth>depth(n));
            
            for k=1:length(pot_links)
                cc_temp_node(k)=cc(n,pot_links(k));
            end
            [sortval,sortind] = sort(cc_temp_node);%,'descend'
            
            if ~isempty(sortval)
                for m=1:length(sortval)
                    parent(t) = n;
                    child(t) = find(cc(n,:)==sortval(m),1);
                    t=t+1;
                end
                clear sortval sortind
            end
           
           % clear parent child
        
    end
     
            tree = [child' parent'];
            tree  = unique(tree,'rows');
            
            unipar = unique(tree(:,2));
            for par=1:length(unipar)
                NumPar(par) = numel(find(tree(:,2)==unipar(par)));
            end
           [~,sortrootind] = sort(NumPar,'descend');
            for si=1:length(sortrootind)
            ROOTS_TREES(si) = (sortrootind(si));
            end
    TREE_10nodes(i).name = tree;
    ROOTFROMIPT(i,:) = ROOTS_TREES(1:3);
    toc
    if size(tree,1)>size(ORIGINAL_EDGES_NDFI,1)
        Accuracy_NDFI(i)=sum(ismember(ORIGINAL_EDGES_NDFI,tree,'rows'))/length(ORIGINAL_EDGES_NDFI);
    else
        Accuracy_NDFI(i)=sum(ismember(tree,ORIGINAL_EDGES_NDFI,'rows'))/length(ORIGINAL_EDGES_1NDFI);
    end
  
  clear  sortrootind unipar treeind_ACTUAL imgind rootmax parent child tree vect cc_temp_node sortval sortind pot_links REMnodes depth
      clear ROOTS_TREES NumPar cc vect vect_old vect_uni sortrootsv sortrootsi sortnodesi multroots multnodes multroots_cc mean_cc remroots correct err
                   
end

disp('IPT Recon 10 nodes:')
mean(Accuracy_NDFI)
 
numel(find(ROOTFROMIPT(:,1)==1))/length(ROOTS) % Rank 1
((numel(find(ROOTFROMIPT(:,1)==1)))+(numel(find(ROOTFROMIPT(:,2)==1))))/length(ROOTS) % Rank 2
((numel(find(ROOTFROMIPT(:,1)==1)))+(numel(find(ROOTFROMIPT(:,2)==1)))+(numel(find(ROOTFROMIPT(:,3)==1))))/length(ROOTS) % Rank 3

cd('IPTReconstructions\NDFI');
save('TREES.mat','TREE_10nodes')
save('ROOTSFROMIPT.mat','ROOTFROMIPT')


