function [W0, pos] = KNNGraph_constant(TrainData, K, flag)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% KNNGraph
% Written by Wei Liu (wliu@ee.columbia.edu)
% TrainData(dXn): input data matrix, d: dimension, n: # samples
% K: KNN search, usually fixed to 5-20;
% flag: 0 gives a KNN graph and 1 gives a KNN SGraph
% W0: the output graph adjacency matrix which is sparse
% pos(nxK): the indexes of KNN for every data point 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[dim, n] = size(TrainData); 
Dis = sqdist(TrainData,TrainData)+1e20*eye(n);
clear TrainData;

val = zeros(n,K);
pos = val;
for i = 1:K
    [val(:,i),pos(:,i)] = min(Dis,[],2);
    tep = (pos(:,i)-1)*n+[1:n]';
    Dis(tep) = 1e20; 
end
clear Dis;
clear tep;

val = 1;  %% to be tuned
tep = (pos-1)*n+repmat([1:n]',1,K);

W0 = zeros(n,n);
W0([tep]) = [val];
clear tep;
clear val;
W0 = sparse(W0);

if flag == 0
   W0 = max(W0,W0');
else
   W0 = (W0+W0')/2; 
end

