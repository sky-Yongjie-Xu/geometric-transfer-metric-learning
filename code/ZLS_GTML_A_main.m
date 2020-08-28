clc
clear
rand('seed',1);

%% set hyper-parameter
options.n_neighbor = 1;
options.c = 3;

n_neighbor = options.n_neighbor;
c = options.c;

num_target_data = 50;
num_source_data = 600;
select_source_data = 300;

Label_pair(1:select_source_data,1:select_source_data) = 1;
Label_pair(1:select_source_data,select_source_data+1:3*select_source_data) = -1;
Label_pair(select_source_data+1:2*select_source_data,1:select_source_data) = -1;
Label_pair(select_source_data+1:2*select_source_data,select_source_data+1:2*select_source_data) = 1;
Label_pair(select_source_data+1:2*select_source_data,2*select_source_data+1:3*select_source_data) = -1;
Label_pair(2*select_source_data+1:3*select_source_data,1:2*select_source_data) = -1;
Label_pair(2*select_source_data+1:3*select_source_data,2*select_source_data+1:3*select_source_data) = 1;
    
%% load data
load HR_SAR_DATA
load AIS_DATA
load result_zls_hr_sar_gtml_a

%% random select source index
for t1 = 1:10
    l_source_data(t1,:) = randperm(num_source_data,select_source_data);
end

for num = 1:10
    
    etas = result_zls_hr_sar_gtml_a(num,1);
    etad = result_zls_hr_sar_gtml_a(num,2);
    mium = result_zls_hr_sar_gtml_a(num,3);
    miuc = result_zls_hr_sar_gtml_a(num,4);
    gamma = result_zls_hr_sar_gtml_a(num,5);
    
    n_source_data = [l_source_data(num,:) l_source_data(num,:)+num_source_data l_source_data(num,:)+2*num_source_data];
    
    %% select data
    xSr = AIS_FEATURE(n_source_data,:);
    ySr = AIS_LABEL(n_source_data,:);
    xTr = HR_SAR_FEATURE;
    yTr = HR_SAR_LABEL;
    xTe = HR_SAR_FEATURE;
    yTe = HR_SAR_LABEL;
    
    %% normalization
    xSr = xSr*diag(sparse(1./sqrt(sum(xSr.^2))));
    xTr = xTr*diag(sparse(1./sqrt(sum(xTr.^2))));
    xTe = xTe*diag(sparse(1./sqrt(sum(xTe.^2))));
    
    X = [xSr;xTr];
    Y = [ySr;yTr];
    
    [ms,ns] = size(xSr);
    [mt,nt] = size(xTr);
    [m,n] = size(X);
    
    %% similar constraints
    Ws = max(Label_pair,0);
    Ls = diag(sum(Ws))-Ws;
    S = xSr'*Ls*xSr;
    
    %% dissimilar constraints
    Wd = max(-Label_pair,0);
    Ld = diag(sum(Wd))-Wd;
    D = xSr'*Ld*xSr;
    
    %% marginal distribution
    eb = [1/ms*ones(ms,1);-1/mt*ones(mt,1)];
    Lmd = eb*eb'*length(unique(Y(1:n)));
    
    %% conditional distribution
    % Generate pseudo labels for the target domain
    if ~isfield(options,'Yt0')
        % 1NN
        knn_model = fitcknn(sparse(xSr),ySr,'NumNeighbors',1);
        Y_tar_pseudo = knn_model.predict(sparse(xTr));
    else
        Y_tar_pseudo = options.Yt0;
    end
    
    Lcd = 0;
    for class = 1:c
        et = zeros(m,1);
        et(Y(1:ms)==class) = 1/(length(find(Y(1:ms)==class)))^2;
        et(ms+find(Y_tar_pseudo==class)) = -1/(length(find(Y_tar_pseudo==class)))^2;
        et(isinf(et)) = 0;
        Lcd = Lcd + et*et';
    end
    
    %% manifold regularization
    W1 = KNNGraph_constant(X', n_neighbor, flag);
    Lm = diag(sum(W1))-W1;
    
    %% optimization
    M = eye(n);
    cvx_begin sdp quiet
    cvx_solver SDPT3
    variable M(11,11) semidefinite
    minimize(mium*trace(X'*Lmd*X*M)+miuc*trace(X'*Lcd*X*M)+etas*trace(S*M)-etad*trace(D*M)+gamma*trace(X'*Lm*X*M))
    cvx_end
    
    preds = KNN_M(ySr,xSr,M,n_neighbor,xTe);
    
    %% test
    acc_test_gtml_a(num,1) = sum(preds==yTe)/size(yTe,1);
    fprintf('%4f\n',acc_test_gtml_a(num,1));
    
end

acc_gtml_a_mean = mean(acc_test_gtml_a);
acc_gtml_a_std = std(acc_test_gtml_a);
fprintf('%0.4f %0.4f\n',acc_gtml_a_mean,acc_gtml_a_std);
