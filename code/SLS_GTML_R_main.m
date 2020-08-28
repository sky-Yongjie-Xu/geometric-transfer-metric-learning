clc
clear

%% set hyper-parameter
n_neighbor = 1;
c = 3;

num_source_data = 600;
select_source_data = 300;
num_target_data = 50;
select_train_data = 3;

Label_pair_target_train_val(1:select_train_data,1:select_train_data) = 1;
Label_pair_target_train_val(1:select_train_data,select_train_data+1:3*select_train_data) = -1;
Label_pair_target_train_val(select_train_data+1:2*select_train_data,1:select_train_data) = -1;
Label_pair_target_train_val(select_train_data+1:2*select_train_data,select_train_data+1:2*select_train_data) = 1;
Label_pair_target_train_val(select_train_data+1:2*select_train_data,2*select_train_data+1:3*select_train_data) = -1;
Label_pair_target_train_val(2*select_train_data+1:3*select_train_data,1:2*select_train_data) = -1;
Label_pair_target_train_val(2*select_train_data+1:3*select_train_data,2*select_train_data+1:3*select_train_data) = 1;

%% load data
load HR_SAR_DATA
load AIS_DATA
load result_sls_hr_sar_gtml_r

%% random select data index of source domain
rand('seed',1);
for t1 = 1:10
    l_source_train_data(t1,:) = randperm(num_source_data,select_source_data);
end

%% random select data index of target domain
rand('seed',1);
for t2 = 1:10
    l_target_data(t2,:) = randperm(num_target_data);
    l_target_train_data(t2,:) = l_target_data(t2,1:select_train_data);
    l_target_test_data(t2,:) = l_target_data(t2,select_train_data+1:end);
end

for num = 1:10
    
    etas = result_sls_hr_sar_gtml_r(num,1);
    etad = result_sls_hr_sar_gtml_r(num,2);
    mium = result_sls_hr_sar_gtml_r(num,3);
    miuc = result_sls_hr_sar_gtml_r(num,4);
    gamma = result_sls_hr_sar_gtml_r(num,5);
    
    n_source_train_data = [l_source_train_data(num,:) l_source_train_data(num,:)+num_source_data l_source_train_data(num,:)+2*num_source_data];
    n_target_train_data = [l_target_train_data(num,:) l_target_train_data(num,:)+num_target_data l_target_train_data(num,:)+2*num_target_data];
    n_target_test_data = [l_target_test_data(num,:) l_target_test_data(num,:)+num_target_data l_target_test_data(num,:)+2*num_target_data];
    
    %% select data
    xSr = AIS_FEATURE(n_source_train_data,:);
    ySr = AIS_LABEL(n_source_train_data,:);
    xTr = HR_SAR_FEATURE(n_target_train_data,:);
    yTr = HR_SAR_LABEL(n_target_train_data,:);
    xTe = HR_SAR_FEATURE(n_target_test_data,:);
    yTe = HR_SAR_LABEL(n_target_test_data,:);
    
    X = [xSr;xTr];
    Y = [ySr;yTr];
    [ms,ns] = size(xSr);
    [mt,nt] = size(xTr);
    [m,n] = size(X);
    
    %% similar constraints
    Ws = max(Label_pair_target_train_val,0);
    Ls = diag(sum(Ws))-Ws;
    
    %% dissimilar constraints
    Wd = max(-Label_pair_target_train_val,0);
    Ld = diag(sum(Wd))-Wd;
    S = xTr'*Ls*xTr;
    
    %% marginal distribution
    eb = [1/ms*ones(ms,1);-1/mt*ones(mt,1)];
    Lb = eb*eb'*length(unique(Y(1:n)));
    D = xTr'*Ld*xTr;
    
    %% conditional distribution
    Lc = 0;
    for C = 1:c
        et = zeros(m,1);
        et(Y(1:ms)==C) = 1/(length(find(Y(1:ms)==C)))^2;
        et(ms+find(yTr==C)) = -1/(length(find(yTr==C)))^2;
        et(isinf(et)) = 0;
        Lc = Lc + et*et';
    end
    
    %% manifold regularization
    W1 = KNNGraph_constant(xTr', n_neighbor, flag);
    Lm = diag(sum(W1))-W1;
    
    %% optimization
    M = eye(n);
    cvx_begin sdp quiet
    cvx_solver SDPT3
    variable M(11,11) semidefinite
    minimize(mium*trace(X'*Lb*X*M)+miuc*trace(X'*Lc*X*M)+etas*trace(S*M)+gamma*trace(xTr'*Lm*xTr*M))
    subject to
    etad*trace(D*M)==1
    cvx_end
    
    preds = KNN_M(yTr,xTr,M,n_neighbor,xTe);
    
    %% test
    acc_test_gtml_r(num,1) = sum(preds==yTe)/size(yTe,1);
    fprintf('%4f\n',acc_test_gtml_r(num,1));

end

mean_acc_gtml_r = mean(acc_test_gtml_r);
std_acc_gtml_r = std(acc_test_gtml_r);
fprintf('%.4f %.4f\n',mean_acc_gtml_r,std_acc_gtml_r);
