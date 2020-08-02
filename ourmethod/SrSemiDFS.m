% StrSSC.m
% Description: This code is for the Structured Sparse Subspace Clustering.
%
% Ref:
%   Chun-Guang Li and Ren¨¦ Vidal, "Structured Sparse Subspace Clustering: A Unified Optimization Framework",
%   In CVPR, pp.277-286, June 7-12, 2015.
%
%   [acc_i, Theta, C, eval_iter] = StrSSC(D, idx, opt, DEBUG)
%
%    Inputs:  D - data matrix, each column as a data point
%                 idx - data points groundtruth label;
%                 opt - the parameters setting:
%                     opt.affine: 0 or 1
%                     opt.outliers: 1 or 0
%                     opt.gamma0 - e.g., 0.1 ( 0.1 to 0.25), the parameter to re-weight with Theta
%                     opt.nu  - set as 1, to make the first run of StrSSC the same as SSC
%                     opt.lambda - it is the lambda for other algorithm and the alpha for SSC's code
%                     opt.r - the dimension of the target space when applying PCA or random projection
%                     opt.iter_max - eg. 5, 10, to set the maximum iterations of the StrSSC's outer loop
%                     opt.maxIter - the maximum iteration number of ADMM
%                     opt.SSCrho - the thresholding paramter sometimes is used in SSC (e.g. rho = 0.7 for MotionSegmentation). By
%                     default, it is set as 1
%
%    Outputs:
%                acc_i  - accuracy
%                Theta -  subspace membership matrix, i.e. the structure indicator matrix
%                C  -  sparse representation
%                eval_iter - record the acc in each iteration for StrSSC
%
%    How to Use:
%
%             %% paramters for StrSSC
%             opt.iter_max =10;
%             opt.gamma0 =0.1; % This is for reweighting the off-diagonal entries in Z
%             opt.nu =1;
%
%             %% paramters for ssc
%             opt.affine = 0;
%             opt.outliers =1;
%             opt.lambda =10;
%             opt.r =0;
%             opt.SSCrho =1;
%
%             %% paramters for ADMM
%             opt.tol =1e-5;
%             opt.rho=1.1;
%             opt.maxIter =150;
%             opt.mu_max = 1e8;
%             opt.epsilon =1e-3;
%             %opt.tol =1e-3;
%             %opt.rho =1.1;
%
%
%             [acc_i, Theta_i, C, eval_iter] = StrSSC(D, idx, opt);
%
% Copyright by Chun-Guang Li
% Date: Jan. 7th, 2014
% Modified: Oct.31. 2014
% Revised by July 28, 2015
% Associating with ldaL21FS.m
function [feature_idx, A, eval_iter, grps] = SrSemiDFS(xTr,yTr,xTe,opt,yTe)

% yTr and yTe are label vectors, each data point associate with an label
% entry.
dataX = [xTr, xTe];
[~, N] = size(dataX);
%% parameter settings:
% StrSSC specific parameters

if exist('opt','var')
    opt = optSetforS3C(opt);
else
    opt = optSetforS3C();
end

% parameters used in SSC
Xp = DataProjection(dataX,opt.dimR);  % opt.dimR: the dimension of the target space when applying PCA or random projection

%% Initialization
C =zeros(size(dataX,2));    % C is initialized as a zero matrix
Theta_old =ones(size(dataX, 2));
%Theta_old =zeros(size(D, 2));
grps =0;
eval_iter =zeros(1, opt.iter_max);     % iter_max =10
iter =0;  
while (iter < opt.iter_max)
    iter = iter +1;
    
    if (iter <= 1)
        %% This is the standard SSC when iter <=1
        nu =1;
    else
        %% This is for re-weighted SSC
        nu = nu * 1.2;%1.1, 1.2, 1.5,
    end
    
    %% run ADMM to solve the re-weighted SSC problem
    C = ADMM_S3R(Xp, opt.outliers, opt.affine, opt.lambda, Theta_old, opt.gamma0, nu,  opt.maxIter, C);
    %% Initialize Z with the previous optimal solution
    %   CKSym = BuildAdjacency(thrC(C,rho));
    %   W = (abs(CKSym) + abs(CKSym)')/2;
    W = (abs(C) + abs(C)')/2;
    %% --------Semi-Supervised and Unsupervised label prediction------
    [~, grpsU, Acc] = my_grf(yTr, W, yTe);
    grps = [yTr(:); grpsU(:)];
    disp(['iter = ',num2str(iter),', Semi-Supervised Acc = ', num2str(Acc)]);
    eval_iter(iter) = Acc;
    
    % vecY = [full(sparse(yTr,1:numel(yTr),1)), preYu];
    vecY = full(sparse(grps,1:N,1));
    % Theta1 = L2_distance(vecY(:,1:N),vecY(:,1:N));  Theta = (Theta.^2)/2;
    Theta1 = 1 - form_structure_matrix(grps);
    %% ---------------- Feature Selection-----------------------------
    gamma_W = opt.alphaTheta*opt.gamma0/2;  % According to our plan on the parameter setting, maybe suboptimal
    [feature_idx, A] = ldaL21FS(dataX, grps, opt.gamma_A, W, gamma_W);
    
    Yw = A*dataX; Theta2 = L2_distance(Yw(:,1:N),Yw(:,1:N));  Theta2 = Theta2.^2;
    Theta =((1-opt.alphaTheta) * Theta1 + opt.alphaTheta*Theta2)/2;
%     %% Checking stop criterion
%     tmp =Theta - Theta_old;
%     if (max(abs(tmp(:))) < 10^(-9))
%         disp('max error lower than threshold and code break!')
%         break; % if Theta didn't change, stop the iterations.
%     end
    Theta_old =Theta;
end
end

function M = form_structure_matrix(idx,n)
if nargin<2
    n =size(idx,2);
end
M =zeros(n);
id =unique(idx);
for i =1:length(id)
    idx_i =find(idx == id(i));
    M(idx_i,idx_i)=ones(size(idx_i,2));
end
end


function [preYu,labelU, Acc] = my_grf(yTr, W, yTe)

l = numel(yTr);
n = size(W, 1);
% total number of points
YTr = full(sparse(yTr,1:l,1));

% the graph Laplacian L=D-W
L = diag(sum(W)) - W;
epsilon1 = 10^(-9); 
% the harmonic function.
preYu = - YTr * L( 1:l,l+1:n)/(L(l+1:n, l+1:n)+epsilon1*eye(n-l)); %%%Matrix is close to singular or badly scaled.

% compute the CMN solution
q = sum(YTr,2)+1; % the unnormalized class proportion estimate from labeled data, with Laplace smoothing
preYu = preYu .* repmat(q./sum(preYu,2), 1, n-l);

[~,labelU] = max(preYu,[],1);
if exist('yTe','var')
    Acc = 100*sum(labelU(:) == yTe(:))/numel(yTe);
end
end