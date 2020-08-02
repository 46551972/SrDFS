function [feature_idx, A, obj]=ldaL21FS(xTr,yTr, r, W, gamma_W)
%% 21-norm loss with 21-norm regularization
% xTr is a Dim * N data matrix, each column is a data vector
% yTr is a numClass * N label matrix, each column is a label vector
% gnd is a label vector, each entry indicates the class of a data point
%% Problem
%
%  min_A  || XA - Y||_F^2 + r * ||A||_12 + gamma_W Tr( AX L (XA)^T)

% Ref: Feiping Nie, Heng Huang, Xiao Cai, Chris Ding. 
% Efficient and Robust Feature Selection via Joint L21-Norms Minimization.  
% Advances in Neural Information Processing Systems 23 (NIPS), 2010.

if ~exist('r','var'),  r = 0.1;   end

if exist('W','var'),   
    L = diag(sum(W)) - W; 
else
    L = 0;  gamma_W = 0; 
end

labelSet = unique(yTr);  
numClass = numel(labelSet);   
[Dim, N] = size(xTr);

NIter = 15;
dA = ones(Dim,1);

% Yregress = generateY(yTr);
Yregress = full(sparse(yTr,1:numel(yTr),1));

for iter = 1:NIter
    Da = diag(dA);
    A = (Yregress*xTr'*Da)/(xTr*xTr'*Da+r*eye(Dim) + gamma_W*xTr*L*xTr'*Da);
    
    Ai = sqrt(sum(A.*A,1));
    dA = Ai;
    
    Xi1 = A*xTr - Yregress;
    obj(iter) = norm(Xi1,'fro')^2 + r*sum(Ai);
end;



[~, idx] = sort(sum(A.*A,1),'descend');
feature_idx = idx(1:Dim);
scoresFea = sum(A.*A,2).^(0.5); 
% figure(8); cla;
% title(['AforldaL21FS method, r = ' num2str(r)]);
% hold on; plot(sum(A.^2,1).^(0.5),'r-');  drawnow; 

end

% 
% function Y = generateY(yTr)
% Labels = unique(yTr);
% nClass = length(Labels);
% nSmp = numel(yTr);
% % rand('state',0);
% Y = rand(nClass,nClass);
% Z = zeros(nSmp,nClass);
% for i=1:nClass
%     idx = find(yTr==Labels(i));
%     Z(idx,:) = repmat(Y(i,:),length(idx),1);
% end
% Z(:,1) = ones(nSmp,1);
% [Y,R] = qr(Z,0);
% Y(:,1) = [];
% Y = Y'; 
% end