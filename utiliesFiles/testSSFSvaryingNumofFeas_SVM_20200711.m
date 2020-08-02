function testSSFSvaryingNumofFeas_SVM_20200711(gX,gY,filepath,filename,dataName)

[~,N] = size(gX);    labelSet = unique(gY);    numClass = numel(labelSet);
%%%%%%%%%%%--------------------Parameters Setting---------------%%%%%%%%
Nround = 5;
if strcmp(dataName,'ORL_32x32')
    ldcRange = 1:5;   feaRange = [10:20:80 100:25:250];   
elseif strcmp(dataName,'MSRA25_uni')
    ldcRange = 1:5;   feaRange = [5:10:50 50:20:150];   
end

cmdSVMtext = [' -c ',num2str(2^9),' -g ',num2str( 2^2 )];

for T = 1:Nround
    %% Random permutation of the data points
    rnperm = randperm(N);    
    dataX = gX(:,rnperm);  labelY = gY(rnperm);
    [Dim,N] = size(dataX);
    %% index of each class
    Dind=cell(numClass,1);
    for iterC=1:numClass,      
        Dind{iterC}=find(labelY==labelSet(iterC));  
    end
    for iter1 = 1:length(ldcRange)
        ldc = ldcRange(iter1);
        ind1 =[];  ind2=[];
        for c=1:numClass
            ind1 = [ind1; Dind{c}(1:ldc)];                              %%%  labeled index
            ind2 = [ind2; Dind{c}((1+ldc):end)];                        %%%  unlabeled index
        end
        xTr = dataX(:,ind1);    yTr = labelY(ind1);     nTr = numel(yTr);    %%%  labeled data
        xTe = dataX(:,ind2);    yTe = labelY(ind2);     nTe = numel(yTe);    %%%  unlabeled data
        vYSSL = [full(sparse(yTr,1:nTr,1)), zeros(numClass,nTe)];            %%%  vYSSL = [YL, YU]; 
        vYr = full(sparse(yTr,1:numel(yTr),1));
        
        %% all feature methods
        %% ------------lib SVM classifier
        modelsvm = svmtrain1(yTr,xTr',cmdSVMtext);
        [~, accuracy, ~] = svmpredict1(yTe, xTe', modelsvm);
        AccSVM(T,iter1)= accuracy(1)
        %% -Structure regularized Discriminant Feature Selection----
        opt.nbcluster = numClass;  opt.gamma0 = 0.05; 
        [ind_SrSemiDFS] = SrSemiDFS(xTr,yTr,xTe,opt,yTe);
      
        for iter2 = 1:length(feaRange)
            numFeat = feaRange(iter2);
            %% ----------- SrSemiDFS FS --------------
            modelsvm = svmtrain1(yTr,xTr(ind_SrSemiDFS(1:numFeat),:)',cmdSVMtext);
            [~, accuracy,~] = svmpredict1(yTe, xTe(ind_SrSemiDFS(1:numFeat),:)', modelsvm);
            SrSemiDFSAccSVM(T,iter1,iter2)= accuracy(1)
        end
    end
end
a

%% --------------------------SVM classification-------------------------------
[SrSemiDFSAcc1,SrSemiDFSAcc2] = processResults(SrSemiDFSAccSVM) ;

iind =ldcRange;
figure; hold on;  title('SVM classification');
errorbar(iind,mean(AccSVM,1), 0.001*std(AccSVM), ...
    'k-','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','k','MarkerSize',8);  %%%  SVM
errorbar(iind,mean(SrSemiDFSAcc2,1), 0.1*std(SrSemiDFSAcc2),...
    'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','y','MarkerSize',8); %%% Sr-SemiDFS
grid on;
legend('SVM','Sr-SemiDFS');
hold off

%% ----------------Varying Num of Features
iind =feaRange;
figure; hold on;title('SVM classification'); 
errorbar(iind,ones(size(iind))*mean(AccSVM(:),1), 0.001*ones(size(iind))*std(AccSVM(:)), ...
    'k-','LineWidth',2,'MarkerEdgeColor','b','MarkerFaceColor','k','MarkerSize',8);  %%%  1NN
errorbar(iind,mean(SrSemiDFSAcc1,1), 0.1*std(SrSemiDFSAcc1),...
    'ro-','LineWidth',2,'MarkerEdgeColor','r','MarkerFaceColor','y','MarkerSize',8); %%% Sr-SemiDFS

grid on;
legend('SVM','Sr-SemiDFS');
hold off
filename1 = [filepath, filename];
save(filename1)