%%%%%%%-------------数据读入---- dataX 的每一列表示一个样本----------
clear; clc;

  load('MSRA25_uni');   dataName = 'MSRA25_uni';  
  gY = Y(:);   gX = NormalizeFea(X',0);

fileSaveName = [dataName,'_SVM_Varying_ldc']; %% filename to save result
fileSavePath = 'C:\SSFSexp\';  %% path to save result
if ~exist(fileSavePath,'dir'),    mkdir(fileSavePath);   end       % 若不存在，则产生该目录
testSSFSvaryingNumofFeas_SVM_20200711(gX,gY,fileSavePath,fileSaveName,dataName)