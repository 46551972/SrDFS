%%%%%%%-------------���ݶ���---- dataX ��ÿһ�б�ʾһ������----------
clear; clc;

load('ORL_32x32');    dataName = 'ORL_32x32';  
gY = gnd(:);   gX = fea';    gX = NormalizeFea(double(gX),0); 

fileSaveName = [dataName,'_SVM_Varying_ldc']; %% filename to save result
fileSavePath = 'C:\SSFSexp\';  %% path to save result
if ~exist(fileSavePath,'dir'),    mkdir(fileSavePath);   end       % �������ڣ��������Ŀ¼
testSSFSvaryingNumofFeas_SVM_20200711(gX,gY,fileSavePath,fileSaveName,dataName)