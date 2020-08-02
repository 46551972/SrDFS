%This make.m is used under Windows

mex -O -c  svm.cpp
mex -O -c  svm_model_matlab.c
mex -O  svmtrain.c svm.obj svm_model_matlab.obj
mex -O  svmpredict.c svm.obj svm_model_matlab.obj
mex -O  read_sparse.c

mex -O -c -largeArrayDims svm.cpp
mex -O -c -largeArrayDims svm_model_matlab.c
mex -O -largeArrayDims svmtrain.c svm.obj svm_model_matlab.obj
mex -O -largeArrayDims svmpredict.c svm.obj svm_model_matlab.obj
mex -O -largeArrayDims read_sparse.c

mex -g -c -largeArrayDims svm.cpp
mex -g -c -largeArrayDims svm_model_matlab.c
mex -g -largeArrayDims svmtrain.c svm.obj svm_model_matlab.obj
mex -g -largeArrayDims svmpredict.c svm.obj svm_model_matlab.obj
mex -g -largeArrayDims read_sparse.c
