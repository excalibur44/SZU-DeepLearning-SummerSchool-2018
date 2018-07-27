caffe.exe train -solver lenet_solver.prototxt

caffe.exe test -model lenet_train_test.prototxt -weights trained/lenet_iter_1000.caffemodel -iterations 100
