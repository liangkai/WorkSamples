./train.sh 1 256 256 0 0.005 rmsprop 0.5 0.5 5 embedlayer flickr8k -dropout -gpu_mode| tee -a "log_emblayer.txt"
./train.sh 1 256 256 0 0.001 rmsprop 0.5 0.5 2 concatprojlayer flickr8k -gpu_mode -dropout| tee -a "log_concatprojlayer.txt"

./train.sh 1 512 512 0 0.005 rmsprop 0.3 0.3 5 embedlayer coco -dropout -gpu_mode| tee -a "log_emblayer.txt"

