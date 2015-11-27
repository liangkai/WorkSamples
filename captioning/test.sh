train_file=${1-output_train.pred}
val_file=${2-output_val.pred}
test_file=${3-output_test.pred}

# Do flickr8k predictions
cd predictions/bleu/flickr8k
echo 'FLICKR8k ===============   ' 
echo 'bleu score on train set is '
perl multi-bleu.pl gold_standard_train < $train_file

echo 'bleu score on val set is '
perl multi-bleu.pl gold_standard_val < $val_file

echo 'bleu score on test set is '
perl multi-bleu.pl gold_standard_test < $test_file
cd ../../../

# Do coco predictions
cd predictions/bleu/coco
echo 'COCO ===============   ' 
echo 'bleu score on train set is '
perl multi-bleu.pl gold_standard_train < $train_file

echo 'bleu score on val set is '
perl multi-bleu.pl gold_standard_val < $val_file

echo 'bleu score on test set is '
perl multi-bleu.pl gold_standard_test < $test_file
cd ../../../

