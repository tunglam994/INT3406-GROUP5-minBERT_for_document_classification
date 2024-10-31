mkdir -p ANDREWID

python3 classifier.py --option pretrain --epochs 10 --lr 1e-3 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt

