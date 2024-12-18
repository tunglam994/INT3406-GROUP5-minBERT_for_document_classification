mkdir -p ANDREWID

python classifier.py --option pretrain --pretrained_model models/supervised_CL_bert.pt --epochs 5 --batch_size 32 --lr 0.0001 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt

python classifier.py --option pretrain --epochs 5 --batch_size 32 --lr 0.0001 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt

python classifier_unsupervised_CL.py --train data_small/sample_wiki1m.txt --dev data_small/sst-dev.txt --test data_small/sst-test.txt --epochs 5 --batch_size 32 --lr 0.0001 --option finetune

python classifier_supervised_CL.py --train data_small/sample_nli_for_simcse.csv --dev data_small/sample_nli_for_simcse.csv --test data_small/sample_nli_for_simcse.csv --epochs 5 --batch_size 32 --lr 0.0001 --option finetune --temperature 0.05