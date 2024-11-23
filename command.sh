mkdir -p ANDREWID

python3 classifier.py --option pretrain --epochs 10 --lr 1e-3 --train data/sst-train.txt --dev data/sst-dev.txt --test data/sst-test.txt

#use cpu
python classifier.py --train data/book_summaries_train.csv --dev data/book_summaries_valid.csv --test data/book_summaries_test.csv --dev_out book_summaries_valid_output.txt --test_out book_summaries_test_output.txt --epochs 5 --batch_size 32 --lr 0.001 --option finetune

#use gpu
python classifier.py --train data/book_summaries_train.csv --dev data/book_summaries_valid.csv --test data/book_summaries_test.csv --use_gpu --dev_out book_summaries_valid_output.txt --test_out book_summaries_test_output.txt --epochs 5 --batch_size 32 --lr 0.0001 --option finetune

#use cpu and author embedding
python classifier.py --train data/book_summaries_train.csv --dev data/book_summaries_valid.csv --test data/book_summaries_test.csv --use_author --dev_out book_summaries_valid_output.txt --test_out book_summaries_test_output.txt --epochs 5 --batch_size 32 --lr 0.0001 --option finetune

#use gpu
python classifier.py --train data/book_summaries_train.csv --dev data/book_summaries_valid.csv --test data/book_summaries_test.csv --use_author --use_gpu --dev_out book_summaries_valid_output.txt --test_out book_summaries_test_output.txt --epochs 5 --batch_size 32 --lr 0.0001 --option finetune