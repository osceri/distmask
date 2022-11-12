RESUME_DIR='./output_dir'
DEVICE='cpu'

python3 run.py \
	--batch_size 1 \
	--resume_dir ${RESUME_DIR} \
	--data_length_test 100 \
	--data_length_train 100
