export PYTHONPATH=../../:$PYTHONPATH
CUDA_VISIBLE_DEVICES=$2 python -m torch.distributed.launch --nproc_per_node=$1 --master_port 15530 --use_env ../../tools/train_val.py