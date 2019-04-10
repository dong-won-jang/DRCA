## train
# BI, scale 2, 4, DIV2K
# DRCA_BIX2_C5R36P48, input=48x48, output=96x96
LOG=./../experiment/DRCA_BIX2_C5R36P48-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model DRCA --save DRCA_BIX2_C5R36P48 --scale 2 --n_resgroups 5 --n_resblocks 36 --n_feats 64  --reset --chop --save_results --print_model --patch_size 96 2>&1 | tee $LOG

# DRCA_BIX4_C5R36P48, input=48x48, output=192x192
LOG=./../experiment/DRCA_BIX4_C5R36P48-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model DRCA --ext sep --save DRCA_BIX4_C5R36P48 --scale 4 --n_resgroups 5 --n_resblocks 36 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/DRCA_BIX2_C5R36P48/model/model_latest.pt 2>&1 | tee $LOG

# BI, scale 4, DF2K
# DRCA_BIX4_C5R36P48_DF2K, input=48x48, output=192x192
LOG=./../experiment/DRCA_BIX4_C5R36P48_DF2K-`date +%Y-%m-%d-%H-%M-%S`.txt
CUDA_VISIBLE_DEVICES=0 python main.py --model DRCA --save DRCA_BIX4_G5R36P48_DF2K --scale 4 --n_resgroups 5 --n_resblocks 36 --n_feats 64  --reset --chop --save_results --print_model --patch_size 192 --pre_train ../experiment/DRCA_BIX2_G5R36P48/model/model_latest.pt --data_train DF2K --data_test DF2K --n_train 3450 --offset_val=3450 --test_every 1100 2>&1 | tee $LOG
