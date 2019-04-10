# For release

declare -a dbName=("Set5" "Set14" "B100" "Urban100" "Manga109")
arrLen=${#dbName[@]}
scale=4
trained="DF2K"
model="DRCA_BIX$scale""_$trained"

# Large Model
for ((i=0;i<$arrLen;i++));
do
	cmd="CUDA_VISIBLE_DEVICES=0 python main.py --self_ensemble --data_test ${dbName[$i]} --scale $scale --model DRCA --n_resgroups 5 --n_resblocks 36 --n_feats 64 --pre_train ../model/$model.pt --test_only --save_results --chop --save 'DRCA_Self_$trained/${dbName[$i]}/X$scale' --testpath ../benchmark"
	eval "$cmd"
done
