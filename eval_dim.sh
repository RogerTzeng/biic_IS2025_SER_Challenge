ssl_type=wavlm-large
GPU=0

pool_type=AttentiveStatisticsPooling
for model in dim_ser; do
    for data in Audios; do
        for seed in 7; do
            CUDA_VISIBLE_DEVICES=$GPU python eval_test1/eval_dim_ser_test.py \
                --ssl_type=${ssl_type} \
                --pooling_type=${pool_type} \
                --model_path=model/$model/${seed}  \
                --store_path=result/$model/${seed}.txt \
                --testset=test1 \
                --audio_path=/path/to/MSP-Podcast/$data|| exit 0;
        done
    done
done