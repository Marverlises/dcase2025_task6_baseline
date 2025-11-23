python -m d25_t6.predict \
--load_ckpt_path=/share/project/baiyu/project/dcase2025_task6_baseline/checkpoints/alignment_audiocaps/last.ckpt \
--retrieval_audio_path=/share/project/baiyu/my_datasets/dcase2025/AudioCaps/AUDIOCAPS/audio_32000Hz/test \
--retrieval_captions=/share/project/baiyu/my_datasets/dcase2025/AudioCaps/AUDIOCAPS/audio_32000Hz/test.csv \
--predictions_path=./test_result