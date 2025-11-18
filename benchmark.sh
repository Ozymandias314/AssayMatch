python assaymatch/step1_prepare_trak.py -b benchmark
python scripts/dispatcher_multi.py -c benchmark/run_1/trak_training_config.json -l logs --max_jobs_per_gpu 1
python scripts/dispatcher_multi.py -c benchmark/run_1/trak_training_config_st.json -l logs --max_jobs_per_gpu 1
python scripts/dispatcher_multi.py -c benchmark/run_1/trak_featurization_config.json -l logs --max_jobs_per_gpu 1
python scripts/dispatcher_multi.py -c benchmark/run_1/trak_featurization_config_st.json -l logs --max_jobs_per_gpu 1

python assaymatch/step2_prepare_finetune.py -b benchmark/run_1
python scripts/dispatcher_multi.py -c benchmark/run_1/finetune_config_chemprop.json -l logs --max_jobs_per_gpu 1
python scripts/dispatcher_multi.py -c benchmark/run_1/finetune_config_st.json -l logs --max_jobs_per_gpu 1

python assaymatch/step3_generate_data.py -b benchmark/run_1
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL340_config_chemprop.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL340_config_chemprop.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL340_config_chemprop.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL340_config_chemprop.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL340_config_chemprop.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL340_config_chemprop.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL340_config_st.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL240_config_st.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL247_config_st.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL220_config_st.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL279_config_st.json -l logs --max_jobs_per_gpu 4
python scripts/dispatcher_multi.py -c benchmark/run_1/CHEMBL203_config_st.json -l logs --max_jobs_per_gpu 4