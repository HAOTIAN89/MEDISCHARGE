BHC_FILE=/home/haotian/make-discharge-me/data/infered/bhc_7b_infered_5.csv
DI_FILE=/home/haotian/make-discharge-me/data/test_phase_1/DI_test_dataset_sub_5.jsonl

python3 DI_construct.py \
    --generated_bhc_test  $BHC_FILE \
    --constructed_di_test $DI_FILE