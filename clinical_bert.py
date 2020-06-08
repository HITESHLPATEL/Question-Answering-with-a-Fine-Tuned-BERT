import os

def clinical_bert_finetune():
    train_command = """
            python ./BERT_files/run_squad.py \
                --vocab_file=./models/pretrained/clinicalbert/vocab.txt \
                --bert_config_file=./models/pretrained/clinicalbert/bert_config.json \
                --init_checkpoint=./models/pretrained/clinicalbert/model.ckpt-100000 \
                --do_train=True \
                --train_file=./data/relation-train.json \
                --do_predict=True \
                --do_lower_case=False \
                --predict_file=./data/relation-dev.json \
                --train_batch_size=12 \
                --learning_rate=6e-5 \
                --num_train_epochs=2 \
                --max_seq_length=384 \
                --doc_stride=128 \
                --output_dir=./models/output/clinicalbert_train_model/
                """
    # running run_squad script from BERT repo with following configurations
    os.system(train_command)

def clinical_bert_inference():
    inference_command = """
        python ./BERT_files/run_squad.py \
            --vocab_file=./models/pretrained/clinicalbert/vocab.txt \
            --bert_config_file=./models/pretrained/clinicalbert/bert_config.json \
            --init_checkpoint=/models/output/clinicalbert_train_model/model.ckpt-20000 \
            --do_train=False \
            --do_predict=True \
            --do_lower_case=False \
            --predict_file=./data/relation-test.json \
            --train_batch_size=12 \
            --learning_rate=6e-5 \
            --num_train_epochs=2 \
            --max_seq_length=384 \
            --doc_stride=128 \
            --output_dir=./models/output/clinicalbert_test/
            """
    # running run_squad script from BERT repo with following configurations
    os.system(inference_command)

def clinical_bert_evaluation():
    eval_command = """
        python ./squad_file/evaluate-v1.1.py ./data/relation-test.json ./models/output/clinicalbert_test/predictions.json
    """
    # running run_squad script from BERT repo with following configurations
    os.system(eval_command)