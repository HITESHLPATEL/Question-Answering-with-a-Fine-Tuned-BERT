from bert_base import *
from biobert import *
from clinical_bert import *
from datatset_prep import *

print("########### Begin dataset preparation ###############")

dataset_prep(data_dir="./data",
            filename="relation-train.json",
            output_dir="./data")


print("########### begin finetuning BERT base model #############")
#fine tune clinical bert
bert_base_finetune()

#Inference on clinical bert
bert_base_inference()

print("########### BERT base model performance #############")
#Evaluation on clinical bert
bert_base_evaluation()


print("########### begin finetuning biobert model #############")
#fine tune biobert
biobert_finetune()

#Inference on biobert
biobert_inference()

print("########### biobert model performance #############")
#Evaluation on biobert
biobert_evaluation()


print("########### begin finetuning Clinical BERT model #############")
#fine tune clinical bert
clinical_bert_finetune()

#Inference on clinical bert
clinical_bert_inference()

print("########### Clinical BERT model performance #############")
#Evaluation on clinical bert
clinical_bert_evaluation()

