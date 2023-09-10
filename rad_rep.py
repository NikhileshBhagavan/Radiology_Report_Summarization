from transformers import BartTokenizer, BartForConditionalGeneration
import locale
import numpy as np
from evaluate import load
from radgraph import F1RadGraph
from sagemaker.huggingface import HuggingFace
from preprocessing import clean_and_process
from transformers import pipeline, Trainer, TrainingArguments
import pandas as pd
from rouge_score import calculate_rouge_scores
import torch
import gc


def preprocess_all():
    # Step-1

    clean_and_process("./rrs-mimiciii/rrs-mimiciii/all/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/all/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/all/test.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/all/train.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/all/train.impression.tok", "./rrs-mimiciii/rrs-mimiciii/all/train.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/all/validate.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/all/validate.impression.tok", "./rrs-mimiciii/rrs-mimiciii/all/validate.csv")
    ##
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/test.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/train.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/train.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/train.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/validate.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/validate.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_abdomen-pelvis/validate.csv")
    ##

    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_chest/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_chest/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_chest/test.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_chest/train.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_chest/train.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_chest/train.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_chest/validate.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_chest/validate.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_chest/validate.csv")

    ##

    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_head/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_head/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_head/test.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_head/train.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_head/train.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_head/train.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_head/validate.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_head/validate.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_head/validate.csv")

    ##

    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_neck/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_neck/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_neck/test.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_neck/train.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_neck/train.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_neck/train.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_neck/validate.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_neck/validate.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_neck/validate.csv")

    ##
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_sinus/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_sinus/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_sinus/test.csv")

    ##

    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_spine/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_spine/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_spine/test.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_spine/train.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_spine/train.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_spine/train.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/CT_spine/validate.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/CT_spine/validate.impression.tok", "./rrs-mimiciii/rrs-mimiciii/CT_spine/validate.csv")
    ##
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/MR_abdomen/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/MR_abdomen/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/MR_abdomen/test.csv")

    ##
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/MR_neck/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/MR_neck/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/MR_neck/test.csv")

    ##
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/MR_pelvis/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/MR_pelvis/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/MR_pelvis/test.csv")

    ##
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/MR_spine/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/MR_spine/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/MR_spine/test.csv")

    ##

    clean_and_process("./rrs-mimiciii/rrs-mimiciii/MR_head/test.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/MR_head/test.impression.tok", "./rrs-mimiciii/rrs-mimiciii/MR_head/test.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/MR_head/train.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/MR_head/train.impression.tok", "./rrs-mimiciii/rrs-mimiciii/MR_head/train.csv")
    clean_and_process("./rrs-mimiciii/rrs-mimiciii/MR_head/validate.findings.tok",
                      "./rrs-mimiciii/rrs-mimiciii/MR_head/validate.impression.tok", "./rrs-mimiciii/rrs-mimiciii/MR_head/validate.csv")


preprocess_all()

# from transformers import PegasusTokenizer, PegasusForConditionalGeneration

# model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
# tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-large")


model = BartForConditionalGeneration.from_pretrained("facebook/bart-large-cnn")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")


train_new_dataset = pd.read_csv(
    "./rrs-mimiciii/rrs-mimiciii/MR_head/train.csv")
train_df = train_new_dataset
# print(train_df)
validate_new_dataset = pd.read_csv(
    "./rrs-mimiciii/rrs-mimiciii/MR_head/validate.csv")
val_df = validate_new_dataset
# print(val_df)

train_encodings = tokenizer(
    list(train_df['text']), truncation=True, padding=True, max_length=1024)

train_label_encodings = tokenizer(
    list(train_df['summary']), truncation=True, padding=True, max_length=1024)

val_encodings = tokenizer(
    list(val_df['text']), truncation=True, padding=True, max_length=1024)
val_label_encodings = tokenizer(
    list(val_df['summary']), truncation=True, padding=True, max_length=1024)


train_dataset = torch.utils.data.TensorDataset(
    torch.tensor(train_encodings['input_ids']),
    torch.tensor(train_encodings['attention_mask']),
    torch.tensor(train_label_encodings['input_ids']),
    torch.tensor(train_label_encodings['attention_mask']),
    torch.tensor(train_label_encodings['input_ids'])
)

val_dataset = torch.utils.data.TensorDataset(
    torch.tensor(val_encodings['input_ids']),
    torch.tensor(val_encodings['attention_mask']),
    torch.tensor(val_label_encodings['input_ids']),
    torch.tensor(val_label_encodings['attention_mask']),
    torch.tensor(val_label_encodings['input_ids'])
)

# here we can also use early stopping criteria too
# warmup steps should be around 5-10% of total training steps
# we can use load best model at end parameter or we can save checkpoints manually and get best checkpoint saved based on eval loss
# the metric used is eval_loss we can change it to rougel
training_args = TrainingArguments(
    output_dir='./results/MR_head',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=1,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_strategy="epoch",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-5,
)

# compute_metrics=compute_metrics,
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=lambda data: {'input_ids': torch.stack([f[0] for f in data]),
                                'attention_mask': torch.stack([f[1] for f in data]),
                                'decoder_input_ids': torch.stack([f[2] for f in data]),
                                'decoder_attention_mask': torch.stack([f[3] for f in data]),
                                'labels': torch.stack([f[4] for f in data])},

)


trainer.train()
gc.collect()

# # history = trainer.state.log_history
# # history
# # for i in range(1,31,2):
# #   print("Epoch "+str(history[i]["epoch"])+" validation loss: "+str(history[i]["eval_loss"]),end="\n")

# model = PegasusForConditionalGeneration.from_pretrained(
#     "/content/drive/MyDrive/lbp_final/colab/colab_results/checkpoint-9")

# summarizer = pipeline("text2text-generation", model=model,
#                       tokenizer="google/pegasus-large")

# # # model = BartForConditionalGeneration.from_pretrained(
# # #     "/content/drive/MyDrive/lbp_final/colab/colab_results/checkpoint-3")

# # # summarizer = pipeline("text2text-generation", model=model,
# # #                       tokenizer="facebook/bart-large-cnn")

# # TO calculate F1RadGraph Score:


# def calculate_rad_graph_scores(hyps, refs):
#     f1radgraph = F1RadGraph(reward_level="partial")
#     score, _, hypothesis_annotation_lists, reference_annotation_lists = f1radgraph(hyps=hyps,
#                                                                                    refs=refs)
#     return score


# # To calculate BertScore:
# #!pip install numpy


# def calculate_bert_scores(predictions, references):
#     bertscore = load("bertscore")
#     results = bertscore.compute(
#         predictions=predictions, references=references, lang="en")
#     return np.average(results["f1"], weights=None)

# # For MIMIC-III, the evaluation metrics will be BLEU, ROUGE, ROUGE-L and F1RadGraph.
# # We have to pass bassline scores of bertscore, f1radgraph,rougel


# # TO Calculate rougeL score:
# df_test = pd.read_csv(
#     "/content/drive/MyDrive/lbp_final/colab/rrs-mimiciii/rrs-mimiciii/all/test.csv")


# input_text = list(df_test['text'])
# input_text = input_text[:10]

# # print(input_text)


# ref_summaries = list(df_test['summary'])
# ref_summaries = ref_summaries[:10]

# # print(ref_summaries)


# candidate_summaries = []
# # num_beams,length_penalty,early_stopping based on previous research paper
# for str in input_text:
#     summary_text = summarizer(str, max_length=1024,
#                               min_length=10, do_sample=False, num_beams=5, length_penalty=0.8, early_stopping=True)[0]['generated_text']
#     candidate_summaries.append(summary_text)


# print(calculate_rouge_scores(candidate_summaries, ref_summaries))

# print(calculate_rad_graph_scores(candidate_summaries, ref_summaries))

# locale.getpreferredencoding = lambda: "UTF-8"
# print(calculate_bert_scores(candidate_summaries, ref_summaries))
