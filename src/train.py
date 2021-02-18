import pandas as pd
import numpy as np

import joblib
import torch

from sklearn import preprocessing
from sklearn import model_selection

from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

import config
import dataset
import engine
import json
from model import EntityModel

import wandb
# 1. Start a new run
wandb.init(project=config.params["language"],config=config.params)

def read_bilou(data_path):
    with open(data_path,'r') as fh:
        bilou_data = json.load(fh)
    
    sentences = []
    sent_tags = []
    sentences = [sent for sent,tags in bilou_data]
    sent_tags = [tags for sent,tags in bilou_data]
    enc_tag = preprocessing.LabelEncoder()
    tags_list = []
    for tags in sent_tags:
        tags_list.extend(tags)
 
    enc_tag.fit(tags_list)
    sent_tags = [enc_tag.transform(tags) for tags in sent_tags]

    return sentences, sent_tags, enc_tag

def read_prof_bilou(data_path):
    with open(data_path,'r',encoding="utf-8") as fh:
        bilou_data = json.load(fh)
    
    sentences = []
    sent_tags = []
    classification_tags = []
    sentences = [sent for sent,tags,classification_tag in bilou_data]
    sent_tags = [tags for sent,tags,classification_tag in bilou_data]
    classification_tags = [classification_tag for sent,tags,classification_tag in bilou_data]

    enc_tag = preprocessing.LabelEncoder()
    classify_enc_tag = preprocessing.LabelEncoder()

    tags_list = []
    for tags in sent_tags:
        tags_list.extend(tags)
 
    classification_tags_list = []
    for tags in classification_tags:
        classification_tags_list.extend([tags])

    enc_tag.fit(tags_list)
    classify_enc_tag.fit(classification_tags_list)

    sent_tags = [enc_tag.transform(tags) for tags in sent_tags]
    classification_tags = [classify_enc_tag.transform([classification_tag]) for classification_tag in classification_tags]

    return sentences, sent_tags, classification_tags,enc_tag, classify_enc_tag


def process_data_conll(data_path):
    """To read CONLL type data

    Args:
        data_path ([str]): File name

    Returns:
        Sentences, Sentence_wise_tags, Encoding Tags
    """
    df = pd.read_csv(data_path, encoding="latin-1", sep="\t")
    sentences = []
    sent_tags = []
    tmp_snt = [] # To stop collecting words after a line break i.e Instance Boundary
    tmp_tag = [] # To stop collecting words after a line break i.e Instance Boundary

    df = df.fillna("-1")

    for wrd, tg in df.values:
        if wrd=="-1" and tg=="-1": 
            sentences.append(tmp_snt)
            sent_tags.append(tmp_tag)
            tmp_snt = []
            tmp_tag = []
            continue

        tmp_snt.append(wrd)
        tmp_tag.append(tg)

    enc_tag = preprocessing.LabelEncoder()
    tags_list = []
    for tags in sent_tags:
        tags_list.extend(tags)
 
    enc_tag.fit(tags_list)
    sent_tags = [enc_tag.transform(tags) for tags in sent_tags]

    return sentences, sent_tags, enc_tag


if __name__ == "__main__":
    
    train_sentences, train_tag, train_classify_tag, enc_tag, classify_enc_tag = read_prof_bilou(config.params["TRAINING_FILE"])
    test_sentences, test_tag, test_classify_tag, _, _ = read_prof_bilou(config.params["VALIDATION_FILE"])

    meta_data = {
        "enc_tag": enc_tag,
        "classify_enc_tag": classify_enc_tag
    }

    joblib.dump(meta_data, "meta.bin")

    num_tag = len(list(enc_tag.classes_))
    num_classify_tag = len(list(classify_enc_tag.classes_))

    train_dataset = dataset.EntityDataset(
        texts=train_sentences, tags=train_tag, classification_tags=train_classify_tag, O_tag_id= enc_tag.transform(["O"])[0]
    )

    train_data_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.params["TRAIN_BATCH_SIZE"], num_workers=4
    )

    valid_dataset = dataset.EntityDataset(
        texts=test_sentences, tags=test_tag, classification_tags=test_classify_tag, O_tag_id= enc_tag.transform(["O"])[0]
    )

    valid_data_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=config.params["VALID_BATCH_SIZE"], num_workers=1
    )
    
    model = EntityModel(num_tag=num_tag, num_classify_tag= num_classify_tag)
    device = torch.device("cuda" if config.params["CUDA"] else "cpu")
    model.to(device) #BioBERT is taking alot of space

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    num_train_steps = int(len(train_sentences) / config.params["TRAIN_BATCH_SIZE"] * config.params["EPOCHS"])
    optimizer = AdamW(optimizer_parameters, lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )

    best_loss = np.inf
    for epoch in range(config.params["EPOCHS"]):
        train_loss = engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        test_loss, metrics, classification_metrics = engine.eval_with_metrics_combined(valid_data_loader, model, device, enc_tag, classify_enc_tag)

        df_metrics = pd.DataFrame.from_dict(metrics)
        df_metrics = df_metrics.transpose()

        weighted_average = df_metrics["f1-score"][-1]
        micro_average = df_metrics["f1-score"][-3]

        table = wandb.Table(dataframe=df_metrics.transpose())
        wandb.log({"Train loss":train_loss, "Valid loss": test_loss})
        wandb.log({"Validation Metric details": table})
        wandb.log({"Weighted average": weighted_average, "Micro average": micro_average})

        print(f"Train Loss = {train_loss} Valid Loss = {test_loss}, \nMetrics = {df_metrics}, \nClassify Metrics = {pd.DataFrame.from_dict(classification_metrics)} ")
        if test_loss < best_loss:
            torch.save(model.state_dict(), config.params["MODEL_PATH"])
            model.save_pretrained_model(config.params["BASE_MODEL_PATH"]+"_finetuned_"+config.params["language"])
            config.TOKENIZER.save_pretrained(config.params["BASE_MODEL_PATH"]+"_finetuned_"+config.params["language"])
            best_loss = test_loss
            wandb.run.summary["best_valloss"] = best_loss
