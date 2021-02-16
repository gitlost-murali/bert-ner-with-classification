import torch
import copy
import config
from tqdm import tqdm
from typing import List
from sklearn.metrics import classification_report

def train_fn(data_loader, model, optimizer, device, scheduler):
    model.train()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items():  #BioBERT is taking alot of space
            data[k] = v.to(device)
        optimizer.zero_grad()
        _, _, loss = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"],data["target_classification_tag"])
        loss.backward()
        optimizer.step()
        scheduler.step()
        final_loss += loss.item()
    return final_loss / len(data_loader)


def eval_fn(data_loader, model, device):
    model.eval()
    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        for k, v in data.items(): #BioBERT is taking alot of space
            data[k] = v.to(device)
        _,_, loss = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"],data["target_classification_tag"])
        final_loss += loss.item()
    return final_loss / len(data_loader)

def eval_with_metrics(data_loader, model, device, enc_tag):
    model.eval()

    y_pred = []
    y_ground = []

    for data in tqdm(data_loader):
        sentence_lengths = data["length"].tolist()
        for k, v in data.items(): #BioBERT is taking alot of space
            data[k] = v.to(device)
        tags, classify_tags, _ = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"],data["target_classification_tag"])
        for ix, tag in enumerate(tags):
            result = enc_tag.inverse_transform(
                    tag.argmax(1).cpu().numpy()
                )

            y_pred.extend(result[:sentence_lengths[ix]])

            ground_truth_seq = data["target_tag"][ix][:sentence_lengths[ix]].tolist()
            ground_truth = enc_tag.inverse_transform(ground_truth_seq)
            y_ground.extend(ground_truth)

    class_labels = (enc_tag.classes_).tolist()
    class_labels.remove("O")
    metrics = classification_report(y_pred=y_pred, y_true=y_ground, labels= class_labels)

    return metrics

def undo_bpe(bpe_tok_sent: List[str], result: List[str], ground_truth: List[str]):
    """
    Undo the BPE tokenization
    For example,
        Normalization => Norma + ##li + ##za + ##tion
    
    Undo the BPE tokenization along with the tags

    Args:
        bpe_tok_sent ([str]): List of BPE tokenized words
        result ([str]): List of predicted tags for those BPE tokenized words
        ground_truth ([str]): List of ground-truth tags for those BPE tokenized words
    Returns:
        concatenated_bpes ([str]): List of BPE-undone tokenized words
        concatenated_tags ([str]): List of predicted tags for those BPE-undone tokenized words
        concatenated_ground_tags ([str]): List of ground-truth tags for those BPE-undone tokenized words
    """
    prev_tok_tag = ''


    concatenated_bpe = ''
    concatenated_bpes = []
    concatenated_tags = []
    concatenated_ground_tags = []
    
    new_result = []
    for idx, (bpe_tok, tag, ground_tag) in enumerate(zip(bpe_tok_sent, result, ground_truth)):
        if not "##" in bpe_tok:
            if idx!=0: 
                concatenated_bpes.append(concatenated_bpe)
                concatenated_tags.append(main_tok_tag)
                concatenated_ground_tags.append(main_ground_tag)
            concatenated_bpe = ''
            concatenated_bpe+=(bpe_tok).replace("##","")
            main_tok_tag = tag
            main_ground_tag = ground_tag
        else:
            concatenated_bpe+=(bpe_tok).replace("##","")
        
    return concatenated_bpes, concatenated_tags, concatenated_ground_tags

def eval_with_metrics_combined(data_loader, model, device, enc_tag, classify_enc_tag):
    """Calculate Precision, Recall and F-score

    Args:
        data_loader ([generator]): Generator
        model ([pytorch-model]): Pytorch model
        device ([cuda/cpu]): Device - CUDA/CPU
        enc_tag ([LabelEncoder]): Sklearn PreProcessing LabelEncoder

    Returns:
        [type]: [description]
    """
    y_pred = []
    y_ground = []

    classify_y_pred = []
    classify_y_ground = []

    model.eval()
    
    final_loss = 0
    
    for data in tqdm(data_loader):
        sentence_lengths = data["length"].tolist()
        for k, v in data.items(): #BioBERT is taking alot of space
            data[k] = v.to(device)
        tags, classify_tags, loss = model(data["ids"],data["mask"], data["token_type_ids"], data["target_tag"], data["target_classification_tag"])
        
        # Calculating Validation loss
        final_loss += loss.item()

        # Code for Undoing BPE and calculating F-1 score
        for ix, (tag, classify_tag) in enumerate(zip(tags,classify_tags)):

            result = enc_tag.inverse_transform(
                    tag.argmax(1).cpu().numpy()
                )

            result_shortened = result[:sentence_lengths[ix]]

            classified_result = classify_enc_tag.inverse_transform(
                    [classify_tag.argmax(0).cpu().numpy()]
                )

            ground_truth_seq = data["target_tag"][ix][:sentence_lengths[ix]].tolist()
            ground_truth = enc_tag.inverse_transform(ground_truth_seq)

            text_sentence = config.TOKENIZER.convert_ids_to_tokens(data["ids"][ix][:sentence_lengths[ix]].tolist())
            
            
            text, res, groundtruth = undo_bpe(text_sentence, result_shortened, ground_truth)
            
            y_pred.extend(res)
            y_ground.extend(groundtruth)


            classify_ground_truth_seq = data["target_classification_tag"][ix].item()
            classify_ground_truth = classify_enc_tag.inverse_transform([classify_ground_truth_seq])

            classify_y_pred.extend(classified_result)
            classify_y_ground.extend(classify_ground_truth)

    class_labels = copy.deepcopy(enc_tag.classes_).tolist()
    class_labels.remove("O")
    metrics = classification_report(y_pred=y_pred, y_true=y_ground, labels= class_labels, output_dict=True)
    val_loss = final_loss / len(data_loader)

    classify_class_labels = copy.deepcopy(classify_enc_tag.classes_).tolist()
    classification_metrics = classification_report(y_pred=classify_y_pred, y_true=classify_y_ground, labels= classify_class_labels, output_dict=True)

    return val_loss, metrics, classification_metrics
