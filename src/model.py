import config
import torch
import transformers
import torch.nn as nn

lfn = nn.CrossEntropyLoss()

def loss_fn(output, target, mask, num_labels):
    lfn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(lfn.ignore_index).type_as(target)
    )
    loss = lfn(active_logits, active_labels)
    return loss


class EntityModel(nn.Module):
    def __init__(self, num_tag, num_classify_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_classify_tag = num_classify_tag
        self.bert = transformers.AutoModel.from_pretrained(config.params['BASE_MODEL_PATH'])
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(config.params['BASE_MODEL_DIM'], self.num_tag) 
        self.out_classify_tag = nn.Linear(config.params['BASE_MODEL_DIM'], self.num_classify_tag) 
        # BASE_MODEL_DIM => 1024 for bio & 768 for BERT

    def save_pretrained_model(self, new_model_path):
        self.bert.save_pretrained(new_model_path)

    def forward(self, ids, mask, token_type_ids, target_tag, target_classification_tag):
        o1, o2 = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bo_tag = self.bert_drop_1(o1)
        bo_classify_tag = self.bert_drop_2(o2)

        tag = self.out_tag(bo_tag)
        classify_tag = self.out_classify_tag(bo_classify_tag)

        loss_tag = loss_fn(tag, target_tag, mask, self.num_tag)

        loss_classify_tag = lfn(classify_tag,target_classification_tag)

        loss = (loss_tag+loss_classify_tag)/2 # Add Classification later

        return tag, classify_tag, loss
