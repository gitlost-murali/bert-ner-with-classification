import config
import torch


class EntityDataset:
    def __init__(self, texts, tags, classification_tags, O_tag_id):
        # texts: [["hi", ",", "my", "name", "is", "abhishek"], ["hello".....]]
        # tags: [[1 2 3 4 1 5], [....].....]]
        # tags: [1, 0]
        # O_tag_id: ID of the entity tag O. Needed for padding and CLS,SEP tokens
        #           For padding, doesn't matter, since we don't attend but anyway.
        self.texts = texts
        self.tags = tags
        self.classification_tags = classification_tags
        self.O_tag_id = O_tag_id

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = self.texts[item]
        tags = self.tags[item]
        classification_tag = self.classification_tags[item][0]
        # print(f"class tag is {classification_tag}")
        ids = []
        target_tag =[]

        for i, s in enumerate(text):
            inputs = config.TOKENIZER.encode(
                s,
                add_special_tokens=False
            )
            # Converting to BPE style. Antiestablishment => Anti ##establish ##ment 
            input_len = len(inputs)
            ids.extend(inputs)
            target_tag.extend([tags[i]] * input_len)

        ids = ids[:config.params["MAX_LEN"] - 2]
        target_tag = target_tag[:config.params["MAX_LEN"] - 2]

        ids = [101] + ids + [102] #Should not be manual. Since these are BERT, it is fine. But still
        target_tag = [self.O_tag_id] + target_tag + [self.O_tag_id] #Change this. Should not be manual for O.
        seq_len = len(ids) 

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        padding_len = config.params["MAX_LEN"] - len(ids)

        ids = ids + ([0] * padding_len)
        mask = mask + ([0] * padding_len)
        token_type_ids = token_type_ids + ([0] * padding_len)
        target_tag = target_tag + ([0] * padding_len) #Is it 0? or O_tag_id

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "target_tag": torch.tensor(target_tag, dtype=torch.long),
            "target_classification_tag": torch.tensor(classification_tag, dtype=torch.long),
            "length": torch.tensor(seq_len),
        }
