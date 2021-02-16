import numpy as np

import joblib
import torch

import config
import dataset
import engine
from model import EntityModel


if __name__ == "__main__":

    meta_data = joblib.load("demo_meta.bin")
    enc_tag = meta_data["enc_tag"]

    num_tag = len(list(enc_tag.classes_))

    sentence = """
    President Trump has addressed the nation on US supremacy over the world.
    """
    tokenized_sentence = config.params["TOKENIZER"].encode(sentence)

    sentence = sentence.split()
    print(sentence)
    print(tokenized_sentence)
    print(config.params["TOKENIZER"].convert_ids_to_tokens(tokenized_sentence))

    test_dataset = dataset.EntityDataset(
        texts=[sentence],
        tags=[[0] * len(sentence)], O_tag_id= enc_tag.transform(["O"])[0]
    )

    model = EntityModel(num_tag=num_tag)
    model.load_state_dict(torch.load(config.params["MODEL_PATH"]))
    device = torch.device("cuda")
    if torch.cuda.is_available(): model.to(device)

    with torch.no_grad():
        data = test_dataset[0]
        for k, v in data.items():
            data[k] = v.to(device).unsqueeze(0)
        tag, _ = model(**data)

        print(
            enc_tag.inverse_transform(
                tag.argmax(2).cpu().numpy().reshape(-1)
            )[:len(tokenized_sentence)]
        )
