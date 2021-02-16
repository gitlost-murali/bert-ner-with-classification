import sys
from transformers import AutoTokenizer
import json

## ixam-es.json
## ixam-en.json
## ixam-enes.json
with open("config_settings/ixam-testing.json","r") as fh:
    params = json.load(fh)

TOKENIZER = AutoTokenizer.from_pretrained(params["BASE_MODEL_PATH"])
RANDOM_STATE = 42 #Use this for consistent shuffle/splits.