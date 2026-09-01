import evaluate
import pandas as pd
import numpy as np
import torch
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import Dataset

from src.settings import *


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

model.eval()

rouge = evaluate.load('rouge')

test_set = Dataset.load_from_disk(DATA_DIR / 'test_tokenized')
#test_set = test_set.select(range(24))

def generation(batch):
    inputs = {
        "input_ids" : torch.tensor(batch['input_ids']).to(model.device),
        "attention_mask" : torch.tensor(batch['attention_mask']).to(model.device),
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens = MAX_TARGET_LENGTH,
            num_beams = 4,
        )

    batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    labels = np.where(
        np.array(batch['labels']) != -100,
        batch['labels'],
        tokenizer.pad_token_id,
    )

    batch_targets = tokenizer.batch_decode(labels, skip_special_tokens=True)

    return {"predictions": batch_predictions, "targets": batch_targets}


print("Running generation on test set...")
evaluation_dataset = test_set.map(
    generation,
    batched=True,
    batch_size=8,
    remove_columns=test_set.column_names, # unnecessary, we only want predicted and target text, no tokens
)

print("Computing evaluation metrics on test set...")
results = rouge.compute(
    predictions=evaluation_dataset['predictions'],
    references=evaluation_dataset['targets'],
    use_aggregator=False,
)

print("_____________ROUGE RESULTS____________")
print(results)

print("Saving evaluation results to disk...")
with open(BASE_DIR / "base-results.json", "w") as f:
    json.dump(results, f)

print("Saving table to csv")

df = pd.DataFrame({
    "predictions" : evaluation_dataset['predictions'],
    "targets" : evaluation_dataset['targets'],
    "rouge1" : results["rouge1"],
    "rouge2" : results["rouge2"],
    "rougeL" : results["rougeL"],
    "rougeLsum" : results["rougeLsum"],
})

print(df.head)

df.to_csv(DATA_DIR / "base_results.csv", index=False)
