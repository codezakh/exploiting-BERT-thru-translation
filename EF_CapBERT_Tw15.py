import transformers
from transformers import (
    BertModel,
    BertTokenizer,
    AdamW,
    get_linear_schedule_with_warmup,
)
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from collections import defaultdict
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
import json
import random


# Put model inputs here. #
train_tsv = "/home/zaid/Desktop/TomBERTDatasetTSVs/twitter2015/train.tsv"
dev_tsv = "/home/zaid/Desktop/TomBERTDatasetTSVs/twitter2015/dev.tsv"
test_tsv = "/home/zaid/Desktop/TomBERTDatasetTSVs/twitter2015/test.tsv"
# DOUBLE CHECK THE CAPTIONS FILE IS FOR THE RIGHT DATASET! #
captions_json = "/home/zaid/Desktop/TwitterDSCaptions/twitter2015_images.json"
PRE_TRAINED_MODEL_NAME = "bert-base-uncased"
MAX_LEN = 80
BATCH_SIZE = 16
DROPOUT_PROB = 0.1
NUM_CLASSES = 3
DEVICE = "cuda:1"
EPOCHS = 6
LEARNING_RATE = 5e-5
ADAMW_CORRECT_BIAS = True
NUM_WARMUP_STEPS = 0
NUM_RUNS = 10
RANDOM_SEEDS = list(range(NUM_RUNS))
# # # # # # # # # # # # #



# Load and massage the dataframes.
test_df = pd.read_csv(test_tsv, sep="\t")
train_df = pd.read_csv(train_tsv, sep="\t")
val_df = pd.read_csv(dev_tsv, sep="\t")

test_df = test_df.rename(
    {
        "index": "sentiment",
        "#1 ImageID": "image_id",
        "#2 String": "tweet_content",
        "#2 String.1": "target",
    },
    axis=1,
)
train_df = train_df.rename(
    {
        "#1 Label": "sentiment",
        "#2 ImageID": "image_id",
        "#3 String": "tweet_content",
        "#3 String.1": "target",
    },
    axis=1,
).drop(["index"], axis=1)
val_df = val_df.rename(
    {
        "#1 Label": "sentiment",
        "#2 ImageID": "image_id",
        "#3 String": "tweet_content",
        "#3 String.1": "target",
    },
    axis=1,
).drop(["index"], axis=1)

# Load the image captions.
with open(captions_json, "r") as f:
    image_captions = json.load(f)


# Instantiate the tokenizer.
tokenizer = BertTokenizer.from_pretrained(PRE_TRAINED_MODEL_NAME)


# Construct the dataset.
class TwitterDataset(Dataset):
    def __init__(
        self,
        tweets: np.array,
        labels: np.array,
        sentiment_targets: np.array,
        image_ids: np.array,
        image_captions,
        tokenizer,
        max_len: int,
    ):
        """
        Downstream code expects reviews and targets to be NumPy arrays.
        """
        self.tweets = tweets
        self.labels = labels
        self.tokenizer = tokenizer
        self.sentiment_targets = sentiment_targets
        self.image_captions = image_captions
        self.max_len = max_len
        self.image_ids = image_ids

    def __len__(self):
        return len(self.tweets)

    def __getitem__(self, item):
        tweet = str(self.tweets[item])
        label = self.labels[item]
        sentiment_target = self.sentiment_targets[item]
        try:
            caption = self.image_captions[self.image_ids[item]]
        except KeyError:  # A couple of the images have no content.
            caption = ""

        encoding = self.tokenizer.encode_plus(
            tweet,
            text_pair=sentiment_target + "." + caption,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt",
            truncation=True,
        )

        return {
            "review_text": tweet,
            "sentiment_targets": sentiment_target,
            "caption": caption,
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "targets": torch.tensor(label, dtype=torch.long),
        }


# Construct the data loaders.
def create_data_loader(df, tokenizer, max_len, batch_size, image_captions):
    ds = TwitterDataset(
        tweets=df.tweet_content.to_numpy(),
        labels=df.sentiment.to_numpy(),
        sentiment_targets=df.target.to_numpy(),
        image_ids=df.image_id.to_numpy(),
        image_captions=image_captions,
        tokenizer=tokenizer,
        max_len=max_len,
    )
    return DataLoader(ds, batch_size=batch_size, num_workers=2)


train_data_loader = create_data_loader(
    train_df, tokenizer, MAX_LEN, BATCH_SIZE, image_captions
)
val_data_loader = create_data_loader(
    val_df, tokenizer, MAX_LEN, BATCH_SIZE, image_captions
)
test_data_loader = create_data_loader(
    test_df, tokenizer, MAX_LEN, BATCH_SIZE, image_captions
)


# Construct and instantiate the classifier.
class SentimentClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(PRE_TRAINED_MODEL_NAME)
        self.drop = nn.Dropout(p=DROPOUT_PROB)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        outputs = self.drop(outputs.pooler_output)
        return self.out(outputs)


# Set up the device.
if torch.cuda.is_available():
    device = torch.device(DEVICE)
    print(f"Using {DEVICE}.")
else:
    device = torch.device("cpu")
    print(f"CUDA not available, using CPU.")



def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0

    for d in data_loader:
        input_ids = d["input_ids"].to(device)
        attention_mask = d["attention_mask"].to(device)
        targets = d["targets"].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        _, preds = torch.max(outputs, dim=1)
        loss = loss_fn(outputs, targets)

        correct_predictions += torch.sum(preds == targets).item()
        losses.append(loss.item())

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return correct_predictions / n_examples, np.mean(losses)


def format_eval_output(rows):
    tweets, targets, labels, predictions = zip(*rows)
    tweets = np.vstack(tweets)
    targets = np.vstack(targets)
    labels = np.vstack(labels)
    predictions = np.vstack(predictions)
    results_df = pd.DataFrame()
    results_df["tweet"] = tweets.reshape(-1).tolist()
    results_df["target"] = targets.reshape(-1).tolist()
    results_df["label"] = labels
    results_df["prediction"] = predictions
    return results_df


def eval_model(model, data_loader, loss_fn, device, n_examples, detailed_results=False):
    model = model.eval()

    losses = []
    correct_predictions = 0
    rows = []

    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            targets = d["targets"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            _, preds = torch.max(outputs, dim=1)
            loss = loss_fn(outputs, targets)

            correct_predictions += torch.sum(preds == targets).item()
            losses.append(loss.item())
            rows.extend(
                zip(
                    d["review_text"],
                    d["sentiment_targets"],
                    d["targets"].numpy(),
                    preds.cpu().numpy(),
                )
            )

        if detailed_results:
            return (
                correct_predictions / n_examples,
                np.mean(losses),
                format_eval_output(rows),
            )

    return correct_predictions / n_examples, np.mean(losses)


results_per_run = {}
for run_number in range(NUM_RUNS):

    np.random.seed(RANDOM_SEEDS[run_number])
    torch.manual_seed(RANDOM_SEEDS[run_number])
    
    # Setup the model, test it with a single batch.
    data = next(iter(train_data_loader))
    model = SentimentClassifier(NUM_CLASSES)
    model.to(device)
    input_ids = data["input_ids"].to(device)
    attention_mask = data["attention_mask"].to(device)
    model(input_ids, attention_mask)

    # Configure the optimizer and scheduler.
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE, correct_bias=ADAMW_CORRECT_BIAS)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=NUM_WARMUP_STEPS, num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCHS):
        print(f"Epoch {epoch + 1}/{EPOCHS} -- RUN {run_number}")
        print("-" * 10)
        train_acc, train_loss = train_epoch(
            model, train_data_loader, loss_fn, optimizer, device, scheduler, len(train_df)
        )

        print(f"Train loss {train_loss} accuracy {train_acc}")
        val_acc, val_loss = eval_model(model, val_data_loader, loss_fn, device, len(val_df))

        print(f"Val   loss {val_loss} accuracy {val_acc}")


    test_acc, _, detailed_results = eval_model(
        model, test_data_loader, loss_fn, device, len(test_df), detailed_results=True
    )
    macro_f1 = f1_score(
        detailed_results.label, detailed_results.prediction, average="macro"
    )

    print(f"TEST ACC = {test_acc}\nMACRO F1 = {macro_f1}")

    results_per_run[run_number] = {
        "accuracy": test_acc,
        "macro-f1": macro_f1
    }

with open('./results_per_run.json', 'w+') as f:
    json.dump(results_per_run, f)

print(f"AVERAGE ACC = {np.mean([_['accuracy'] for _ in results_per_run.values()])}")
print(f"AVERAGE MAC-F1= {np.mean([_['macro-f1'] for _ in results_per_run.values()])}")