import torch
from torch.utils.data import Dataset, DataLoader
from typing import List
import json

from transformers import BertTokenizer
from PIL import Image
import argparse
from pathlib import Path
from tqdm import tqdm
import torchvision.transforms.functional as F
import numpy as np

from datasets import coco, utils
from configuration import Config

class SquarePad:
	def __call__(self, image):
		w, h = image.size
		max_wh = np.max([w, h])
		hp = int((max_wh - w) / 2)
		vp = int((max_wh - h) / 2)
		padding = (hp, vp, hp, vp)
		return F.pad(image, padding, 0, 'constant')



parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--path', type=str, help='path to images', required=True)
parser.add_argument('--v', type=str, help='version', default='v3')
args = parser.parse_args()
images_path = args.path
version = args.v

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    raise NotImplementedError('Version not implemented')

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
config = Config()

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

def create_caption_and_mask(start_token, max_length):
    caption_template = torch.zeros((1, max_length), dtype=torch.long)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template

@torch.no_grad()
def evaluate(image, caption, cap_mask):
    model.eval()
    for i in range(config.max_position_embeddings - 1):
        predictions = model(image, caption, cap_mask)
        predictions = predictions[:, i, :]
        predicted_id = torch.argmax(predictions, axis=-1)

        if predicted_id[0] == 102:
            return caption

        caption[:, i+1] = predicted_id[0]
        cap_mask[:, i+1] = False

    return caption

class TwitterImagesDataset(Dataset):
    def __init__(self, image_names: List[Path], start_token, max_position_embeddings):
        self.image_names = image_names
        self.start_token = start_token
        self.max_position_embeddings = max_position_embeddings
        self.sqpad = SquarePad()

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, item):
        image = self.image_names[item]
        image = self.sqpad(Image.open(image))
        image = F.resize(image, 299)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)
        caption, cap_mask = create_caption_and_mask(
            self.start_token, self.max_position_embeddings)

        return {
            "image": image,
            "caption": caption,
            "cap_mask": cap_mask
        }


def create_data_loader(path_to_images: Path, start_token, max_position_embeddings, batch_size):
    image_names = list(path_to_images.glob("*.jpg"))
    ds = TwitterImagesDataset(image_names, start_token, max_position_embeddings)
    return DataLoader(
        ds, 
        batch_size=batch_size,
        num_workers=1
    )

# loader = create_data_loader(Path(images_path), start_token, config.max_position_embeddings, 16)
# model.to(device)
# for d in tqdm(loader):
#     images = d["image"].to(device)
#     captions = d["caption"].to(device)
#     cap_mask = d["cap_mask"].to(device)
#     output = evaluate(images, captions, cap_mask)

model.to(device)
image_ids = []
captions = []
errors = 0
for image_path in tqdm(list(Path(images_path).glob("*.jpg"))):
    try:
        image = Image.open(image_path)
        image = coco.val_transform(image)
        image = image.unsqueeze(0)
        caption, cap_mask = create_caption_and_mask(
            start_token, config.max_position_embeddings)
        image = image.to(device)
        caption = caption.to(device)
        cap_mask = cap_mask.to(device)
        output = evaluate(image, caption, cap_mask)
        result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        result = result.capitalize()
        image_ids.append(image_path.name)
        captions.append(result)
    except:
        errors += 1
        print(f"Error for {str(image_path)}")
        continue

img_to_caption = {image_id: caption for image_id, caption in zip(image_ids, captions)}
with open(f"{Path(images_path).name}.json", "w") as f:
    json.dump(img_to_caption, f)
print(f"Total of {errors} errors.")