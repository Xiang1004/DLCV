import torch
from transformers import BertTokenizer
from PIL import Image
import argparse
import matplotlib.pyplot as plt
import glob
import os
import numpy as np

from models import caption
from datasets import coco
from configuration import Config

parser = argparse.ArgumentParser(description='Image Captioning')
parser.add_argument('--image_path', type=str)
parser.add_argument('--output_path', type=str)
parser.add_argument('--v', type=str, help='version', default='v3')
parser.add_argument('--checkpoint', type=str, help='checkpoint path', default=None)
config = parser.parse_args()

image_path = config.image_path
output_path = config.output_path
version = config.v
checkpoint_path = config.checkpoint

image_file = glob.glob(os.path.join(image_path, '*'))

if version == 'v1':
    model = torch.hub.load('saahiluppal/catr', 'v1', pretrained=True)
elif version == 'v2':
    model = torch.hub.load('saahiluppal/catr', 'v2', pretrained=True)
elif version == 'v3':
    model = torch.hub.load('saahiluppal/catr', 'v3', pretrained=True)
else:
    print("Checking for checkpoint.")
    if checkpoint_path is None:
        raise NotImplementedError('No model to chose from!')
    else:
        if not os.path.exists(checkpoint_path):
            raise NotImplementedError('Give valid checkpoint path')
        print("Found checkpoint! Loading!")
        model, _ = caption.build_model(config)
        print("Loading Checkpoint...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

start_token = tokenizer.convert_tokens_to_ids(tokenizer._cls_token)
end_token = tokenizer.convert_tokens_to_ids(tokenizer._sep_token)

for pic_num in range(len(image_file)):
    img = Image.open(image_file[pic_num])
    pic_name = image_file[pic_num].split('/')[-1].split('.')[0]
    image = coco.val_transform(img)
    image = image.unsqueeze(0)

    def create_caption_and_mask(start_token, max_length):
        caption_template = torch.zeros((1, max_length), dtype=torch.long)
        mask_template = torch.ones((1, max_length), dtype=torch.bool)

        caption_template[:, 0] = start_token
        mask_template[:, 0] = False

        return caption_template, mask_template

    caption, cap_mask = create_caption_and_mask(start_token, Config().max_position_embeddings)

    def plot_attention(image, mask):
        # 拆解句子
        word = []
        letter = ''
        for i in range(len(result)):
            if result[i] == '.':
                word.append(letter)
                word.append('<end>')
                break
            elif result[i] == ' ':
                word.append(letter)
                letter = ''
            else:
                letter = letter + result[i]

        temp_image = np.array(image)
        fig = plt.figure(figsize=(10, 10))
        # Original image
        ax = fig.add_subplot((len(word) // 4) + 1, 4, 1)
        ax.set_title('<start>')
        ax.imshow(temp_image)

        for i in range(len(word)):
            temp_att = np.resize(mask[i], (19, 19))
            ax = fig.add_subplot((len(word) // 4) + 1, 4, i + 2)
            ax.set_title(word[i])
            img = ax.imshow(temp_image)
            ax.imshow(temp_att, cmap='jet', alpha=0.6, interpolation='sinc', extent=img.get_extent())

        plt.tight_layout()
        picture_name = pic_name + '.png'
        fig.savefig(os.path.join(output_path, picture_name))
        print(result)


    @torch.no_grad()
    def evaluate():
        model.eval()
        _, atsizes = model(image, caption, cap_mask)
        atsize = atsizes[0].size(2)
        mask = np.zeros((Config().max_position_embeddings, atsize))
        for i in range(Config().max_position_embeddings - 1):
            predictions, attention_weights = model(image, caption, cap_mask)

            predictions = predictions[:, i, :]
            predicted_id = torch.argmax(predictions, axis=-1)

            attention_weights = torch.stack(list(attention_weights), dim=0)
            attention_weight = attention_weights[5, :, i, :]
            mask[i] = attention_weight.numpy()

            if predicted_id[0] == 102:
                return caption, mask

            caption[:, i + 1] = predicted_id[0]
            cap_mask[:, i + 1] = False

        return caption, mask


    output, mask = evaluate()
    result = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
    plot_attention(img, mask)
