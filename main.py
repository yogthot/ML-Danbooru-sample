#!/usr/bin/env python

import torch
import torchvision.transforms as transforms
from PIL import Image
import json

from caformer import build_caformer

class MlDanbooru:
    TAG_COUNT = 12547
    
    def __init__(self, path):
        with open('class.json', 'r') as f:
            self.tag_map = json.load(f)
        
        self.load_model(path)
    
    def load_model(self, path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = build_caformer('caformer_m36_384',
                num_classes=self.TAG_COUNT,
                decoder_embedding=384,
                num_head_decoder=8,
                num_layers_decoder=4,
                num_queries=80,
                scale_skip=1
            ).to(self.device)
        
        state = torch.load(path, map_location='cpu')
        self.model.load_state_dict(state, strict=True)
    
    def build_transform(self, image_size, keep_ratio=False):
        if keep_ratio:
            trans = transforms.Compose([
                transforms.Resize(image_size),
                crop_fix,
                transforms.ToTensor(),
            ])
        else:
            trans = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])
        return trans
    
    def infer_(self, img: Image.Image, thr: float):
        img = self.trans(img.convert('RGB')).to(self.device)
        with torch.cuda.amp.autocast():
            img = img.unsqueeze(0)
            output = torch.sigmoid(self.model(img)).cpu().view(-1)
        pred = torch.where(output > thr)[0].numpy()
        
        pre_result_class_list = [(self.tag_map[str(i)], output[i]) for i in pred]
        return pre_result_class_list
    
    @torch.no_grad()
    def infer_one(self, img: Image.Image, threshold: float, image_size: int, keep_ratio: bool):
        self.trans = self.build_transform(image_size, keep_ratio)
        pre_result_class_list = self.infer_(img, threshold)
        pre_result_class_list.sort(reverse=True, key=lambda x: x[1])
        pre_score_result = {cls: float(score) for cls, score in pre_result_class_list}
        return pre_score_result


if __name__ == '__main__':
    import sys
    import time
    sys.stderr.write('loading the model...\n')
    model = MlDanbooru(sys.argv[1])
    
    try:
        while True:
            sys.stderr.write('> ')
            sys.stderr.flush()
            
            path = input()
            img = Image.open(path)
            tags = model.infer_one(img, 0.7, 512, False)
            print(tags)
        
    except (KeyboardInterrupt, EOFError):
        pass
