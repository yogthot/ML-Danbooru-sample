import torch
from torch import nn
from typing import Optional, List, Union
from einops import repeat, rearrange

from timm.models import create_model

from .metaformer_baselines import MetaFormer
from .position_encoding import build_position_encoding
from .ms_decoder import MSDecoder, MSDecoderLayer


class ML_MetaFormer(nn.Module):
    def __init__(self, encoder: MetaFormer, decoder: MSDecoder, num_queries=50, d_model=512, num_classes=1000, scale_skip=0):
        super().__init__()
        self.encoder=encoder
        self.decoder=decoder
        self.cls_head = nn.Linear(d_model, num_classes)
        self.feats_trans=nn.ModuleList([nn.Sequential(
                                            nn.LayerNorm(dim),
                                            nn.Linear(dim, d_model),
                                            nn.LayerNorm(d_model),
                                        ) for dim in self.encoder.scale_dims[scale_skip:]])
        self.pos_encoder = build_position_encoding('sine', d_model)
        self.query_embed = nn.Embedding(num_queries, d_model)

        self.num_classes=num_classes
        self.num_queries=num_queries
        self.d_model=d_model
        self.scale_skip=scale_skip

    def encode(self, x):
        feat_list=[]
        for i in range(self.encoder.num_stage):
            x = self.encoder.downsample_layers[i](x)
            x = self.encoder.stages[i](x)
            if i>=self.scale_skip:
                feat_list.append(x)
        return feat_list

    def decode(self, feat_list):
        q = repeat(self.query_embed.weight, 'q c -> b q c', b=feat_list[0].shape[0])
        pos_emb = [rearrange(self.pos_encoder(x), 'b h w c -> b (h w) c') for x in feat_list]
        feat_list = [trans(rearrange(x, 'b h w c -> b (h w) c')) for x,trans in zip(feat_list, self.feats_trans)]

        out = self.decoder(q, feat_list, pos=pos_emb)
        return out

    def forward(self, x):
        feat_list = self.encode(x)
        pred = self.decode(feat_list) # [B, Nq_scale, L]

        pred = self.cls_head(pred)
        pred = (pred * torch.softmax(pred, dim=1)).sum(dim=1)
        return pred

def build_caformer(model_name, num_classes, decoder_embedding, num_head_decoder, num_layers_decoder, num_queries, scale_skip):
    create_model_args = dict(
        model_name=model_name,
        pretrained=False,
        num_classes=num_classes,
        drop_rate=0.0,
        drop_connect_rate=None,
        drop_path_rate=0.0,
        drop_block_rate=None,
        global_pool=None,
        bn_momentum=None,
        bn_eps=None,
        scriptable=False,
        checkpoint_path=None
    )

    encoder = create_model(**create_model_args)
    dec_layer = MSDecoderLayer(decoder_embedding,num_head_decoder)
    decoder = MSDecoder(dec_layer, num_layers_decoder, norm=nn.LayerNorm(decoder_embedding))
    
    model = ML_MetaFormer(encoder, decoder, num_queries=num_queries, num_classes=num_classes,
                          d_model=decoder_embedding, scale_skip=scale_skip)

    return model
