# ML-Danbooru barebones sample

This repo is just a very basic working sample for the latest ML-Danbooru models made by [7eu7d7](https://github.com/7eu7d7/ML-Danbooru).

Last tested with `ml_caformer_m36_fp16_dec-5-97527.ckpt`.

## Links
- Original repository: https://github.com/7eu7d7/ML-Danbooru
- Models: https://github.com/7eu7d7/ML-Danbooru

## Usage
```
$ python main.py ml_caformer_m36_fp32_dec-5-97527.ckpt
loading the model...
> /data/downloads/sample.jpg
{'1girl': 0.9999992847442627, ... 'heart_earrings': 0.701194703578949}
>
```
