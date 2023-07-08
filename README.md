# ML-Danbooru barebones sample

This repo is just a very basic working sample for the latest ML-Danbooru models.

Last tested with `ml_caformer_m36_fp16_dec-5-97527.ckpt`.

# Links
- Original repository: https://github.com/7eu7d7/ML-Danbooru
- Models: https://github.com/7eu7d7/ML-Danbooru

# Usage
```
$ python main.py ml_caformer_m36_fp32_dec-5-97527.ckpt
loading the model...
> /data/downloads/sample.jpg
{'1girl': 0.9999992847442627, 'solo': 0.9999544620513916, 'tongue': 0.9974948167800903, 'tongue_out': 0.996187150478363, 'twintails': 0.9765990376472473, 'brown_hair': 0.9741637706756592, 'skirt': 0.9704353213310242, 'shirt': 0.9655012488365173, 'brown_eyes': 0.9652729630470276, 'hair_ornament': 0.9603611826896667, 'heart': 0.9594945907592773, 'choker': 0.9406992197036743, 'jewelry': 0.9362187385559082, 'long_sleeves': 0.9307852387428284, 'earrings': 0.9209005832672119, 'hairclip': 0.9197644591331482, 'black_skirt': 0.9154150485992432, 'white_background': 0.9153125286102295, 'looking_at_viewer': 0.9150885939598083, 'breasts': 0.9149472713470459, 'bangs': 0.9128955602645874, 'pink_shirt': 0.9057062864303589, 'blush': 0.902079701423645, 'piercing': 0.891681969165802, 'eyebrows_visible_through_hair': 0.8860618472099304, 'akanbe': 0.8727487325668335, 'ear_piercing': 0.8644703030586243, 'collared_shirt': 0.8290213346481323, 'medium_breasts': 0.8283935785293579, 'frills': 0.810139000415802, 'black_choker': 0.8026968836784363, 'cropped_torso': 0.7972100973129272, ':p': 0.7921291589736938, 'upper_body': 0.7909840941429138, 'simple_background': 0.7692321538925171, 'scrunchie': 0.7589906454086304, 'hand_up': 0.7567225098609924, '!?': 0.7510464191436768, 'two_side_up': 0.7412595152854919, 'heart_ring': 0.7361526489257812, 'ribbon': 0.7353011965751648, 'suspenders': 0.7328245639801025, 'stud_earrings': 0.7312374711036682, 'black_neckwear': 0.7233155965805054, 'frilled_shirt_collar': 0.7204340696334839, 'suspender_skirt': 0.7193049788475037, 'bow': 0.7142541408538818, 'heart_choker': 0.7042593359947205, 'hair_scrunchie': 0.7030725479125977, 'long_hair': 0.7023851275444031, 'heart_earrings': 0.701194703578949}
>
```