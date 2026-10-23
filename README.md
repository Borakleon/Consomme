# Consommé: Conditioned Token Merging for Split Inference

This repository contains the implementation of Consommé.
Consommé is a token reduction framework for split inference based on conditioned token merging. 
Our proposed method outperforms existing token reduction methods by being able to reach low token counts while keeping a superior classification accuracy.

## Getting started

For our project, we used python 3.14.6. The required packages can be installed via
```
pip install -r requirements.txt
```

Download and extract ImageNet images. Consommé requires the images to be in the standard structure of:
```
│ILSVRC2012/
├──train/
│  ├── n01440764/
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── n01440764_10028.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764/
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ILSVRC2012_val_00003014.JPEG
│  │   ├── ......
│  ├── ......
```

The following models were used:
- vit_base_patch16_224.augreg2_in21k_ft_in1k
- deit_small_patch16_224
- deit_tiny_patch16_224

## Gate Training

To train the merging gate for Consommé run the following script:
```
python main.py --data_dir path/to/dataset/root --model_name model_backbone_name pretrained_path path/to/model_backbone --split_block_nr 3 --gate_top_k 8
```


If a run should be continued, run the following script:
```
python main.py --resume --model_path path/to/model/to/continue --data_dir path/to/dataset/root --model_name model_backbone_name pretrained_path path/to/model_backbone --split_block_nr 3 --gate_top_k 8
```
