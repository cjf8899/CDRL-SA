# CDRL-SA

## Update Note
* (24.08.19) The paper has been accepted into Remote Sensing Letters(RSL).

> [**Unsupervised Change Detection Based on Image Reconstruction Loss with Segment Anything**](https://doi.org/10.1080/2150704X.2024.2388851)
> 
> [Hyeoncheol Noh](https://scholar.google.co.kr/citations?user=XTmafQgAAAAJ&hl), [Jingi Ju](https://scholar.google.co.kr/citations?user=hlJYrqAAAAAJ&hl), [Yuhyun Kim](), [Minwoo Kim](https://scholar.google.com/citations?user=c7el4JwAAAAJ), [Dong-Geol Choi](https://scholar.google.co.kr/citations?user=1498JWoAAAAJ&hl)
> 
> *[Remote Sensing Letters](https://doi.org/10.1080/2150704X.2024.2388851)*
## Getting Started

Dataset download link : 
* [LEVIR-CD](https://drive.google.com/file/d/18RGfTqPo1atw_IMm6xPOnND-Vl4ok_o3/view?usp=sharing) (Our multi-class LEVIR dataset is included in that link.)
* [LEVIR-CD_A2B_B2A](https://drive.google.com/file/d/1-LERpM7GOxviKna47bbO_mLQON3Q0YcA/view?usp=sharing)
* [CLCD-CD](https://drive.google.com/file/d/1F4RfWSvoghmIrir_2YlBYfgrJt-flzY8/view?usp=sharing)
* [CLCD-CD_A2B_B2A](https://drive.google.com/file/d/1Q9COBNxg7r5PhgNzY60GTugotbS8AzUg/view?usp=sharing)

SAM download link : 
* [SAM weight](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth)

```angular2html
CDRL-SA
    └──datasets
        ├── LEVIR-CD
            ├── val
            ├── test
            └── train
                ├── A
                ├── B
                └── label
        ├── LEVIR-CD_A2B_B2A
            └── train
                ├── A
                └── B
        ├── CLCD-CD
        └── CLCD-CD_A2B_B2A
    └──pretrain_weight
        └── sam_vit_h_4b8939.pth
        
```

## Train
```bash
python main.py --root_path ./datasets/ --dataset_name LEVIR-CD --save_name levir
```

## CDRL Difference Map Generate
```bash
python test.py --root_path ./datasets/ --dataset_name LEVIR-CD --save_name levir
```

## CDRL-SA Refine Map Generate
```bash
python test_sam.py --root_path ./datasets/ --dataset_name LEVIR-CD --save_name levir
```
