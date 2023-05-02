# CDRL-SA

## Update Note
* This repository will be completed on the date of the #### presentation.

### [Project Page](-) | [Video](-) | [Paper](-) | 

## Getting Started

Dataset download link : 
* [LEVIR-CD](https://drive.google.com/file/d/18RGfTqPo1atw_IMm6xPOnND-Vl4ok_o3/view?usp=sharing)
* [LEVIR-CD_A2B_B2A](https://drive.google.com/file/d/1-LERpM7GOxviKna47bbO_mLQON3Q0YcA/view?usp=sharing)
* [CLCD-CD](https://drive.google.com/file/d/1F4RfWSvoghmIrir_2YlBYfgrJt-flzY8/view?usp=sharing)
* [CLCD-CD_A2B_B2A](https://drive.google.com/file/d/1Q9COBNxg7r5PhgNzY60GTugotbS8AzUg/view?usp=sharing)

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
```

## Train
```bash
python main.py
```

## Eval
```bash
python test.py
```

