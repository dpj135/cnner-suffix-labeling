# Chinese Nested Named Entity Recognition Based on Suffix Labeling
Our code mainly refers to [W2NER](https://github.com/ljynlp/W2NER).\
We optimize the span-based approach using the suffix labeling method,aiming to improve the performance of Chinese NER.\
\
Paper:https://ieeexplore.ieee.org/abstract/document/10800298
## Environments
- python 3.8.19
- pytorch 2.4.0+cu118
## Dependencies
- numpy (1.24.1)
- transformers (4.44.2)
- scikit-learn (1.3.2)
- prettytable (3.11.0)
## Preparation
Download dataset\
Process them to fit the same format as the example in **data/** \
Put the processed data into the directory **data/** \
Here is an example of the processed data
```json
{"sentence": ["中", "国", "科", "学", "院", "大", "学"], "ner": [{"index": [0,1,2,3,4,5,6], "type": "organization"} ] }
```
We provide the code *process_CMeEE.py* for processing the original CMeEE dataset

Create configuration files(*.json) in **configs/** \
We have provided configuration files about Weibo, Resume, OntoNotes4 and CMeEE-V2


## Training
```python
python main.py --config ./configs/example.json
```
The experiment records are stored in the directory **train_logs/**

## Citation
If you use this codebase, or otherwise find our work valuable, please cite our paper:
```tex
@inproceedings{diao2024chinese,
  title={Chinese Nested Named Entity Recognition Based on Suffix Labeling},
  author={Diao, Pengjie and Zhang, Wanfeng and Li, Shengyang and Liu, Yunfei and Tang, Decheng},
  booktitle={2024 7th International Conference on Machine Learning and Natural Language Processing (MLNLP)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```
