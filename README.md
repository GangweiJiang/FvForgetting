## Function Vector Guided Training
This is the code for our paper "Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning".

### Setup
You can install the required libraries by running the following command:
```
pip install -r requirements.txt
```
You are also required to download the language model and put it to the folder. 

### Data preparation
You can download the data from the following link [Google Drive](https://drive.google.com/file/d/1fu3ea4xmf9_-boLzLT23Z4qTSirHUq8y/view?usp=sharing).

Then put the data to the folder ``data/``.

### Training and Evaluation
You need first to extract the function vectors from the language model. Then you can train the model with the function vectors. Finally, you can evaluate the model.

**Function vector extraction**
This is the step to extract the function vectors for training tasks. You can run the following command:
```
bash scripts/fv/uni_fv.sh seq0
bash scripts/fv/uni_fv.sh seq1
bash scripts/fv/uni_fv.sh seq2
```
**Function vector guided training**
Then you can train the model with the function vectors regularization term. You can run the following command:
```
bash scripts/train/seq0_fvg.sh {seed}
bash scripts/train/seq1_fvg.sh {seed}
bash scripts/train/seq2_fvg.sh {seed}
```
**Model evaluation**
After training the model, you can evaluate the model with the following command:
```
bash scripts/eval/general_score.sh {exp_name}
bash scripts/eval/icl_score.sh {exp_name}
bash scripts/eval/seq0_score.sh {exp_name}
bash scripts/eval/seq1_score.sh {exp_name}
bash scripts/eval/seq2_score.sh {exp_name}
```

### Citation
If you find this code useful, please cite our paper:
```
@inproceedings{jiangunlocking,
  title={Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning},
  author={Jiang, Gangwei and JIANG, Caigao and Li, Zhaoyi and Xue, Siqiao and ZHOU, JUN and Song, Linqi and Lian, Defu and Wei, Ying},
  booktitle={The Thirteenth International Conference on Learning Representations}
}
```

## Acknowledgments

This repo benefits from [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), [Function vector](https://github.com/ericwtodd/function_vectors), and [baukit](https://github.com/davidbau/baukit). Thanks for their wonderful works.