# T-PGD
Code and data of the Findings of ACL 2023 [paper](https://arxiv.org/abs/2110.15317) "Bridge the Gap Between CV and NLP! A Gradient-based Textual Adversarial Attack Framework"

# How to run
Please check T-PGD/LaunchTPGD.ipynb to see about the details of hyperparameters and we have all the commands to run our main experiments there.

# Set up Metric
Before running the experiments, please download the USE-4 model from https://tfhub.dev/google/universal-sentence-encoder/4 and set the path variable in utils/Metric.py


# Requirements
The main packages we used in this project are listed below:
```
python==3.10.0
torch==1.13.1
transformers==4.29.0
tensorflow==2.12.0
tensorflow_hub==0.13.0
language-tool-python==2.7.1
```


# Citation
Please kindly cite our paper:

```
@inproceedings{yuan-etal-2023-bridge,
    title = "Bridge the Gap Between {CV} and {NLP}! A Gradient-based Textual Adversarial Attack Framework",
    author = "Yuan, Lifan  and
      Zhang, YiChi  and
      Chen, Yangyi  and
      Wei, Wei",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    year = "2023",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.446"
}
```


