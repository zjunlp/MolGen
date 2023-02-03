# MolGen

Code for the paper "[Molecular Language Model as Multi-task Generator](https://arxiv.org/pdf/2301.11259.pdf)".
- ❗NOTE: We provide a NLP for science paper-list at [https://github.com/zjunlp/NLP4Science_Papers](https://github.com/zjunlp/NLP4Science_Papers).
- ❗NOTE: We release our pre-trained model at [huggingface](https://huggingface.co/zjunlp/MolGen).

<div align=center><img src="molgen.png" width="80%" height="80%" /></div>

# Requirements

To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

# Resource Download
    
You can download the pre-trained model via this [link](https://drive.google.com/file/d/1ViU5BEgdkKmZ0mlVVQMrHFCS6LwhT79B/view?usp=sharing).

Moreover, the dataset used for downstream tasks can be found [here](https://github.com/zjunlp/MolGen/tree/main/moldata/finetune).

The expected structure of files is:

```
moldata
├── checkpoint 
│   ├── molgen.pkl          # pre-trained model
├── finetune
│   ├── np_test.csv         # nature product test data
│   ├── np_train.csv        # nature product train data
│   ├── plogp_test.csv      # synthetic test data for plogp optimization
│   ├── qed_test.csv        # synthetic test data for plogp optimization
│   └── zinc250k.csv        # synthetic train data
├── generate                # generate molecules
├── output                  # molecule candidates
└── vocab_list
    └── zinc.npy            # SELFIES alphabet
``` 

# How to run


+ ## Fine-tune

    - First, preprocess the finetuning dataset by generating candidate molecules using our pre-trained model. The preprocessed data will be stored in the folder ``output``.

    ```shell
        cd MolGen
        bash preprocess.sh
    ```

    - Then do multi-task prefix tuning in combine with the self-feedback paradigm. The fine-tuned model will be stored in the folder ``checkpoint``.


    ```shell
        bash finetune.sh
    ```

+ ## Generate

    To generate molecules, run this script. Please specify the ``checkpoint_path`` to determine whether to use the pre-trained model or the fine-tuned model.

    ```shell
    cd MolGen
    bash generate.sh
    ```
    
# Citation

If you use or extend our work, please cite the paper as follows:

```bibtex
@article{fang2023molecular,
  title={Molecular Language Model as Multi-task Generator},
  author={Fang, Yin and Zhang, Ningyu and Chen, Zhuo and Fan, Xiaohui and Chen, Huajun},
  journal={arXiv preprint arXiv:2301.11259},
  year={2023}
}
```
    
