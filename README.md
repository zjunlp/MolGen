# MolGen
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjunlp/MolGen/blob/main/LICENSE)

Code for the paper "[Domain-Agnostic Molecular Generation with Self-feedback](https://arxiv.org/pdf/2301.11259.pdf)".

<div align=center><img src="molgen.png" width="100%" height="100%" /></div>


# 🔔News 

- **2023-6 We release [Mol-Instructions](https://github.com/zjunlp/Mol-Instructions) is a large-Scale biomolecules instruction dataset for large language models.**
- **2023-4 We provide a NLP for science paper-list at [https://github.com/zjunlp/NLP4Science_Papers](https://github.com/zjunlp/NLP4Science_Papers).**
- **2023-3 We release our pre-trained and fine-tuned model on 🤗 **Hugging Face** at [MolGen-large](https://huggingface.co/zjunlp/MolGen-large) and [MolGen-large-opt](https://huggingface.co/zjunlp/MolGen-large-opt).**
- **2023-2 We provide a demo on 🤗 **Hugging Face** at [Space](https://huggingface.co/spaces/zjunlp/MolGen).**



# 📕 Requirements

To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

# 📚 Resource Download
    
You can download the pre-trained model via this [link1](https://drive.google.com/file/d/1ViU5BEgdkKmZ0mlVVQMrHFCS6LwhT79B/view?usp=sharing), and the fine-tuned models via this [link2](https://drive.google.com/drive/folders/1AFU28y6H9mbe4ALq9yD_LgtDZ1HjjnVT?usp=sharing).

Moreover, the dataset used for downstream tasks can be found [here](https://github.com/zjunlp/MolGen/tree/main/moldata/finetune).

The expected structure of files is:

```
moldata
├── checkpoint 
│   ├── molgen.pkl              # pre-trained model
│   ├── syn_qed_model.pkl       # fine-tuned model for QED optimization on synthetic data
│   ├── syn_plogp_model.pkl     # fine-tuned model for p-logP optimization on synthetic data
│   ├── np_qed_model.pkl        # fine-tuned model for QED optimization on natural product data
│   ├── np_plogp_model.pkl      # fine-tuned model for p-logP optimization on natural product data
├── finetune
│   ├── np_test.csv             # nature product test data
│   ├── np_train.csv            # nature product train data
│   ├── plogp_test.csv          # synthetic test data for plogp optimization
│   ├── qed_test.csv            # synthetic test data for plogp optimization
│   └── zinc250k.csv            # synthetic train data
├── generate                    # generate molecules
├── output                      # molecule candidates
└── vocab_list
    └── zinc.npy                # SELFIES alphabet
``` 

# 🚀 How to run


+ ## Fine-tune

    - First, preprocess the finetuning dataset by generating candidate molecules using our pre-trained model. The preprocessed data will be stored in the folder ``output``.

    ```shell
        cd MolGen
        bash preprocess.sh
    ```

    - Then utilize the self-feedback paradigm. The fine-tuned model will be stored in the folder ``checkpoint``.


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
  title={Domain-Agnostic Molecular Generation with Self-feedback},
  author={Fang, Yin and Zhang, Ningyu and Chen, Zhuo and Fan, Xiaohui and Chen, Huajun},
  journal={arXiv preprint arXiv:2301.11259},
  year={2023}
}
```
    
