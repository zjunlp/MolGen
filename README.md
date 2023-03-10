# MolGen
[![Pytorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?e&logo=PyTorch&logoColor=white)](https://pytorch.org/)
![](https://img.shields.io/badge/version-1.0.1-blue)
[![license](https://img.shields.io/github/license/mashape/apistatus.svg?maxAge=2592000)](https://github.com/zjunlp/MolGen/blob/main/LICENSE)

Code for the paper "[Molecular Language Model as Multi-task Generator](https://arxiv.org/pdf/2301.11259.pdf)".
- βNOTE: We provide a NLP for science paper-list at [https://github.com/zjunlp/NLP4Science_Papers](https://github.com/zjunlp/NLP4Science_Papers).
- βNOTE: We release our pre-trained and fine-tuned model on π€ **Hugging Face** at [MolGen-large](https://huggingface.co/zjunlp/MolGen-large) and [MolGen-large-opt](https://huggingface.co/zjunlp/MolGen-large-opt).
- βNOTE: We provide a demo on π€ **Hugging Face** at [Space](https://huggingface.co/spaces/zjunlp/MolGen).

<div align=center><img src="molgen.png" width="80%" height="80%" /></div>

# π Requirements

To run the codes, you need to install the requirements:
```
pip install -r requirements.txt
```

# π Resource Download
    
You can download the pre-trained model via this [link1](https://drive.google.com/file/d/1ViU5BEgdkKmZ0mlVVQMrHFCS6LwhT79B/view?usp=sharing), and the fine-tuned models via this [link2](https://drive.google.com/drive/folders/1AFU28y6H9mbe4ALq9yD_LgtDZ1HjjnVT?usp=sharing).

Moreover, the dataset used for downstream tasks can be found [here](https://github.com/zjunlp/MolGen/tree/main/moldata/finetune).

The expected structure of files is:

```
moldata
βββ checkpoint 
βΒ Β  βββ molgen.pkl              # pre-trained model
β   βββ syn_qed_model.pkl       # fine-tuned model for QED optimization on synthetic data
β   βββ syn_plogp_model.pkl     # fine-tuned model for p-logP optimization on synthetic data
β   βββ np_qed_model.pkl        # fine-tuned model for QED optimization on natural product data
β   βββ np_plogp_model.pkl      # fine-tuned model for p-logP optimization on natural product data
βββ finetune
βΒ Β  βββ np_test.csv             # nature product test data
βΒ Β  βββ np_train.csv            # nature product train data
βΒ Β  βββ plogp_test.csv          # synthetic test data for plogp optimization
βΒ Β  βββ qed_test.csv            # synthetic test data for plogp optimization
βΒ Β  βββ zinc250k.csv            # synthetic train data
βββ generate                    # generate molecules
βββ output                      # molecule candidates
βββ vocab_list
    βββ zinc.npy                # SELFIES alphabet
``` 

# π How to run


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
    
