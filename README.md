
# Dynamic Residual Classifier for Class Incremental Learning

This repo contains the official code of the project "Dynamic Residual Classifier for Class Incremental Learning" (ICCV2023).
Paper link.

## 1.Dependent Packages and Platform

First we recommend to create a conda environment with all the required packages by using the following command.

```
conda env create -f environment.yml
```

This command creates a conda environment named `MAFDRC`. You can activate the conda environment with the following command:

```
conda activate MAFDRC
```

In the following sections, we assume that you use this conda environment or you manually install the required packages.

Note that you may need to adapt the `environment.yml/requirements.txt` files to your infrastructure. The configuration of these files was tested on Linux Platform with a GPU (RTX3080 Ti).

If you see the following error, you may need to install a PyTorch package compatible with your infrastructure.

```
RuntimeError: No HIP GPUs are available or ImportError: libtinfo.so.5: cannot open shared object file: No such file or directory
```

For example if your infrastructure only supports CUDA == 11.1, you may need to install the PyTorch package using CUDA11.1.

```
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## 2.Dataset

We have implemented the pre-processing of `CIFAR100`, `ImageNet100`, `ImageNet1000`. When training on `CIFAR100`, this framework will automatically download it. When training on `ImageNet100/ImageNet1000`, you should specify the folder of your dataset in `utils/data.py`.

```python
    def download_data(self):
        assert 0,"You should specify the folder of your dataset"
        train_dir = '[DATA-PATH]/train/'
        test_dir = '[DATA-PATH]/val/'
```

## 2.Run

`To run the testing code:`

Due to the limitation of the supplementary material size, we only upload the results of CIFAR100.

- Test CIFAR100

    ```
    python main.py --config=mafdrc-cifar100.json --test True
    ```

`To run the training code:`

- Train CIFAR100

    ```
    python main.py --config=mafdrc-cifar100.json
    ```

- Train ImageNet100

    1.You first edit the `[mafdrc-imagenet100].json` file for global settings.

    2.Edit the hyperparameters and model name in the corresponding `[mafdrc].py` file (e.g., `models/mafdrc.py`).

    ```
    python main.py --config=mafdrc-imagenet100.json
    ```

- Train ImageNet1000

    1.You first edit the `[mafdrc-imagenet100].json` file for global settings.

    2.Edit the hyperparameters and model name in the corresponding `[mafdrc].py` file (e.g., `models/mafdrc.py`).

    ```
    python main.py --config=mafdrc-imagenet1000.json 
    ```        

## 3.Results

`CIFAR100:`

Tasks | 10 | 20 | 30 | 40 | 50 | 60 | 70 | 80 | 90 | 100 | Avg |
:---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
Ours (MAFDRC) | **93.4** | **83.0** | **80.9** | **76.72** | **74.18** | **71.3** | **69.17** | **65.45** | **63.49** | **62.04** | **73.97**

Note:

If you use other versions of pytorch, it will get a different result.

## 4.Citation

If you find this code useful, please kindly cite the following paper:

```
@article{,
  title={Dynamic Residual Classifier for Class Incremental Learning},
  author={Xiuwei Chen, Xiaobin Chang},
  booktitle = {IEEE International Conference on Computer Vision (ICCV)},
  year={2023}
}
```
