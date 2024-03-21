
# <a id="title"> Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements
</a>

This repo contains the source code of the Python package `iamcl2r` and it is the official implementation of: <br>
**Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements**, *Niccolò Biondi, Federico Pernici, Simone Ricci, Alberto Del Bimbo* at CVPR2024. 
Arxiv link: (*coming soon*)
<br>

The repo also contains several examples on [Google Colab](./notebooks/NOTEBOOKS.md#google-colab-links) and [IPython Notebook](./notebooks/NOTEBOOKS.md#ipython-notebooks). <br> See [NOTEBOOKS.md](./notebooks/NOTEBOOKS.md) for more details.
We only support PyTorch for now.

See our paper for a detailed description of $d$-Simplex-HOC and the Improved Asynchronous Model Compatible Lifelong Learning Representation ($\text{IAM-CL}^2\text{R}$) scenario.

## Repository Overview

There are several directories in this repo:
 * [notebooks/](./notebooks/) contains the code for the [Google Colab](./notebooks/NOTEBOOKS.md#google-colab-links) and [IPython Notebook](./notebooks/NOTEBOOKS.md#ipython-notebooks) examples we provide;
 * [src/iamcl2r/](src/iamcl2r/) contains the source code for the package `iamcl2r`;
 * [configs/](./configs/) contains the configuration files for the examples;

## Quickstart


1. **Installation**
    Installing `iamcl2r` is simply
    ```bash
    git clone https://github.com/miccunifi/iamcl2r
    cd iamcl2r
    make install
    ```
    This creates a virtual enviroment and installs all the required dependencies.

    <details>
      <summary>Create the .env file (Optional but suggested)</summary>

    Create an `.env` file to automatically export your env vars before launching training.
    An example of an `.env` file is
    ```.env
    WANDB_USERNAME=<your wandb username>
    WANDB_API_KEY=<your wandb private api key> 
    WANDB_ENTITY=<wandb entity>
    WANDB_PROJECT=<wandb project name>
    ```
    The `.env` file is not released for security reasons.

    </details>
  

<br>

2. **Download the pretrained models**
    These models are used to replace the fine-tuned model in a $\text{IAM-CL}^2\text{R}$ training.
    ```bash
    make download-pretrained-models
    ```

<br>

3. **Run  $\text{IAM-CL}^2\text{R}$ training** 
    We only support $d$-Simplex-HOC and ER baseline for now.
    ```bash
    # d-Simplex-HOC training
    make run CONFIG=configs/hoc_7tasks.yaml
    make run CONFIG=configs/hoc_31tasks.yaml

    # ER baseline training
    make run CONFIG=configs/er_7tasks.yaml
    make run CONFIG=configs/er_31tasks.yaml
    ```


4. **Run a compatibility evaluation**
    Customize the configuration file [eval.yaml](./configs/eval.yaml) by changing the `checkpoint_path` line
    ```yaml
    ...
    checkpoint_path: <FOLDER_PATH>
    ...
    ```
    After that, you can evaluate the compatibility of the models which checkpoints are in the `<FOLDER_PATH>` you have specified above. 

    ```bash
    make run CONFIG=configs/eval.yaml
    ```

<p align="right">(<a href="#title">back to top</a>)</p>

#### Released results
We have released some experimental results obtained using the code in the repo.
See [NOTEBOOKS.md](./notebooks/NOTEBOOKS.md#download-results) for more details.


<p align="right">(<a href="#title">back to top</a>)</p>

## Citation
If you use this code in your research, please kindly cite the following paper:

```BibTeX
@inproceedings{biondi2024stationary,
  title={Stationary Representations: Optimally Approximating Compatibility and Implications for Improved Model Replacements},
  author={Biondi, Niccolò and Pernici, Federico and Ricci, Simone and Del Bimbo, Alberto},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year={2024},
}
```

<p align="right">(<a href="#title">back to top</a>)</p>

## Contact

For help, new features, or reporting bugs associated with this repo, please open a [GitHub issue](https://github.com/miccunifi/iamcl2r/issues) or contact us if you have any questions.


- Niccolò Biondi <niccolo.biondi (at) unifi.it>[![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/NiccoBio.svg?style=social&label=NiccoBio)](https://twitter.com/NiccoBio) [![Google Scholar Citations](https://img.shields.io/badge/Google%20Scholar-Niccolò%20Biondi-blue.svg)](https://scholar.google.com/citations?hl=en&user=B7VHm9UAAAAJ)
- Federico Pernici <federico.pernici (at) unifi.it> [![Twitter URL](https://img.shields.io/twitter/url/https/twitter.com/FedPernici.svg?style=social&label=FedPernici)](https://twitter.com/FedPernici) [![Google Scholar Citations](https://img.shields.io/badge/Google%20Scholar-Federico%20Pernici-blue.svg)](https://scholar.google.com/citations?user=I8nFKUsAAAAJ&hl=en)
- Simone Ricci <simone.ricci (at) unifi.it>[![Google Scholar Citations](https://img.shields.io/badge/Google%20Scholar-Simone%20Ricci-blue.svg)](https://scholar.google.com/citations?user=jtj_lhAAAAAJ&hl=en)
- Alberto Del Bimbo <alberto.delbimbo (at) unifi.it>[![Google Scholar Citations](https://img.shields.io/badge/Google%20Scholar-Alberto%20Del%20Bimbo-blue.svg)](https://scholar.google.com/citations?hl=en&user=bf2ZrFcAAAAJ)

<p align="right">(<a href="#title">back to top</a>)</p>



