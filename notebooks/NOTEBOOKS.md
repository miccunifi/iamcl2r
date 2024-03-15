# Notebooks for $\text{IAM-CL}^2\text{R}$ training and evaluation

We provide both Google Colab and IPython Notebooks. They contain the same code.

## Google Colab links
- [$\text{IAM-CL}^2\text{R}$ with $d$-Simplex-HOC](https://colab.research.google.com/drive/11GuNPe3DSvS0qZzCRCuC1nUs7ejOuTvQ?usp=sharing)
- [$\text{IAM-CL}^2\text{R}$ with ER baseline](https://colab.research.google.com/drive/1zzzLP_rSYDrVn8U_EXmhaIJ60MIKZRY0?usp=sharing)
- [Compatibility evaluation script](https://colab.research.google.com/drive/1bDHXF_i1kWRO20LqDbxxaNY5D-zsdZ_k?usp=sharing)

## IPython Notebooks

### Create virtual env and install dependencies
The following command will create a virtual enviroments and automatically install the required dependencies.

```bash
make install
```
Remember to select this virtual env as notebook kernel.

### Download pretrained-models
Before executing any training script please use the following command to download the checkpoints of the pretrained models that are used to replace the fine-tuned model.

```bash
make download-pretrained-models
```

### Implementations
These are the currently available implementations:
- [ER baseline](./iamcl2r_ce.ipynb) 
- [d-Simplex-HOC](./iamcl2r_hoc.ipynb) 

Using this code we got the following results:

|   | Model | 7 Tasks | 31 Tasks |
|---|-------|---------|----------|
|   | ER baseline | $\mathit{AC}: 0.00 \, \mathit{AA}: 35.18$ | $\mathit{AC}: 0.00 \, \mathit{AA}: 30.32$|
|   | d-Simplex-HOC | $\mathit{AC}: 0.95 \, \mathit{AA}: 68.90$ | $\mathit{AC}: 0.65 \, \mathit{AA}: 68.07 $ |

#### Download results
Use the following link to directly download the checkpoints that we got from the notebooks.

```bash
make download-results
```
