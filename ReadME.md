Use data folder with code

```
conda create -n xai_timeseries
conda activate xai_timeseries
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install vector-quantize-pytorch
pip install pandas
pip install matplotlib
pip install wandb
pip install scikit-learn
pip install scikit-plot
pip install tqdm
```
use vqvae.py for training and testing 
use --num_quantizers to change number of vq-vae quantizers (1 == vq-vae, greater than 2 == residual vq)

