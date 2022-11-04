# StrengthNet
Implementation of  "Accurate Emotion Strength Assessment for Seen and Unseen Speech Based on Data-Driven Deep Learning"

https://arxiv.org/abs/2206.07229


## Dependency
Ubuntu 18.04.5 LTS

- GPU: Quadro RTX 6000
- Driver version: 450.80.02
- CUDA version: 11.0

Python 3.5
- tensorflow-gpu 2.0.0b1 (cudnn=7.6.0)
- scipy
- pandas
- matplotlib
- librosa

### Environment set-up
For example,
```
conda create -n strengthnet python=3.5
conda activate strengthnet
pip install -r requirements.txt
conda install cudnn=7.6.0
```

## Usage
  
1. Run `python utils.py` to extract .wav to .h5;

2. Run `python train.py` to train a CNN-BLSTM based StrengthNet;
 
 
 

### Evaluating new samples

1. Put the waveforms you wish to evaluate in a folder. For example, `<path>/<to>/<samples>`

2. Run `python test.py --rootdir <path>/<to>/<samples>`

This script will evaluate all the `.wav` files in `<path>/<to>/<samples>`, and write the results to `<path>/<to>/<samples>/StrengthNet_result_raw.txt`. 

By default, the `output/strengthnet.h5` pretrained model is used.


## Citation

If you find this work useful in your research, please consider citing:
```
@inproceedings{liu22i_interspeech,
  author={Rui Liu and Berrak Sisman and Björn Schuller and Guanglai Gao and Haizhou Li},
  title={{Accurate Emotion Strength Assessment for Seen and Unseen Speech Based on Data-Driven Deep Learning}},
  year=2022,
  booktitle={Proc. Interspeech 2022},
  pages={5493--5497},
  doi={10.21437/Interspeech.2022-534}
}
```
 
  

## Resources

The [ESD corpus](https://github.com/HLTSingapore/Emotional-Speech-Data) is released by the [HLT lab](https://www.eng.nus.edu.sg/ece/hlt/), NUS, Singapore.<br>

The strength scores for the English samples of the ESD corpus are available [here](https://github.com/ttslr/StrengthNet/blob/main/Score_List.csv).

<br>


## Acknowledgements:

MOSNet: [https://github.com/lochenchou/MOSNet](https://github.com/lochenchou/MOSNet)

Relative Attributes: [Relative Attributes](https://github.com/chaitanya100100/Relative-Attributes-Zero-Shot-Learning)


## License

This work is released under MIT License (see LICENSE file for details).
