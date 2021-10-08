# StrengthNet
Implementation of  "StrengthNet: Deep Learning-based Emotion Strength Assessment for Emotional Speech Synthesis"

https://arxiv.org/abs/2110.03156


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
@misc{liu2021strengthnet,
      title={StrengthNet: Deep Learning-based Emotion Strength Assessment for Emotional Speech Synthesis}, 
      author={Rui Liu and Berrak Sisman and Haizhou Li},
      year={2021},
      eprint={2110.03156},
      archivePrefix={arXiv},
      primaryClass={cs.SD}
}
```
 
  

## Resources

The [ESD corpus](https://github.com/HLTSingapore/Emotional-Speech-Data) is released by the [HLT lab](https://www.eng.nus.edu.sg/ece/hlt/), NUS, Singapore.<br>

The strength scores for the english samples of ESD corpus are available at [here](https://github.com/ttslr/StrengthNet/blob/main/Score_List.csv).

<br>


## Acknowledgements:

MOSNet: [https://github.com/lochenchou/MOSNet](https://github.com/lochenchou/MOSNet)

Relative Attributes: [Relative Attributes](https://github.com/chaitanya100100/Relative-Attributes-Zero-Shot-Learning)


## License

This work is released under MIT License (see LICENSE file for details).
