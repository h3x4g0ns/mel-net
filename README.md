# mel-net

We know that Neural Networks can do a pretty good job learning latent information about images. The idea is to use mel spectograms with neural networks to learn this latent information and perform some pretty basic audio classification benchmarks. Also I wanted an excude to use [wandb.ai]. 

## different vanilla architectures

For the sake of consistency, I'll be running tests with the [spoken_digit](https://www.tensorflow.org/datasets/catalog/spoken_digit) dataset which is basically MNIST but with audio. 

- [ ] vanilla CNN
- [ ] auto encoder into MLP
- [ ] modified ResNet

For a more complicated dataset, I will use RNN cells + ResNet to predict words in a sequence with the [crema_d](https://www.tensorflow.org/datasets/catalog/crema_d) audo-visual dataset. This dataset consists of facial and vocal emotional expression in sentences spoken in a range of basic emotion states.


## relevant papers

- [audio recognition + cnn](http://noiselab.ucsd.edu/ECE228_2019/Reports/Report38.pdf)