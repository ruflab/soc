# Roadmap

- Check model loading
- Do the following experience
  - pre-train the CNN for next-step prediction, fix it, use a pretrained fixed Bert for text encoding, learn only the fusion module and reuse the fixed heads.
  - pre-train the CNN for next-step prediction, fix it, use a pretrained fixed Bert for text encoding, learn only the fusion module and the new heads
  - pre-train the CNN for next-step prediction, use a pretrained fixed Bert for text encoding, learn the fusion module and the new heads + let the CNN on learning mode
  - use a pretrained fixed Bert for text encoding, learn directly the CNN, the fusion module and heads
- Analyse the work and start questioning the fusion module.
- Run all the pipeline on 10000 training games (20 epochs) + 100 validation game


### Other ideas
- Refactor the code to handle pytorch-lightning >= 0.9.1
- test SIREN networks
- Implement a HP tuning lib
- Check SWA update: https://github.com/PyTorchLightning/pytorch-lightning/issues/1894