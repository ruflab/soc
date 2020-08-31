# Roadmap

- Make a first multimodal pipeline with hopfield networks
- Refactor inputs
  - Separate players info into categorical distribution
  - change devcardsleft from regression to categories
- Build a function to compute accuracy per action types
- Do the following experience
  - pre-train the CNN for next-step prediction, fix it, use a pretrained fixed Bert for text encoding, learn only the fusion module and reuse the fixed heads.
  - pre-train the CNN for next-step prediction, fix it, use a pretrained fixed Bert for text encoding, learn only the fusion module and the new heads
  - pre-train the CNN for next-step prediction, use a pretrained fixed Bert for text encoding, learn the fusion module and the new heads + let the CNN on learning mode
  - use a pretrained fixed Bert for text encoding, learn directly the CNN, the fusion module and heads
- JAVA work
    - Improve the code to add missing or buggy data in the DB if any
    - Add a column counting players full turn
    - Store chat negotiations in the DB
    - Store server game descriptions in the DB
- Debug work
- Run all the pipeline on 10000 training games (20 epochs) + 100 validation game
- Analyse the work and start questioning the fusion module.

### Other ideas
- Refactor the code to handl pytorch-lightning >= 0.9.1
- test SIREN networks
- Implement a HP tuning lib
- Implement history_length/future_length for sequential models
- Check SWA update: https://github.com/PyTorchLightning/pytorch-lightning/issues/1894