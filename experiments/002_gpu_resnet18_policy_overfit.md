# GPU resnet 18 overfit

This experiment explore the capacity of a resnet 18 to completely overfit a small batch of data.

### Task
Learn to predict future states and actions from past states and actions.
One timestep is the tuple (state, action).

Note:
- The game is settlers of catan
- The history length is 16
- The future length is 4
- Input are represented as a tensor of size: `[batch_size, seq * (C_s + C_a), H, W]`
- Outputs are represented as a tuple: `([batch_size, seq * C_s, H, W], [batch_size, seq, C_a])`

### Hypothesis
A resnet 18 can learn the mapping between past and future states for a small batch of data of the game settlers of catan for a fixed set of 4 robot players. The output is separated, taking into accounts the differents types (categorical, binary, float, etc.)

### Results
The results are much better with multiple heads for different predictions. So far I still have trouble to learn the regression ones. I need to explore this in details

Other notes:
I've explored the L2 regularisation. As it is known, the original adam implementation has trouble with L2 reg. This is why AdamW has been created. Used in conjunction with amsgrad, it allows me to regularize and keep the same learning curve. It also seems to accelerate the learning but does not solve the regression problems.

My idea on this problem, is to separate those heads. I wouldn't be surprised if mixing parameters head for regression and category is harmful for regression (happened to be wrong).

I've explored the outputs of the model. It is actually quite accurate (predicting target +-2 in general), the problem seems to be coming from the fact that the true values are very close to each other and so the network needs to predict them very accurately.
As an example, for the `devcardsleft` input, you have a segment of 0.04 to be correct which means that the squared loss must fall down under 4e-4 to ensure a good accuracy.

After more exploration, I finally find out the problem. The hint came from the training logs which were not consistent with predictions for the different regressions. The problem was coming from Batch norm.
First, as a general case, tt does not work well for small batchs and you need a certain amount of batches to approximate the statistics in a sufficient manner.
But even though, the training goes very well in the overfit setting, but Batchnorm completely fail for regression in evaluation mode. If I keep the batchnorm training setting then I indeed see that my model learns absolutely everything. The exact problem is coming from the faxct that the statistics is not computed the same way during training and evaluation. This adds noise when doing a prediction and this noise is sufficient to push the model off when trying to regress.

Conclusion: I'm going to switch norms. Since regression is a generative problem, I'll look into instance norm which has proven its capacity in the style transfer task.