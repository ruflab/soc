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

My idea on this problem, is to separate those heads. I wouldn't be surprised if mixing parameters head for regression and category is harmful for regression.

I've explored the outputs of the model. It is actually quite accurate, the problem seems to be coming from the fact that the true values are very close to each other and so the network needs to predict them very accurately.
For the maps, this mean, it must compensate all the inputs to stabilize quite perfectly its output.
For devcardsleft, you have a segment of 0.04 to be correct which means that the squared loss must fall down between 4e-4 to ensure a good accuracy.

All right, so I've been bitten by Batch norm. It does not work well for small batchs and you need a certain amount of batches to approximate the statistics in a sufficient manner. The training goes very well in the overfit setting, but Batchnorm completely fail for regression in evaluation mode. If I keep the batchnorm training setting then I indeed see that my model learns absolutely everything.
I'm going to switch this batchnorm for instance norm.