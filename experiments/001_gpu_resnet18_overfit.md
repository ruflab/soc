# GPU resnet 18 overfit

This experiment explore the capacity of a resnet 18 to completely overfit a small batch of data.

### Task
Learn to predict future states and actions from past states and actions.
One timestep is the tuple (state, action).

Note:
- The game is settlers of catan
- The history length is 16
- The future length is 4
- Input and output states are represented as a tensor of size `[batch_size, seq * (C_s + C_a), H, W]`

### Hypothesis
A resnet 18 can learn the mapping between past and future states for a small batch of data of the game settlers of catan for a fixed set of 4 robot players using a very generic MSE loss (no specialization based on the type of inputs).

### Results
Hypothesis false.

The learning is indeed happening but the current network barely improve on chance for some states part. I believe this is due to the fact that the (state, action) tuple is very sparse and contains a lot of 0. The network will then achieve quite a high accuracy by predicting zero everywhere for those parts.
Investigation shows that it is the case, the network learn to predict only zeros for the sparse parts. It also struggles to learn to copy the map from the input to the output.

To address those issues:
- For actions, I will replace the sparse tensor by usual policy which enables the use of the cross-entropy loss. It adds a policy head.
- Every non spatial data will be predicted as non spatial. This add a non spatial state head.
- We keep a head for spatial states
- Finally, for pieces which are very sparse, we will change the loss, I will look into sub-sampling the zeros so for each element, we actually have a loss on an equal amount of zeros and ones. Let's see.
