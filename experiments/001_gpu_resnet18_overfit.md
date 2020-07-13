# GPU resnet 18 overfit

This experiement explore the capacity of a resnet 18 to completely overfit a small batch of data.

### Task
Learn to predict future states and actions from past states and actions.
One timestep is the tuple (state, action).

Note:
- The game is settlers of catan
- The history length is 16
- The future length is 4

### Hypothesis
A resnet 18 can learn the mapping between past and future states for a small batch of data of the game settlers of catan for a fixed set of 4 robot players.

### Results
The learning is indeed happening but the current network barely improve on chance for some states part. I believe this is due to the fact that the (state, action) tuple is very sparse and contains a lot of 0. The network will then achieve quite a high accuracy by predicting zeroz everywhere.
I'm going to change the output part, any information which is not spatial will be predicted directly by a scalar. It will make the output much more dense.
