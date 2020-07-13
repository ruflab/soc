# GPU resnet 18

This experiement explore the capacity of a resnet 18 to learn to predict future state and actions.

### Task
Learn to predict future states and actions from past states and actions.
One timestep is the tuple (state, action).

Note:
- The game is settlers of catan
- The history length is 16
- The future length is 4

### Hypothesis
A resnet 18 can learn to predict the next 4 states and actions of the game settlers of catan for a fixed set of robot player up to 95% accuracy
