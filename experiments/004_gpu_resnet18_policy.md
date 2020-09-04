# GPU resnet 18

This experiement explore the capacity of a resnet 18 to learn to predict future state and actions.

### Task
Learn to predict future states and actions from past states and actions.
One timestep is the tuple (state, action).

Note:
- The game is settlers of catan
- The history length is 16
- The future length is 4 or 1

### Hypothesis
A resnet 18 can learn to predict the next 4 states and actions of the game settlers of catan for a fixed set of robot player up to 95% accuracy.

### Results
After training the bot on a 50, 150 and 300 games training set. The learning curve follows the known DL behaviors: more data, better generalisation.
We go from 55% to 68% accuracy on the validation set which hints for increasing the training set size.
This would be pretty cheap to build a 10000 training set and I wouldn't be surprised if we can reach a validation of 90%.

One thing to note with the current data. 3 of the bots are instances of a RandomForest bot and only one is a Stac agent from StacSettlers.
I need to improve the training set data to ease the analysis.

One last though on my overall research. My goal is to study language grouding and not predictive capacity in itself. The prediction task is just a way to let emerge interesting representation in the network. This is why I'm going to shift my focus on improving the training set data and insert chat negotiations in the near future.
