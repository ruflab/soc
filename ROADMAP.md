# Roadmap

- Refactor the outputs for predictive coding task so we avoid the very sparse targets and predict an actual policy for actions
- Change the JAVA so we make sure all games finished by the action WIN
- Implement history_length/future_length for sequential models and handle the roll_dice to avoid penalizing the model trajectories due to randomness
- Change the JAVA to have an column counting players full turn
- Work on the JAVA repository to include and align soclog files. The goal is to end up with a well-formated DB schema.
- Add a dataset to handle this new training data.