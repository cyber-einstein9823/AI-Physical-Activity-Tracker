# Human Activity Recognition — Experimental Analysis

## 1. Experiment Overview

This experiment evaluates a deep learning model for Human Activity Recognition (HAR) using raw smartphone inertial sensor signals from the UCI Human Activity Recognition Using Smartphones dataset.

Each sample contains 128 time steps and 9 sensor channels, representing a 2.56-second window sampled at 50 Hz.

The system classifies six activities:
- WALKING
- WALKING_UPSTAIRS
- WALKING_DOWNSTAIRS
- SITTING
- STANDING
- LAYING

## 2. Model

HARNet uses a Bidirectional LSTM to learn temporal patterns from sensor sequences. Temporal attention pooling assigns varying importance to different time steps before a linear classifier predicts one of six activity classes.

The model contains 46,982 trainable parameters.

## 3. Training

The model was trained for 30 epochs using Adam with a learning rate of 0.001 and batch size of 64. Gradient clipping was applied with a maximum norm of 1.0.

The dataset was divided into training, validation, and test sets using subject-based separation. Standardization statistics were calculated using training data only.

The best checkpoint was selected using validation loss.

- Best validation loss: 0.0765
- Best validation loss epoch: 18
- Peak validation accuracy: approximately 98.08% at epoch 3

The validation accuracy peak and minimum validation loss occurred at different epochs.

## 4. Test Performance

The final evaluation was performed on 2,947 test windows from subjects excluded from training and validation.

Overall test accuracy:

**93.45%**

### Per-Class Accuracy

| Activity | Correct | Total | Accuracy |
|---|---:|---:|---:|
| WALKING | 484 | 496 | 97.58% |
| WALKING_UPSTAIRS | 469 | 471 | 99.58% |
| WALKING_DOWNSTAIRS | 415 | 420 | 98.81% |
| SITTING | 402 | 491 | 81.87% |
| STANDING | 448 | 532 | 84.21% |
| LAYING | 536 | 537 | 99.81% |

## 5. Confusion Matrix Analysis

The model correctly recognizes most walking activities and laying.

The primary confusion occurs between SITTING and STANDING:

- 87 SITTING samples were predicted as STANDING.
- 84 STANDING samples were predicted as SITTING.

This confusion is understandable because both activities produce relatively stationary sensor patterns. The differences between them can be subtle and may depend on body posture, phone orientation, and sensor placement.

Walking-related activities are comparatively easier to distinguish because they contain more characteristic movement patterns.

## 6. Training Behavior

Training loss decreased rapidly during the initial epochs and subsequently stabilized.

Validation loss reached its lowest value at epoch 18. Later epochs showed increasing validation loss while training loss remained comparatively low. This behavior suggests overfitting during later training.

Selecting the checkpoint based on validation loss helps avoid using a model from a later epoch that performs worse on validation data.

## 7. Attention Visualization

Temporal attention provides a representation of the relative importance assigned to time steps in a sensor window.

The attention visualizations allow inspection of how the model distributes attention across the sequence for different activity predictions.

Attention weights are useful for model inspection, but they should not automatically be interpreted as definitive causal explanations of a prediction.

## 8. Limitations

- Sitting and standing are difficult to distinguish using inertial signals alone.
- Performance may vary with smartphone placement and orientation.
- The model was evaluated on the UCI HAR dataset and may not generalize equally to all users or environments.
- The test results represent this dataset and experimental setup.
- This system is intended for educational use, not medical or safety-critical applications.

## 9. Conclusion

The experiment achieved 93.45% test accuracy on the UCI HAR test set using raw inertial signals and a Bidirectional LSTM with temporal attention.

The results demonstrate that recurrent deep learning models can learn meaningful temporal patterns for activity recognition. Strong performance was achieved for walking activities and laying, while sitting and standing remained the most challenging classes.

The experiment also demonstrates a complete machine learning workflow involving preprocessing, subject-based data splitting, model training, validation-based checkpoint selection, independent testing, performance analysis, visualization, and interactive inference.
