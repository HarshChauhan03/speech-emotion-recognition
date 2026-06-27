from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout

# Build Model

model = Sequential()

model.add(
    Dense(
        128,
        activation="relu",
        input_shape=(13,)
    )
)

model.add(
    Dropout(0.3)
)

model.add(
    Dense(
        64,
        activation="relu"
    )
)

# Example Emotion Classes:
# Happy
# Sad
# Angry
# Neutral

model.add(
    Dense(
        4,
        activation="softmax"
    )
)

model.summary()

print("Speech Emotion Recognition Model Ready")