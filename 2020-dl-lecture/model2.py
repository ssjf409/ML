from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.layers import Dropout


class ANN_seq_class(models.Sequential):
    def __init__(self, Nin, Nh, Nout):
        super().__init__()
        optm = optimizers.Adam(lr=0.001)

        self.add(layers.Dense(Nh/3, kernel_initalizer='he_normal', activation='relu', input_shape=(Nin,)))
        self.add(Dropout(0.25))
        self.add(layers.Dense(Nh/3*2, kernel_initalizer="he_normal", activation="relu"))
        self.add(Dropout(0.25))
        self.add(layers.Dense(Nh/3, kernel_initalizer="he_normal", activation="relu"))
        self.add(Dropout(0.25))
        self.add(layers.Dense(Nh/3*2, kernel_initalizer="he_normal", activation="relu"))
        self.add(Dropout(0.25))
        self.add(layers.Dense(Nh/3, kernel_initalizer="he_normal", activation="relu"))
        self.add(Dropout(0.25))
        self.add(layers.Dense(Nout, activation="softmax"))
        self.compile(loss="categorial_crossentropy", optimizers=optm, metrics = ["accuracy"])
