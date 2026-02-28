import tensorflow as tf
import keras

class FocalLoss(keras.losses.Loss):
    """
    Focal Loss with integrated label smoothing for binary classification.
    
    Parameters
    ----------
    gamma : float
        Focusing parameter. Increases weight for hard examples.
    alpha : float
        Balance parameter for class imbalance (label = 1).
    label_smoothing : float
        Smoothing factor. 0 = no smoothing (standard focal loss).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: float = 0.6,
        label_smoothing: float = 0.05,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        # Apply label smoothing
        y_smooth = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = (
            -y_smooth * tf.math.log(y_pred)
            - (1.0 - y_smooth) * tf.math.log(1.0 - y_pred)
        )
        # Use original (hard) labels for the focusing factor
        p_t = tf.where(tf.cast(y_true, tf.bool), y_pred, 1.0 - y_pred)
        alpha_factor = tf.where(tf.cast(y_true, tf.bool), self.alpha, 1.0 - self.alpha)
        modulating_factor = tf.pow(1.0 - p_t, self.gamma)

        return tf.reduce_mean(alpha_factor * modulating_factor * bce)

    def get_config(self):
        config = super().get_config()
        config.update({
            "gamma": self.gamma,
            "alpha": self.alpha,
            "label_smoothing": self.label_smoothing,
        })
        return config

@keras.saving.register_keras_serializable()
class DiversityFocalLoss(keras.losses.Loss):
    """
    Focal Loss + pénalité de diversité renforcée (diversity_weight=0.4).
    Empêche le modèle de toujours prédire DOWN.
    """
    def __init__(self, gamma=2.0, alpha=0.65, label_smoothing=0.08,
                 diversity_weight=0.4, **kwargs):
        super().__init__(**kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing
        self.diversity_weight = diversity_weight

    def call(self, y_true, y_pred):
        y_true = tf.cast(y_true, tf.float32)
        y_smooth = y_true * (1.0 - self.label_smoothing) + 0.5 * self.label_smoothing
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        bce = -y_smooth * tf.math.log(y_pred) - (1.0 - y_smooth) * tf.math.log(1.0 - y_pred)
        p_t = tf.where(tf.cast(y_true, tf.bool), y_pred, 1.0 - y_pred)
        alpha_t = tf.where(
            tf.cast(y_true, tf.bool),
            tf.ones_like(y_true) * self.alpha,
            tf.ones_like(y_true) * (1.0 - self.alpha),
        )
        focal_loss = tf.reduce_mean(alpha_t * tf.pow(1.0 - p_t, self.gamma) * bce)

        # Pénalité de diversité renforcée
        mean_pred = tf.reduce_mean(y_pred)
        diversity_penalty = tf.square(mean_pred - 0.5) * self.diversity_weight

        return focal_loss + diversity_penalty

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'gamma': self.gamma, 'alpha': self.alpha,
            'label_smoothing': self.label_smoothing,
            'diversity_weight': self.diversity_weight
        })
        return cfg
