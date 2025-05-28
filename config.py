from dataclasses import dataclass

@dataclass
class ModelConfig:
    IMG_SIZE: int = 224
    BATCH_SIZE: int = 32  # Augmenté de 16 à 32
    EPOCHS: int = 1
    SEED: int = 42
    LEARNING_RATE_MOBILENET: float = 1e-3
    LEARNING_RATE_RESNET: float = 1e-4
    DROPOUT_RATE: float = 0.5
    L2_LAMBDA: float = 0.01
    DATA_DIR: str = "data/chest_xray"