# Hyperparameters
LATENT_DIM = 32
D_MODEL = 512
PROJ_SIZE = D_MODEL // 4
RATE = 0  # Specifically, the dropout rate
EPOCHS = 50
BATCH_SIZE = 64
RASTER_LOSS_WEIGHT = 15000.0

# Vectorization sub-model weights
VECTOR_LOSS_WEIGHT_COMMAND = 1.0  # Keep this at 1, normalize others against it
# VECTOR_RASTERIZATION_LOSS_WEIGHT = 0.01
VECTOR_LOSS_WEIGHT_COORD = 100.0
VECTOR_LOSS_WEIGHT_COORD_ABSOLUTE = (
    1  # Ratio of absolute coord loss to relative coord loss
)

SIGNED_AREA_WEIGHT = 0.05
ALIGNMENT_LOSS_WEIGHT = 0.2
CONTRASTIVE_LOSS_WEIGHT = 0.1

EOS_SOFTMAX_TEMPERATURE = 0.1
HUBER_DELTA = (
    3.0 / 512.0
)  # Loss computations are in normalized -1 to 1 space across a 512 pixel image.
LOSS_IMAGE_SIZE = 256  # Size to rasterize images to for raster loss calculation

# Learning rate schedule (per parameter group)
LR_OTHER_START = 1e-3
LR_OTHER_FINAL = 1e-5
LR_LSTM_START = 1e-3
LR_LSTM_FINAL = 1e-5
LR_OUTPUT_COMMAND_START = 1e-4
LR_OUTPUT_COMMAND_FINAL = 1e-5
LR_OUTPUT_COORDS_START = 1e-4
LR_OUTPUT_COORDS_FINAL = 1e-5

WARMUP_STEPS = 100

GEN_IMAGE_SIZE = (512, 512)
RASTER_IMG_SIZE = GEN_IMAGE_SIZE[0]
STYLE_IMAGE_SIZE = (168, 40)
MAX_COMMANDS = 50
LIMIT = 0  # Limit the number of fonts to process for testing

# Scheduled Sampling
# Start decaying the teacher forcing ratio at this epoch
SCHEDULED_SAMPLING_START_EPOCH = 0
# Fully decayed by this epoch
SCHEDULED_SAMPLING_END_EPOCH = 200
# The minimum teacher forcing ratio (1.0 = 100% teacher forcing)
SCHEDULED_SAMPLING_MIN_RATIO = 0.75


MAX_SEQUENCE_LENGTH = MAX_COMMANDS + 2  # +2 for SOS and EOS tokens

ALPHABET = ["a", "d", "h", "e", "s", "i", "o", "n", "t"]
# # While pre-training the vectorizer, shove as many glyphs in as we can

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ1234567890"

BASE_DIR = "/home/simon/others-repos/fonts/ofl"
# BASE_DIR = "/mnt/experiments/fonts/ofl"

NUM_GLYPHS = len(ALPHABET)
