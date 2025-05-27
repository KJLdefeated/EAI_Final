from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import GPTQModifier

MODEL_ID = "meta-llama/Llama-3.2-3B-Instruct"
NUM_CALIBRATION_SAMPLES=512
MAX_SEQUENCE_LENGTH=2048

# Load model.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, device_map="auto", torch_dtype="auto"
)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# Configure the quantization algorithm to run.
recipe = GPTQModifier(targets="Linear", scheme="W4A16", ignore=["lm_head"])

# Apply quantization.
oneshot(
    model=model, dataset=load_dataset(path="wikitext", name="wikitext-2-raw-v1", split=f"train[:{NUM_CALIBRATION_SAMPLES}]"),
    recipe=recipe,
    max_seq_length=MAX_SEQUENCE_LENGTH,
    num_calibration_samples=NUM_CALIBRATION_SAMPLES,
)

# Save to disk compressed.
SAVE_DIR = MODEL_ID.split("/")[1] + "-W4A16-G128"
model.save_pretrained(SAVE_DIR, save_compressed=True)
tokenizer.save_pretrained(SAVE_DIR)