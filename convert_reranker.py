from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.onnx import export
from transformers.onnx.features import FeaturesManager

MODEL_ID = "BAAI/bge-reranker-base"
SAVE_PATH = Path("./models/bge-reranker-onnx")
SAVE_PATH.mkdir(parents=True, exist_ok=True)

print(f"Loading model {MODEL_ID}...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)

# Determine the appropriate ONNX feature
feature = "sequence-classification"
model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(model, feature=feature)
onnx_config = model_onnx_config(model.config)

print("Exporting to ONNX...")
export(
    preprocessor=tokenizer,
    model=model,
    config=onnx_config,
    opset=14,  # Stable opset
    output=SAVE_PATH / "model.onnx",
)

# Save tokenizer alongside
tokenizer.save_pretrained(SAVE_PATH)
print(f"✅ ONNX model and tokenizer saved to {SAVE_PATH}")