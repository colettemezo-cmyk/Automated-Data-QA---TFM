"""ONNX embedding model knobs (shared encoder implementation)."""

MODEL_NAME = "Snowflake/snowflake-arctic-embed-m-v2.0"
ONNX_FILE_GPU_FP16 = "onnx/model_fp16.onnx"
ONNX_FILE_CPU_INT8 = "onnx/model_int8.onnx"
ONNX_FILE_FALLBACK = "onnx/model.onnx"

PASSAGE_PREFIX = ""
EMBED_ROW_CHUNK = 50_000
MAX_SEQ_LENGTH = 128
ENCODE_BATCH_SIZE_CPU = 32
ENCODE_BATCH_SIZE_GPU = 256
