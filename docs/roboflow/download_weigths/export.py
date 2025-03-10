import os
os.environ["MODEL_CACHE_DIR"] = "./models_cache"
from inference.models.utils import get_model

model_id = "<id_model>"
model = get_model(model_id=model_id, api_key="<api_key>")

