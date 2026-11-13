FROM python:3.11-slim
WORKDIR /app

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY model/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake model weights into the image
ARG MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
RUN python -c "from transformers import AutoTokenizer, AutoModelForCausalLM; \
AutoTokenizer.from_pretrained('${MODEL_NAME}'); AutoModelForCausalLM.from_pretrained('${MODEL_NAME}')"

COPY model/ .
ARG APP_VERSION=dev
ENV APP_VERSION=${APP_VERSION}
EXPOSE 8000
CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]