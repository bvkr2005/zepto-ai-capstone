FROM python:3.11-slim

WORKDIR /app

COPY support_assistant /app/support_assistant

RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic \
    chromadb \
    sentence-transformers \
    langgraph \
    joblib

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]