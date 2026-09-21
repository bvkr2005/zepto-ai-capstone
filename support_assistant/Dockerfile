FROM python:3.11-slim

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY support_assistant/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY support_assistant /app/support_assistant

ENV MOCK_LLM=1

EXPOSE 7860

CMD ["uvicorn", "support_assistant.main:app", "--host", "0.0.0.0", "--port", "7860"]
