FROM ghcr.io/e-dream-ai/gpu-container-ltx:latest

ENV PYTHONUNBUFFERED=1

WORKDIR /app
COPY worker/ /app/
RUN chmod +x /app/start.sh

ENV LTX_WORKFLOW_PATH=/app/workflow_base.json

CMD ["/app/start.sh"]
