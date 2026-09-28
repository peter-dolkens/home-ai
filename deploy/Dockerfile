# home-ai bridge: stdlib-only Python, talks to Home Assistant over REST/websocket and to Ollama over HTTP.
FROM python:3.12-slim
WORKDIR /app
COPY homeai /app/homeai
COPY configs/bridge.example.json /app/configs/bridge.example.json
ENV PYTHONUNBUFFERED=1
EXPOSE 8090
CMD ["python", "-m", "homeai.bridge.cli", "serve", "--config", "/config/bridge.json", "--port", "8090"]
