FROM python:3.13-slim
WORKDIR /bot
COPY . .
# Standard library only. Runtime has no network and writes no persistent files.
USER 65534:65534
ENTRYPOINT ["python3", "-u", "/bot/bot/main.py"]
