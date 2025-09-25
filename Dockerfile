FROM debian:stable-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
  ffmpeg ca-certificates python3 python3-pip && \
  rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir fastapi "uvicorn[standard]" pydantic

WORKDIR /app
COPY app.py /app/app.py

# Railway fournit $PORT ; on s’aligne dessus
ENV PORT=8080
EXPOSE 8080
CMD ["sh","-c","uvicorn app:app --host 0.0.0.0 --port ${PORT:-8080}"]
