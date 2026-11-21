FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY dashboard ./dashboard
COPY scripts ./scripts

ENV ALGO_DATA_DIR=/data
ENV ALGO_REPORT_DIR=/reports
ENV PYTHONUNBUFFERED=1

VOLUME ["/data", "/reports"]

EXPOSE 8501

ENTRYPOINT ["python", "-m", "src.pipeline"]
CMD ["run-daily"]
