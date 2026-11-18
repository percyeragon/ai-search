FROM docker-virtual.dev.dvz-mv.net/python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install \
    --index-url https://dev.dvz-mv.net:443/artifactory/api/pypi/pypi-remote/simple/ \
    --trusted-host dev.dvz-mv.net \
    -r requirements.txt

COPY models/de_50k.txt models/de_50k.txt
COPY data/pgeocode/ /root/.cache/pgeocode/
COPY src/ src/

EXPOSE 8000

CMD ["uvicorn", "src.api.rest_service:app", "--host", "0.0.0.0", "--port", "8000"]