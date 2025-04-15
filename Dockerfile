ARG PYTHON_VERSION=3.10
FROM python:${PYTHON_VERSION} AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install dependencies
COPY requirements.txt requirements.txt
# Create virtual environment
RUN python3 -m venv /app/chatbot

# Upgrade pip and install dependencies

RUN --mount=type=cache,target=/root/.cache/pip \
    /app/chatbot/bin/pip install --upgrade pip && \
    /app/chatbot/bin/pip install -r requirements.txt
# Copy source code and model
COPY . .
COPY LaMini-T5-61M /app/LaMini-T5-61M


# Expose port
EXPOSE 6500

# Run the application
CMD ["/app/chatbot/bin/streamlit", "run", "/app/app.py", "--server.port=6500", "--server.address=0.0.0.0"]
