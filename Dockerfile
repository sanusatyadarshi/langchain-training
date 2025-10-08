# LangChain Workshop Docker Image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV JUPYTER_ENABLE_LAB=yes

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv for faster package installation
RUN pip install --upgrade pip uv

# Set working directory
WORKDIR /workshop

# Copy requirements first for better caching
COPY requirements/requirements.txt .

# Install Python packages using uv
RUN uv pip install --system -r requirements.txt

# Create necessary directories
RUN mkdir -p /root \
    && mkdir -p /workshop/data \
    && mkdir -p /workshop/faiss_indexes

# Copy all workshop files
COPY task1/ ./task1/
COPY task2/ ./task2/
COPY task3/ ./task3/
COPY task4/ ./task4/
COPY task5/ ./task5/
COPY task6/ ./task6/
COPY task7/ ./task7/
COPY data/ ./data/
COPY *.ipynb ./
COPY .env.example ./
COPY workshop_config.py ./

# Copy startup script
COPY start.sh .
RUN chmod +x start.sh

# Set environment variables for the workshop
ENV WORKSHOP_PATH="/workshop"

# LiteLLM/OpenAI API Configuration (can be overridden by --env-file)
ENV OPENAI_API_BASE=""
ENV OPENAI_API_KEY=""
ENV DEFAULT_MODEL="gpt-4"
ENV FAST_MODEL="gpt-3.5-turbo"
ENV CODING_MODEL="gpt-4"
ENV CREATIVE_MODEL="gpt-4"
ENV USE_REAL_API="false"
ENV API_TIMEOUT="30"
ENV MAX_TOKENS="1000"
ENV DEBUG_MODE="false"

# Expose ports for Jupyter and Gradio
EXPOSE 8888 7860

# Configure Jupyter
RUN jupyter lab --generate-config && \
    echo "c.ServerApp.ip = '0.0.0.0'" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.port = 8888" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.open_browser = False" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.token = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.password = ''" >> ~/.jupyter/jupyter_lab_config.py && \
    echo "c.ServerApp.allow_root = True" >> ~/.jupyter/jupyter_lab_config.py

# Default command
CMD ["./start.sh"]