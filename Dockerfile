# Use a Debian-based image
FROM python:3.9-slim-bookworm

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install Java and other dependencies
RUN apt-get update && \
    apt-get install -y openjdk-17-jdk && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the entire project
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r EvaluacionQPP/requirements.txt

# Pre-download NLTK corpora used by main.py
ENV NLTK_DATA=/usr/share/nltk_data
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab stopwords

# Default command that matches your usage
CMD ["python", "-X", "utf8", "-m", "EvaluacionQPP.main", \
     "--datasets", "antique_test", \
     "--num-results", "1000", \
     "--correlations", "kendall", \
     "--use-uef"]


