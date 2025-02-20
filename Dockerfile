# Use a Debian-based image
FROM python:3.9-slim-buster

# Set environment variables
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV PYTHONIOENCODING=utf-8
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV PATH="$JAVA_HOME/bin:$PATH"

# Install Java and other dependencies
RUN apt-get update && \
    apt-get install -y openjdk-11-jdk && \
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

# Default command that matches your usage
CMD ["python", "-X", "utf8", "-m", "EvaluacionQPP.main", \
     "--datasets", "antique_test", \
     "--num-results", "1000", \
     "--correlations", "kendall", \
     "--use-uef"]


