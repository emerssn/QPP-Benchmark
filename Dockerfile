# Use a Debian-based image

FROM python:3.9-slim-buster



# Set environment variables

ENV LANG=C.UTF-8

ENV LC_ALL=C.UTF-8

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



# Copy requirements.txt first

COPY EvaluacionQPP/requirements.txt .



COPY . /app

# Install Python dependencies

RUN pip install --upgrade pip && \

    pip install --no-cache-dir -r requirements.txt



# Set default values for configurable parameters

ENV DATASETS="antique_test"

# ENV MAX_QUERIES=100  # Comment out or remove this line to process all queries

ENV LIST_SIZE=10

ENV NUM_RESULTS=1000

ENV METRICS="ndcg@10 ap"

ENV CORRELATIONS="kendall"

ENV USE_UEF="false"

ENV SKIP_PLOTS="false"

ENV OUTPUT_DIR="/app/output"



# Create output directory

RUN mkdir -p /app/output



# Command to run the application with configurable parameters

ENTRYPOINT ["python", "-m", "EvaluacionQPP.main"]

CMD ["--datasets", "${DATASETS}", \

     "--max-queries", "${MAX_QUERIES}", \

     "--list-size", "${LIST_SIZE}", \

     "--num-results", "${NUM_RESULTS}", \

     "--metrics", "${METRICS}", \

     "--correlations", "${CORRELATIONS}", \

     "--output-dir", "${OUTPUT_DIR}", \

     ${USE_UEF:+"--use-uef"} \

     ${SKIP_PLOTS:+"--skip-plots"}]


