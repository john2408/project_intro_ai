ARG PYTHON_VERSION=3.12.7
FROM python:${PYTHON_VERSION}-bullseye
LABEL authors="John Torres <john.torres.consulting@gmail.com>"

ARG spark_version="3.3.0"
ARG hadoop_version="3"
ARG scala_version
ARG spark_checksum="1e8234d0c1d2ab4462d6b0dfe5b54f2851dcd883378e0ed756140e10adfb5be4123961b521140f580e364c239872ea5a9f813a20b73c69cb6d4e95da2575c29c"
ARG openjdk_version="17"
ARG poetry_version="1.8.3"

ENV APACHE_SPARK_VERSION="${spark_version}" \
    HADOOP_VERSION="${hadoop_version}" \
    POETRY_VERSION="${poetry_version}"

RUN apt-get update && apt-get install -y --no-install-recommends \
        "openjdk-${openjdk_version}-jre-headless" \
        ca-certificates-java && \
        apt-get clean && rm -rf /var/lib/apt/lists/*

# Spark installation
WORKDIR /tmp

RUN if [ -z "${scala_version}" ]; then \
    wget -qO "spark.tgz" "https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}.tgz"; \
  else \
    wget -qO "spark.tgz" "https://archive.apache.org/dist/spark/spark-${APACHE_SPARK_VERSION}/spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}-scala${scala_version}.tgz"; \
  fi && \
  echo "${spark_checksum} *spark.tgz" | sha512sum -c - && \
  tar xzf "spark.tgz" -C /usr/local --owner root --group root --no-same-owner && \
  rm "spark.tgz"

# Configure Spark
ENV SPARK_HOME=/usr/local/spark
ENV SPARK_OPTS="--driver-java-options=-Xms1024M --driver-java-options=-Xmx4096M --driver-java-options=-Dlog4j.logLevel=info" \
    PATH="${PATH}:${SPARK_HOME}/bin"

RUN if [ -z "${scala_version}" ]; then \
    ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}" "${SPARK_HOME}"; \
  else \
    ln -s "spark-${APACHE_SPARK_VERSION}-bin-hadoop${HADOOP_VERSION}-scala${scala_version}" "${SPARK_HOME}"; \
  fi && \
  # Add a link in the before_notebook hook in order to source automatically PYTHONPATH && \
  mkdir -p /usr/local/bin/before-notebook.d && \
  ln -s "${SPARK_HOME}/sbin/spark-config.sh" /usr/local/bin/before-notebook.d/spark-config.sh


WORKDIR /

RUN pip install --upgrade pip

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi --no-root

RUN pip install -U ipywidgets

# Start docker interactive session at bash
CMD ["/bin/bash"]