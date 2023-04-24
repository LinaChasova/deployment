FROM continuumio/miniconda3

WORKDIR /app

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml

# Make RUN commands use the new environment:
SHELL ["conda", "run", "-n", "wine-quality", "/bin/bash", "-c"]

# Demonstrate the environment is activated:
RUN echo "Make sure fast api is installed:"
RUN python -c "import fastapi"

# The code to run when container is started:
COPY app.py .
COPY src src/
COPY models models/
ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "wine-quality", "python", "app.py"]