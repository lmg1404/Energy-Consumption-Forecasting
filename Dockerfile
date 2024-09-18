FROM continuumio/miniconda3
WORKDIR /app
COPY environment.yml .
RUN conda env create -f environment.yml
RUN conda init bash
SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]
COPY . .
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--port", "8888", "--no-browser", "--allow-root"]