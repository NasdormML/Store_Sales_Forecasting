# Базовый образ с Python и Jupyter
FROM jupyter/base-notebook:python-3.11

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser"]
