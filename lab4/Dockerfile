FROM harbor-repo.vmware.com/zyajing/pytorch-training:1.11.0-gpu-py38-cu113-ubuntu20.04

RUN pip install --no-cache-dir -U \
    notebook \
    jupyter \
    datasets \
    transformers\ 
    sacrebleu \
    sentencepiece

WORKDIR /

EXPOSE 8888

CMD ["sh","-c", "jupyter notebook --notebook-dir=/ --ip=0.0.0.0 --no-browser --allow-root --port=8888 --NotebookApp.token='' --NotebookApp.password='' --NotebookApp.allow_origin='*' --NotebookApp.base_url=/"]
