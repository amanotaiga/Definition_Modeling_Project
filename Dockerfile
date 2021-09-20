FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y git
#RUN git clone https://github.com/NVIDIA/apex && cd apex && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./
RUN git clone https://github.com/huggingface/transformers.git && cd transformers && pip install .

RUN conda install -c anaconda jupyter
RUN conda install -c anaconda pandas
RUN conda install -c anaconda nltk
RUN pip install pytorch-lightning==0.9.0
RUN pip install git-python==1.0.3
RUN pip install transformers==3.1.0
RUN pip install sacrebleu
RUN pip install rouge-score
RUN pip install matplotlib
RUN pip install bert_score
RUN pip install sentencepiece
RUN conda install pytorch torchvision cudatoolkit=10.1 -c pytorch

WORKDIR /workspace

