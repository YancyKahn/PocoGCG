
```
bash run_llama3_poco.sh > output_llama3_poco.log 2>&1 &

bash run_llama3_gcg.sh > output_llama3_gcg.log 2>&1 &

bash run_mistral_poco.sh > output_mistral_poco.log 2>&1 &

bash run_mistral_gcg.sh > output_mistral_gcg.log 2>&1 &

bash run_llama2_poco_no_padding.sh > output_llama2_poco_no_padding.log 2>&1 &

bash run_llama2_poco_padding_10.sh > output_llama2_poco_padding_10.log 2>&1 &

bash run_llama2_poco_padding_20.sh > output_llama2_poco_padding_20.log 2>&1 &

bash run_llama2_poco.sh > output_llama2_poco.log 2>&1 &

bash run_vicuna_gcg.sh > output_vicuna_gcg.log 2>&1 &

bash run_llama2_refuse.sh > output_llama2_refuse.log 2>&1 &

bash run_llama2_special_suffix.sh > output_llama2_special_suffix.log 2>&1 &
```

docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

docker run --gpus all -it \
  -v /root/Poco:/root/Poco \
  -v /root/llm:/root/llm \
  nvcr.io/nvidia/pytorch:22.04-py3