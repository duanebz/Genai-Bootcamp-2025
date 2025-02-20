## Running Ollama Third-Party Service

## Choosing a Model
https://ollama.com/library/llama3.2:1b


LLM_ENDPOINT_PORT=8080               # Set desired host port
no_proxy=localhost,127.0.0.1         # Proxy bypass settings
http_proxy=http://proxy.example.com:8080  # HTTP proxy settings
https_proxy=http://proxy.example.com:8080 # HTTPS proxy settings
LLM_MODEL_ID=my-awesome-model         # Model ID to use
host_ip=127.0.0.1                    # Host IP (for local development)

Export LLM_ENDPOINT_PORT=8080 Export LLM_MODEL_ID ="llama3.2:1b" Export host_ip=172.20.2.142 docker-compose up -d