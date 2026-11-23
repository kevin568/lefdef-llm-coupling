Base model for the LLM ranker of lefdef-llm-coupling, for machines that reach github.com but not huggingface.co.
This ref is not fetched by git clone / git pull. tools/install_llm_cpu.sh fetches it into a temporary folder, joins
model.safetensors from its parts, and checks every file against tools/lock/qwen2.5-3b-instruct-bnb-4bit.sha256.

By hand:
  git init --bare /tmp/m && git -C /tmp/m fetch --depth 1 https://github.com/kevin568/lefdef-llm-coupling.git \
      refs/model/qwen2.5-3b-instruct-bnb-4bit
  for i in $(seq -w 0 21); do git -C /tmp/m show FETCH_HEAD:model.safetensors.part$i; done > model.safetensors

See NOTICE and LICENSE (Qwen RESEARCH LICENSE AGREEMENT).
