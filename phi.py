import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
from langchain.llms import HuggingFacePipeline
from langchain import PromptTemplate, LLMChain
import torch

def main():
    st.title("Game story Generator")

    # User input prompt
    prompt = st.text_area("Enter Prompt:")

    if st.button("Build"):
        response = generate_response(prompt)
        st.text("Generated Response:")
        st.write(response)

def generate_response(prompt):
    tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
    quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=True)

    base_model = AutoModelForCausalLM.from_pretrained(
        "microsoft/phi-2",
        torch_dtype=torch.float32,
        device_map='auto',
        quantization_config=quantization_config
    )

    pipe = pipeline(
        "text-generation",
        model=base_model,
        tokenizer=tokenizer,
        max_length=256,
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.2
    )

    pipe.model.config.pad_token_id = pipe.model.config.eos_token_id

    local_llm = HuggingFacePipeline(pipeline=pipe)

    template = """respond to the instruction below. behave like a chatbot and respond to the user. try to be helpful.
    ### Instruction:
    {instruction}
    Answer:"""
    prompt_template = PromptTemplate(template=template, input_variables=["instruction"])

    llm_chain = LLMChain(prompt=prompt_template, llm=local_llm)
    return llm_chain.run(prompt)

if __name__ == "__main__":
    main()
