import os

'''
os.environ["HABANA_VISIBLE_DEVICES"] = "ALL"
os.environ["PT_HPU_ENABLE_LAZY_COLLECTIVES"] = "true"
os.environ["VLLM_MLA_DISABLE_REQUANTIZATION"] = "1"
os.environ["VLLM_RAY_DISABLE_LOG_TO_DRIVER"] = "1"
os.environ["RAY_IGNORE_UNHANDLED_ERRORS"] = "1"
'''
os.environ["VLLM_SKIP_WARMUP"] = "true"
os.environ['VLLM_CONTIGUOUS_PA'] = 'false'
os.environ['VLLM_MLA_DISABLE_REQUANTIZATION']='1'
os.environ['PT_HPU_ENABLE_LAZY_COLLECTIVES']='true'
os.environ['PT_HPU_WEIGHT_SHARING']='0'
os.environ['VLLM_MLA_PERFORM_MATRIX_ABSORPTION']='0'
os.environ['VLLM_MTP_PRINT_ACCPET_RATE']='0'
os.environ['PT_HPU_LAZY_MODE']='1'
os.environ['VLLM_DELAYED_SAMPLING']='false'
#os.environ['VLLM_USE_V1']='1'


if __name__ == "__main__":

    from vllm import LLM, SamplingParams

    # Sample prompts.
    prompts = [
        #"Hello, my name is",
        #"The president of the United States is",
        "The Hongkou district lies to the north and east of the Suzhou River. It was originally developed by American and Japanese concessionaires and in 1863 was combined with the British concession to the south to create the International Settlement. It is an important industrial area, with shipyards and factories spread out along the bank of the Huangpu in the eastern section of the district. Its best-known building, the Shanghai Dasha (Shanghai Mansions Hotel), overlooks the Huangpu. \
        The old Chinese city, which is now part of central Shanghai, is characterized by a random and labyrinthine street pattern. Until the early 20th century the area was surrounded by a wall 3 miles (5 km) long. It is now circumscribed by the two streets of Renmin Lu and Zhonghua Lu, which follow the course of the original wall; and it is bisected by the main north-south artery, Henan Nan Lu (South Henan Road). \
        Western Shanghai is primarily residential in character and is the site of the Shanghai Exhibition Center. To the southwest, the district of Xuhui, formerly Xujiahui, became a centre of Christian missionary activity in China in the 17th century. During the late 1800s, Jesuit priests established a major library, a printing establishment, an orphanage, and a meteorological observatory in the area. \
        Land use patterns in metropolitan Shanghai mirror pre-1949 real-estate-market conditions. Much of the high-value land given over to industrial plants, warehouses, and transport facilities lies close to the Huangpu and Suzhou rivers. South of the Suzhou, which is traversed by some two dozen bridges within the city, residential areas extend south from the industrial strip to the Huangpu. North of the Suzhou, residential areas are less clearly demarcated, and there is a more gradual merging of city and country in the transitional zone. Continuous urban settlement is bounded on the north by the two major east-west arteries of Zhongshan Bei Lu and Siping Lu. \
        Retail trade is concentrated in the old central business district, although the proportional volume of trade conducted there has diminished with the establishment of the industrial satellite towns and villages on the periphery of Shanghai.",
    ]
    # Create a sampling params object.
    sampling_params = SamplingParams(temperature=0.0, max_tokens=128)

    model_name = "/home/HF_models/llama-3-8b"#"/home/HF_models/GLM-4.5-Air-FP8"   #13/13+6,random
    #model_name = "/home/HF_models/GLM-4.5-Air-FP8"
    #model_name = "/home/jkyu/mtp/DeepSeek-R1-G2-static"   #12/12+8
    #model_name = "/home/jkyu/DeepSeek-R1-G2-static"
    #model_name = "/home/jkyu/DeepSeek-R1-G2-dynamic"
    #model_name = "deepseek-ai/DeepSeek-R1"
    #model_name = "/data/models/DeepSeek-R1"
    # model_name = "/data/models/DeepSeek-R1-bf16"
    # model_name = "/data/models/DeepSeek-R1-bf16-small/"
    # model_name = "meta-llama/Meta-Llama-3-8B"
    # model_name = "meta-llama/Llama-3.2-1B"

    # Create an LLM.
    '''
    llm = LLM(model=model_name,
            device="hpu",
            dtype="bfloat16",
            #   load_format="dummy",
            tensor_parallel_size=8,
            trust_remote_code=True,
            max_model_len=1024)
    '''
    llm = LLM(model=model_name,
            trust_remote_code=True,
            enforce_eager=True,
            dtype="bfloat16",
            use_v2_block_manager=True,
            tensor_parallel_size=1,
            #speculative_draft_tensor_parallel_size=8,
            max_model_len=1024,
            num_scheduler_steps=1,
            distributed_executor_backend='mp',
            gpu_memory_utilization=0.5,
            #quantization="inc",
            #kv_cache_dtype="fp8_inc",
            #enable_expert_parallel=True,
            enable_chunked_prefill=True,
            max_num_batched_tokens=256,
            #num_speculative_tokens=1,
            seed=2024)
    # Generate texts from the prompts. The output is a list of RequestOutput objects
    # that contain the prompt, generated text, and other information.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"--Prompt: {prompt!r}, Generated text: {generated_text!r}")

    '''
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"--Prompt: {prompt!r}, Generated text: {generated_text!r}")
    '''
