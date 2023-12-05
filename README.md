

# Fake news bot

## Install
### Dependencies:
`pip install -r requirements.txt`
- ``pytorch==2.1.1``
- ``numpy==1.23.5``
- ``transformers==4.35.2`` for huggingface transformers
- ``datasets==2.15.0``
- ``wandb==0.16.0`` for optional logging
- ``peft==0.6.2`` for model training optimization
- ``tqdm==4.66.1`` for progress bars
- ``python-telegram-bot==20.7`` for telegram bot implementation
- ``pandas==1.5.3``, ``seaborn==0.12.2``, ``matplotlib==3.7.1``, ``wordcloud==1.9.2``, ``regex==2023.6.3``, ``mdutils==1.6.0`` for the whole working with data process 


## Quick start
If you just want to try our project in action, then click on the [link](https://t.me/fsdl2023_fake_news_bot). You will be greeted by a bot with the following commands: 
- start -- for getting the bot description
- help -- for getting commands descriprions
- generate -- for headline generation. You will be provided with a list of topics, you just need to select one of them and you will receive the latest non-existent news. Be careful: sometimes they scare you with their plausibility!

## Baselines
We use GPT2LMHeadModel from transformers as a baseline model both for English and Russian language modeling, but with different initial weights. We use russian language weights from [Sber-AI](https://github.com/ai-forever/ru-gpts) and model pre-trained GPT-2 from HuggingFace for the english language. 

## Efficiency notes
The [peft](https://github.com/huggingface/peft) library was used to speed up the training process. The parameters were taken from the paper [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) and are shown here:  

   

    config = LoraConfig(
	    task_type=TaskType.CAUSAL_LM,
	    inference_mode=False,
	    r=4,
	    lora_alpha=32, 
	    target_modules=["c_attn",  "c_proj"],
	    lora_dropout=0.1,
    )


Using this library allowed us to reduce the training time from 2 hours for 1 epoch on News Category Dataset to just 45 minutes.

## Experiments
We compared the run statistics over different PEFT Model trinings with small rank parameter of 4. The runtime was the same as the PEFT Model was the same, but it was a significant improvement after the full finetuning. We see that the loss for the Russian language model is less. 

Experiment 4 was conducted on BBC dataset, a smaller one, News Category dataset was used for the third experiment. There are no significant difference in terms of loss for these two graphs, the one is just shorter because of the size of the dataset. 

![photo_2023-12-05 19 21 41](https://github.com/app1606/fakenews_bot/assets/54853680/ebae43fa-063a-4573-8790-fe83190c5237)

## Sampling / Inference
Inference example can be found in the Inference section of `notebooks/Fake_News_Generator.ipynb`. 

First you have to load the weights:

    model_to_merge = PeftModel.from_pretrained(model, path_to_peft_weights)
    merged_model = model_to_merge.merge_and_unload()
 
 Then one can use generate_headline function (defined in the notebook)   based on `model.generate()` or write own version.

## Telegram bot
To launch your own telegram bot, you will need to get a special token. The process is described in detail in [this](https://core.telegram.org/bots) tutorial.

### Deployment
We use [Kamatera](https://www.kamatera.com) portal for our server. Our bot works asynchronous and the code can be found in bot.py. 

## Todos
- Explore other embeddings
- Experiment with optimizations and hyperparameters
- Additional logging around network
- Organize code into python scripts

## Acknowledgements
Many thanks to [Panorama News Agency & Publishing House](https://panorama.pub/) for providing the dataset and [Serge Kim](https://github.com/sergevkim) for the guidance throughout the project.

<!--stackedit_data:
eyJoaXN0b3J5IjpbLTE5ODY5NzM1NDUsLTMzNDQwOTMxNSwxNT
E2MzY1Nzc3LDEzOTQ3MDA2MDAsLTg5NzA3ODQxNCwtMTYzNDEx
NDExLC0xODEwNDQ1MDg2LDE3NTc2MTcyNzgsNzE2NTM0NDE3LC
02OTY0MTM3Miw4NjI1OTQwNjJdfQ==
-->

<!--stackedit_data:
eyJoaXN0b3J5IjpbNjU2Mzc1MjYyLC0xODMwMTIwNDcwLC0xND
k3MTk4NzY0LC01MjEwNjA4OTcsNDYyNTQ3NDcxLC0xOTg2OTcz
NTQ1LC0zMzQ0MDkzMTUsMTUxNjM2NTc3NywxMzk0NzAwNjAwLC
04OTcwNzg0MTQsLTE2MzQxMTQxMSwtMTgxMDQ0NTA4NiwxNzU3
NjE3Mjc4LDcxNjUzNDQxNywtNjk2NDEzNzIsODYyNTk0MDYyXX
0=
-->
