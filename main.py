import os
import json
from typing import List, Dict, Tuple

import openai
import gradio as gr
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

client = openai.OpenAI(
    api_key=config['api_key'],
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",  # 使用阿里云大模型API
)

# TODO: 填写以下两行：chatbot_task 和 prompt_for_task，这里以翻译任务为例
# 第一个是用于告诉用户聊天机器人可以执行的任务（注意，真正起作用的实际是prompt）
# 第二个是使聊天机器人能够执行该任务的提示词
chatbot_task = "小学数学老师（输入“开始”）"
prompt_for_task = "现在开始，你将扮演一个出小学数学题的老师，当我说开始时提供一个简单的数学题，接收到正确回答后进行下一题，否则给我答案"


# 清除对话的函数
def reset() -> List:
    return []


# 调用模型生成对话的函数
def interact_customize(chatbot: List[Tuple[str, str]], prompt: str, user_input: str, temp=1.0) -> List[Tuple[str, str]]:
    '''
    * 参数:

      - chatbot: 模型本身，存储在元组列表中的对话记录

      - prompt: 用于指定任务的提示词

      - user_input: 每轮对话中的用户输入

      - temp: 模型的温度参数。温度用于控制聊天机器人的输出。温度越高，响应越具创造性。

    '''
    try:
        messages = []
        # 添加任务提示
        messages.append({'role': 'user', 'content': prompt})

        # 构建历史对话记录
        for input_text, response_text in chatbot:
            messages.append({'role': 'user', 'content': input_text})
            messages.append({'role': 'assistant', 'content': response_text})

        # 添加当前用户输入
        # 如果在这之前再次append prompt，就等价于在每次输入前都固定它的行为，这适用于比较呆的模型和prompt，比如【翻译成中文：】
        messages.append({'role': 'user', 'content': user_input})

        # 发送请求到对应的 API
        response = client.chat.completions.create(
            model="qwen-turbo",  # 使用阿里云 DashScope 的模型
            messages=messages,  # 传递消息记录
            temperature=temp,
            max_tokens=200,
        )

        # 将响应添加到对话记录中
        chatbot.append((user_input, response.choices[0].message.content))

    except Exception as e:
        print(f"发生错误：{e}")
        chatbot.append((user_input, f"抱歉，发生了错误：{e}"))

    return chatbot


# 导出对话记录的函数
def export_customized(chatbot: List[Tuple[str, str]], description: str) -> None:
    '''
    * 参数:

      - chatbot: 模型的对话记录，存储在元组列表中

      - description: 此任务的描述

    '''
    target = {"chatbot": chatbot, "description": description}
    with open("files/part3.json", "w") as file:
        json.dump(target, file)


# 生成 Gradio 的UI界面
with gr.Blocks() as demo:
    gr.Markdown("# 第3部分：定制化任务\n聊天机器人可以执行某项任务，试着与它互动吧！")
    chatbot = gr.Chatbot()
    desc_textbox = gr.Textbox(label="任务描述", value=chatbot_task, interactive=False)
    prompt_textbox = gr.Textbox(label="提示词", value=prompt_for_task, visible=False)
    input_textbox = gr.Textbox(label="输入", value="")

    with gr.Column():
        gr.Markdown("# 温度调节\n温度用于控制聊天机器人的输出。温度越高，响应越具创造性。")
        temperature_slider = gr.Slider(0.0, 2.0, 1.0, step=0.1, label="温度")

    with gr.Row():
        sent_button = gr.Button(value="发送")
        reset_button = gr.Button(value="重置")

    with gr.Column():
        gr.Markdown("# 保存结果\n当你对结果满意后，点击导出按钮保存结果。")
        export_button = gr.Button(value="导出")

    # 连接按钮与函数
    sent_button.click(interact_customize, inputs=[chatbot, prompt_textbox, input_textbox, temperature_slider],
                      outputs=[chatbot])
    reset_button.click(reset, outputs=[chatbot])
    export_button.click(export_customized, inputs=[chatbot, desc_textbox])

# 启动 Gradio 界面
demo.launch(debug=True)