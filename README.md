# ChatPDF v1
ChatPDF是一个可以让你和你的PDF文件对话的问答机器人，有简单的UI界面。其他格式的文件换成相应的loader就可以实现。

## Installation 安装
安装 [LangChain](https://github.com/hwchase17/langchain)和其他依赖的包。
```
pip install -r requirements.txt
```

填入你的[OpenAI API key](https://platform.openai.com/account/api-keys)
```
export OPENAI_API_KEY='sk-...'
```

## Usage 使用
将`data`文件夹中的文件替换成你的文件。

```
> streamlit run personal_assistant.py
```
在网页对话框中输入你的问题，按回车键发送，就可以得到答案。