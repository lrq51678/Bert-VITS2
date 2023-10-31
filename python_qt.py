import os
import webbrowser

import gradio as gr
import yaml

from config import yml_config

bert_model_paths = [
    "./bert/bert-base-japanese-v3/pytorch_model.bin",
    "./bert/bert-large-japanese-v2/pytorch_model.bin",
    "./bert/chinese-roberta-wwm-ext-large/pytorch_model.bin",
    "./bert/deberta-v2-large-japanese/pytorch_model.bin",
    "./bert/deberta-v3-large/pytorch_model.bin",
    "./bert/deberta-v3-large/pytorch_model.generator.bin",
    "./bert/deberta-v3-large/spm.model",
]

emo_model_paths = [
    "./emotional/wav2vec2-large-robust-12-ft-emotion-msp-dim/pytorch_model.bin"
]


def check_if_exists_model(paths: list[str]):
    check_results = {path: os.path.exists(path) for path in paths}
    val = [path for path, exists in check_results.items() if exists]
    return val


def check_bert_models():
    return gr.CheckboxGroup(value=check_if_exists_model(bert_model_paths))


def check_emo_models():
    return gr.CheckboxGroup(value=check_if_exists_model(emo_model_paths))


def load_yaml_data_in_raw(yml_path=yml_config):
    with open(yml_path, 'r', encoding='utf-8') as file:
        # data = yaml.safe_load(file)
        data = file.read()
    return str(data)


def load_yaml_data_in_fact(yml_path=yml_config):
    with open(yml_path, 'r', encoding='utf-8') as file:
        yml = yaml.safe_load(file)
        # data = file.read()
    return yml


def write_yaml_data_in_fact(yml, yml_path=yml_config):
    with open(yml_path, 'w', encoding='utf-8') as file:
        yaml.safe_dump(yml, file, allow_unicode=True)
        # data = file.read()
    return yml


def modify_data_path(data_path):
    yml = load_yaml_data_in_fact()
    yml['dataset_path'] = data_path
    write_yaml_data_in_fact(yml)
    return gr.Dropdown(value=data_path), gr.Code(value=load_yaml_data_in_raw())


def modify_filelist_path(data_path):
    yml = load_yaml_data_in_fact()
    yml['preprocess_text']['transcription_path'] = data_path
    write_yaml_data_in_fact(yml)
    return gr.Dropdown(value=data_path), gr.Code(value=load_yaml_data_in_raw())


if __name__ == '__main__':
    init_yml = load_yaml_data_in_fact()
    with gr.Blocks(title="Bert-VITS-2-v2.0-管理器",
                   theme=gr.themes.Soft(),
                   css=os.path.abspath("./css/custom.css")
                   ) as app:
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("模型状态"):
                    CheckboxGroup_bert_models = gr.CheckboxGroup(
                        label="检测bert模型状态",
                        info="对应文件夹下必须有对应的模型文件 (前2个是兼容旧模型的，也可以不下载)",
                        choices=bert_model_paths,
                        value=check_if_exists_model(bert_model_paths),
                        interactive=False
                    )
                    check_pth_btn1 = gr.Button(value="检查bert模型状态")
                    CheckboxGroup_emo_models = gr.CheckboxGroup(
                        label="检测emo模型状态",
                        info="对应文件夹下必须有对应的模型文件",
                        choices=emo_model_paths,
                        value=check_if_exists_model(emo_model_paths),
                        interactive=False
                    )
                    check_pth_btn2 = gr.Button(value="检查emo模型状态")
                with gr.TabItem("预处理"):
                    with gr.Row():
                        dropdown_data_path = gr.Dropdown(
                            label="选择数据集存放路径 (右侧的dataset_path)",
                            info="详情可见config.yml，点击下方按钮即可更改",
                            interactive=True,
                            allow_custom_value=True,
                            choices=[init_yml['dataset_path']],
                            value=init_yml['dataset_path']
                        )
                        data_path_btn = gr.Button(value="确认更改存放路径", variant="primary")
                    with gr.Tabs():
                        with gr.TabItem("1. 音频重采样"):
                            with gr.Row():
                                resample_in_box = gr.Textbox(
                                    label="输入音频文件夹 in_dir",
                                    lines=1,
                                    interactive=False
                                )
                                resample_out_box = gr.Textbox(
                                    label="输出音频文件夹 out_dir",
                                    lines=1,
                                    interactive=False
                                )
                                resample_config_btn = gr.Button(
                                    value="确认重采样配置",
                                    variant="secondary",
                                )
                            with gr.Row():
                                resample_btn = gr.Button(
                                    value="1. 音频重采样",
                                    variant="primary",
                                )
                            with gr.Row():
                                resample_status = gr.Textbox(
                                    placeholder="执行重采样后得到结果",
                                    show_label=False,
                                    lines=3,
                                    interactive=False
                                )
                        with gr.TabItem("2. 生成转写文本以及提取"):
                            with gr.Row():
                                slider_transcribe = gr.Slider(
                                    label="转写进程数",
                                    info="目的路径与前一节一致 in_dir",
                                    minimum=1,
                                    maximum=10,
                                    value=1
                                )
                                transcribe_btn = gr.Button(
                                    value="2.1 转写文本到.lab文件",
                                    interactive=True
                                )
                            with gr.Row():
                                clean_txt_box = gr.Textbox(
                                    label="非法字符集",
                                    lines=1,
                                    value="{}<>",
                                    interactive=True
                                )
                                clean_trans_btn = gr.Button(
                                    value="2.2 清洗标注文件"
                                )
                            with gr.Row():
                                dropdown_filelist_path = gr.Dropdown(
                                    interactive=True,
                                    label="输入filelist路径",
                                    allow_custom_value=True,
                                    choices=[init_yml['preprocess_text']['transcription_path']],
                                    value=init_yml['preprocess_text']['transcription_path']
                                )
                                filelist_path_btn = gr.Button(
                                    value="更新filelist路径"
                                )
                            with gr.Row():
                                extract_list_btn = gr.Button(
                                    value="2.3 提取到数据集的filelists下",
                                    variant="primary"
                                )
            with gr.Tabs():
                with gr.TabItem("yaml配置文件状态"):
                    code_config_yml = gr.Code(
                        interactive=False,
                        label=yml_config,
                        value=load_yaml_data_in_raw(),
                        language="yaml",
                        elem_id="yml_code"
                    )
                with gr.TabItem("带注释的yaml配置文件"):
                    code_default_yml = gr.Code(
                        interactive=False,
                        label="default_config.yml",
                        value=load_yaml_data_in_raw("default_config.yml"),
                        language="yaml",
                        elem_id="yml_code"
                    )
                with gr.TabItem("其他状态"):
                    pre_pth_btn2 = gr.Button(value="预处理")

        check_pth_btn1.click(fn=check_bert_models,
                             inputs=[],
                             outputs=[CheckboxGroup_bert_models])
        check_pth_btn2.click(fn=check_emo_models,
                             inputs=[],
                             outputs=[CheckboxGroup_emo_models])
        data_path_btn.click(fn=modify_data_path,
                            inputs=[dropdown_data_path],
                            outputs=[dropdown_data_path, code_config_yml])
        filelist_path_btn.click(fn=modify_filelist_path,
                                inputs=[dropdown_filelist_path],
                                outputs=[dropdown_filelist_path, code_config_yml])
    os.environ["no_proxy"] = "localhost,127.0.0.1,0.0.0.0"
    webbrowser.open("http://127.0.0.1:6006")
    app.launch(share=False, server_port=6006)
