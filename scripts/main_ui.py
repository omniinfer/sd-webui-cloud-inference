import modules.scripts as scripts
import html
import sys
import gradio as gr

from modules import images, script_callbacks, errors, processing, ui, shared
from modules.processing import Processed, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing
from modules.shared import opts, state, prompt_styles
from extension import api

from inspect import getmembers, isfunction
import random
import traceback
import os


DEMO_MODE = os.getenv("CLOUD_INFERENCE_DEMO_MODE")


refresh_symbol = '\U0001f504'  # 🔄


class FormComponent:
    def get_expected_parent(self):
        return gr.components.Form


class ToolButton(FormComponent, gr.Button):
    """Small button with single emoji as text, fits inside gradio forms"""

    def __init__(self, *args, **kwargs):
        classes = kwargs.pop("elem_classes", [])
        super().__init__(*args, elem_classes=["tool", *classes], **kwargs)

    def get_block_name(self):
        return "button"


class DataBinding:
    def __init__(self):
        self.enable_cloud_inference = None

        # internal component
        self.txt2img_prompt = None
        self.txt2img_neg_prompt = None
        self.img2img_prompt = None
        self.img2img_neg_prompt = None
        self.txt2img_generate = None
        self.img2img_generate = None

        # custom component, need to sync
        self.txt2img_cloud_inference_model_dropdown = None
        self.img2img_cloud_inference_model_dropdown = None
        self.img2img_cloud_inference_checkbox = None
        self.txt2img_cloud_inference_checkbox = None

        self.txt2img_cloud_inference_suggest_prompts_checkbox = None
        self.img2img_cloud_inference_suggest_prompts_checkbox = None

        # self.cloud_api_dropdown = None

        self.remote_sd_models = None
        self.default_remote_model = None
        self.initialized = False

    def update_selected_model(self, name_index: int, selected_loras: list[str], suggest_prompts_enabled, prompt: str, neg_prompt: str):
        selected: api.StableDiffusionModel = self.remote_sd_models[name_index]
        selected_checkpoint: api.StableDiffusionModel = None
        selected_checkpoint_index: int = 0

        # if selected model is lora, then we need to get base model of it and set selected model to base model
        if selected.kind == 'lora':
            for idx, model in enumerate(self.remote_sd_models):
                if model.name == selected.dependency_model_name:
                    selected_checkpoint = model
                    # selected_checkpoint_index = idx
                    selected_loras = [selected.name]
                    break
        else:
            selected_checkpoint = selected
            # selected_checkpoint_index = name_index

        # name = self.remote_sd_models[name_index].name
        prompt = prompt
        neg_prompt = neg_prompt

        if selected.example is not None:
            if selected.example.prompts is not None and suggest_prompts_enabled:
                prompt = selected.example.prompts
                prompt = prompt.replace("\n", "")
                if len(selected_loras) > 0:
                    prompt = self._update_lora_in_prompt(
                        selected.example.prompts, selected_loras)
                prompt = prompt.replace("\n", "")
            if selected.example.neg_prompt is not None and suggest_prompts_enabled:
                neg_prompt = selected.example.neg_prompt

        return gr.Dropdown.update(
            choices=[_.display_name for _ in self.remote_sd_models],
            value=selected_checkpoint.display_name), gr.update(value=selected_loras), gr.update(value=prompt), gr.update(value=neg_prompt)

    @staticmethod
    def _update_lora_in_prompt(prompt, lora_names, weight=1):
        prompt = prompt
        add_lora_prompts = []

        prompt_split = [_.strip() for _ in prompt.split(',')]

        # add
        for lora_name in lora_names:
            if '<lora:{}:'.format(lora_name) not in prompt:
                add_lora_prompts.append("<lora:{}:{}>".format(
                    lora_name, weight))
        # delete
        for prompt_item in prompt_split:
            if prompt_item.startswith("<lora:") and prompt_item.endswith(">"):
                lora_name = prompt_item.split(":")[1]
                if lora_name not in lora_names:
                    prompt_split.remove(prompt_item)

        prompt_split.extend(add_lora_prompts)

        return ", ".join(prompt_split)

    def update_selected_lora(self, lora_names, prompt):
        print("[cloud-inference] set_selected_lora", lora_names)
        return gr.update(value=self._update_lora_in_prompt(prompt, lora_names))

    def update_cloud_api(self, v):
        # TODO: support multiple cloud api provider
        print("[cloud-inference] set_cloud_api", v)
        self.cloud_api = v

    def get_selected_model_loras(self):
        ret = []
        for ckpt in self.remote_sd_models:
            if ckpt.name == self.selected_checkpoint.name:
                for lora_name in ckpt.child:
                    ret.append(lora_name)
        return ret

    def get_model_loras_cohices(self, base=None):
        ret = []
        for model in self.remote_sd_models:
            if model.kind == 'lora':
                ret.append(model.name)
        return ret

    def choice_to_model(self, choice):  # display_name -> sd_name
        for model in self.remote_sd_models:
            if model.display_name == choice:
                return model

    def get_model_ckpt_choices(self):
        return [_.display_name for _ in self.remote_sd_models]


class CloudInferenceScript(scripts.Script):
    # Extension title in menu UI
    def title(self):
        return "Cloud Inference"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def ui(self, is_img2img):
        tabname = "txt2img"
        if is_img2img:
            tabname = "img2img"

        # data initialize, TODO: move
        if _binding.remote_sd_models is None or len(_binding.remote_sd_models) == 0:
            _binding.remote_sd_models = api.get_instance().list_models()

        top_n = min(len(_binding.remote_sd_models), 50)
        if _binding.default_remote_model is None:
            _binding.default_remote_model = random.choice(
                _binding.remote_sd_models[:top_n]).display_name if len(_binding.remote_sd_models) > 0 else None

        # define ui layouts
        with gr.Accordion('Cloud Inference', open=True):
            with gr.Row():
                cloud_inference_checkbox = gr.Checkbox(
                    label="Enable Cloud Inference",
                    value=lambda: shared.opts.data.get(
                        "cloud_inference_default_enabled", False),
                    visible=not shared.opts.data.get(
                        "cloud_inference_checkbox_hidden", True),
                    elem_id="{}_cloud_inference_checkbox".format(tabname))

                cloud_inference_suggest_prompts_checkbox = gr.Checkbox(
                    value=True,
                    label="Suggest Prompts",
                    elem_id="{}_cloud_inference_suggest_prompts_checkbox".format(tabname))

            with gr.Row():
                gr.Dropdown(
                    label="Service Provider",
                    choices=["Omniinfer"],
                    value="Omniinfer",
                    elem_id="{}_cloud_api_dropdown".format(tabname)
                )

                cloud_inference_model_dropdown = gr.Dropdown(
                    label="Quick Select (Checkpoint/Lora)",
                    choices=_binding.get_model_ckpt_choices(),
                    value=lambda: _binding.default_remote_model,
                    type="index",
                    elem_id="{}_cloud_inference_model_dropdown".format(tabname))

                refresh_button = ToolButton(
                    value=refresh_symbol, elem_id="{}_cloud_inference_refersh_button".format(tabname))

            with gr.Row():
                cloud_inference_lora_dropdown = gr.Dropdown(
                    choices=_binding.get_model_loras_cohices(),
                    label="Lora",
                    elem_id="{}_cloud_inference_lora_dropdown", multiselect=True)

            # define events of components.
            # auto fill prompt after select model
            cloud_inference_model_dropdown.select(
                fn=_binding.update_selected_model,
                inputs=[
                    cloud_inference_model_dropdown,
                    cloud_inference_lora_dropdown,
                    cloud_inference_suggest_prompts_checkbox,
                    getattr(_binding, "{}_prompt".format(tabname)),
                    getattr(_binding, "{}_neg_prompt".format(tabname))

                ],
                outputs=[
                    cloud_inference_model_dropdown,
                    cloud_inference_lora_dropdown,
                    getattr(_binding, "{}_prompt".format(tabname)),
                    getattr(_binding, "{}_neg_prompt".format(tabname))
                ])

            # auto fill prompt after select lora
            cloud_inference_lora_dropdown.select(
                fn=lambda x, y: _binding.update_selected_lora(x, y),
                inputs=[
                    cloud_inference_lora_dropdown,
                    getattr(_binding, "{}_prompt".format(tabname))
                ],

                outputs=getattr(_binding, "{}_prompt".format(tabname)),
            )

            def _model_refresh():
                api.get_instance().refresh_models()
                _binding.remote_sd_models = api.get_instance().list_models()
                return gr.update(choices=[_.display_name for _ in _binding.remote_sd_models]), gr.update(choices=[_.name for _ in _binding.remote_sd_models if _.kind == 'lora'])

            refresh_button.click(
                fn=_model_refresh,
                inputs=[],
                outputs=[cloud_inference_model_dropdown,
                         cloud_inference_lora_dropdown])

        return [cloud_inference_checkbox, cloud_inference_model_dropdown]


_binding = None
if _binding is None:
    _binding = DataBinding()
    from scripts.proxy import _proxy
    _proxy._binding = _binding
    _proxy.monkey_patch()


print('Loading extension: sd-webui-cloud-inference')


def on_after_component_callback(component, **_kwargs):
    if type(component) is gr.Button and getattr(component, 'elem_id', None) == 'txt2img_generate':
        _binding.txt2img_generate = component

    if type(component) is gr.Button and getattr(component, 'elem_id', None) == 'img2img_generate':
        _binding.img2img_generate = component

    if type(component) is gr.Textbox and getattr(component, 'elem_id', None) == 'txt2img_prompt':
        _binding.txt2img_prompt = component
    if type(component) is gr.Textbox and getattr(component, 'elem_id', None) == 'txt2img_neg_prompt':
        _binding.txt2img_neg_prompt = component
    if type(component) is gr.Textbox and getattr(component, 'elem_id', None) == 'img2img_prompt':
        _binding.img2img_prompt = component
    if type(component) is gr.Textbox and getattr(component, 'elem_id', None) == 'img2img_neg_prompt':
        _binding.img2img_neg_prompt = component

    if type(component) is gr.Checkbox and getattr(component, 'elem_id', None) == 'txt2img_cloud_inference_checkbox':
        _binding.txt2img_cloud_inference_checkbox = component
    if type(component) is gr.Checkbox and getattr(component, 'elem_id', None) == 'img2img_cloud_inference_checkbox':
        _binding.img2img_cloud_inference_checkbox = component
    if type(component) is gr.Checkbox and getattr(component, 'elem_id', None) == 'txt2img_cloud_inference_suggest_prompts_checkbox':
        _binding.txt2img_cloud_inference_suggest_prompts_checkbox = component
    if type(component) is gr.Checkbox and getattr(component, 'elem_id', None) == 'img2img_cloud_inference_suggest_prompts_checkbox':
        _binding.img2img_cloud_inference_suggest_prompts_checkbox = component

    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'txt2img_cloud_inference_model_dropdown':
        _binding.txt2img_cloud_inference_model_dropdown = component
    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'img2img_cloud_inference_model_dropdown':
        _binding.img2img_cloud_inference_model_dropdown = component

    if _binding.txt2img_cloud_inference_checkbox and \
            _binding.img2img_cloud_inference_checkbox and \
            _binding.txt2img_cloud_inference_model_dropdown and \
            _binding.img2img_cloud_inference_model_dropdown and \
            _binding.txt2img_cloud_inference_suggest_prompts_checkbox and \
            _binding.img2img_cloud_inference_suggest_prompts_checkbox and \
            _binding.txt2img_generate and \
            _binding.img2img_generate and \
            not _binding.initialized:

        sync_cloud_model(_binding.txt2img_cloud_inference_model_dropdown,
                         _binding.img2img_cloud_inference_model_dropdown)

        sync_two_component(_binding.txt2img_cloud_inference_suggest_prompts_checkbox,
                           _binding.img2img_cloud_inference_suggest_prompts_checkbox, 'change')

        sync_cloud_inference_checkbox(_binding.txt2img_cloud_inference_checkbox,
                                      _binding.img2img_cloud_inference_checkbox, _binding.txt2img_generate, _binding.img2img_generate)

        _binding.initialized = True


def sync_two_component(a, b, event_name):
    def mirror(a, b):
        if a != b:
            b = a
        return a, b
    getattr(a, event_name)(fn=mirror, inputs=[a, b], outputs=[a, b])
    getattr(b, event_name)(fn=mirror, inputs=[b, a], outputs=[b, a])


def sync_cloud_model(a, b):
    def mirror(a, b):
        if a != b:
            b = a

        target_model = _binding.remote_sd_models[b]
        # TODO
        if target_model.kind == 'lora' and target_model.dependency_model_name != None:
            for model in _binding.remote_sd_models:
                if model.name == target_model.dependency_model_name:
                    b = model.display_name
                    break
        elif target_model.kind == 'checkpoint':
            b = target_model.display_name

        return _binding.remote_sd_models[a].display_name, b
    getattr(a, "select")(fn=mirror, inputs=[a, b], outputs=[a, b])
    getattr(b, "select")(fn=mirror, inputs=[b, a], outputs=[b, a])


def sync_cloud_inference_checkbox(txt2img_checkbox, img2img_checkbox, txt2img_generate_button, img2img_generate_button):
    def mirror(source, target):
        enabled = source

        if source != target:
            target = source

        button_text = "Generate"
        if enabled:
            button_text = "Generate (cloud)"
        return source, target, button_text, button_text

    txt2img_checkbox.change(fn=mirror, inputs=[txt2img_checkbox, img2img_checkbox], outputs=[
                            txt2img_checkbox, img2img_checkbox, txt2img_generate_button, img2img_generate_button])
    img2img_checkbox.change(fn=mirror, inputs=[img2img_checkbox, txt2img_checkbox], outputs=[
                            img2img_checkbox, txt2img_checkbox, txt2img_generate_button, img2img_generate_button])


def on_ui_settings():
    section = ('cloud_inference', "Cloud Inference")
    shared.opts.add_option("cloud_inference_default_enabled", shared.OptionInfo(
        False, "Cloud Inference Default Enabled", component=gr.Checkbox, section=section))
    shared.opts.add_option("cloud_inference_checkbox_hidden", shared.OptionInfo(
        False, "Cloud Inference Checkbox Hideen", component=gr.Checkbox, section=section))


script_callbacks.on_after_component(on_after_component_callback)
script_callbacks.on_ui_settings(on_ui_settings)
