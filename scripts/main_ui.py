import modules.scripts as scripts
import html
import sys
import gradio as gr
import importlib

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
        self.enable_remote_inference = None

        # self.remote_inference_enabled = True if DEMO_MODE else False

        # internal state
        self.txt2img_prompt = None
        self.txt2img_neg_prompt = None
        self.txt2img_generate = None
        self.img2img_generate = None

        # custom component
        self.txt2img_enable_remote_inference = None
        self.img2img_enable_remote_inference = None
        self.remote_model_dropdown = None
        self.remote_lora_dropdown = None
        self.cloud_api_dropdown = None
        self.suggest_prompts_checkbox = None

        self.remote_sd_models = None

        self.initialized = False

    def set_remote_inference_enabled(self, v):
        print("[cloud-inference] set_remote_inference_enabled", v)
        return v

    def update_remote_inference_enabled(self, v):
        print("[cloud-infernece] set_remote_inference_enabled", v)
        self.remote_inference_enabled = v

        if v:
            return v, "Generate (cloud)"
        return v, "Generate"

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
        with gr.Accordion('Cloud Inference', open=True):
            with gr.Row():

                inference_button_visilbe = True if not DEMO_MODE else False

                if _binding.enable_remote_inference is None:
                    _binding.enable_remote_inference = gr.Checkbox(
                        value=not inference_button_visilbe,
                        visible=False,
                        label="Enable",
                        elem_id="enable_remote_inference")

                if is_img2img:
                    _binding.img2img_enable_remote_inference = gr.Checkbox(
                        value=not inference_button_visilbe,
                        visble=inference_button_visilbe,
                        label="Enable",
                        elem_id="img2img_enable_remote_inference")

                    _binding.enable_remote_inference.change(
                        fn=lambda x: _binding.update_remote_inference_enabled(
                            x),
                        inputs=[_binding.enable_remote_inference],
                        outputs=[
                            _binding.img2img_enable_remote_inference,
                            _binding.img2img_generate
                        ])

                    _binding.img2img_enable_remote_inference.change(
                        fn=lambda x: _binding.set_remote_inference_enabled(
                            x),
                        inputs=[_binding.img2img_enable_remote_inference],
                        outputs=[_binding.enable_remote_inference])

                else:
                    _binding.txt2img_enable_remote_inference = gr.Checkbox(
                        value=not inference_button_visilbe,
                        visible=inference_button_visilbe,
                        label="Enable",
                        elem_id="txt2img_enable_remote_inference")

                    _binding.enable_remote_inference.change(
                        fn=lambda x: _binding.
                        update_remote_inference_enabled(x),
                        inputs=[_binding.enable_remote_inference],
                        outputs=[
                            _binding.txt2img_enable_remote_inference,
                            _binding.txt2img_generate
                        ])

                    _binding.txt2img_enable_remote_inference.change(
                        fn=lambda x: _binding.set_remote_inference_enabled(
                            x),
                        inputs=[_binding.txt2img_enable_remote_inference],
                        outputs=[_binding.enable_remote_inference])

                _binding.suggest_prompts_checkbox = gr.Checkbox(
                    value=True,
                    label="Suggest Prompts",
                    elem_id="suggest_prompts_enabled")

            with gr.Row():
                # api provider
                _binding.cloud_api_dropdown = gr.Dropdown(
                    label="Service Provider",
                    choices=["Omniinfer"],
                    value="Omniinfer",
                )
                _binding.cloud_api_dropdown.select(
                    fn=_binding.update_cloud_api,
                    inputs=[_binding.cloud_api_dropdown],
                )

                # remote checkpoint
                if not _binding.initialized:
                    try:
                        _binding.remote_sd_models = api.get_instance().list_models()
                        if _binding.remote_sd_models is None or len(_binding.remote_sd_models) == 0:
                            api.get_instance().refresh_models()
                            _binding.remote_sd_models = api.get_instance().list_models()
                    except Exception as e:
                        print(traceback.format_exc())

                    def select_default_checkpoint_of_top_n():
                        top_n = min(len(_binding.remote_sd_models), 50)
                        _binding.selected_checkpoint = random.choice(
                            _binding.remote_sd_models[:top_n]) if len(
                                _binding.remote_sd_models) > 0 else None

                        print("[cloud-inference] default checkpoint {}".format(
                            _binding.selected_checkpoint.name))

                    select_default_checkpoint_of_top_n(
                    )  # TODO: random top n after refresh page

                    _binding.initialized = True

                _binding.remote_model_dropdown = gr.Dropdown(
                    label="Quick Select (Checkpoint/Lora)",
                    choices=_binding.get_model_ckpt_choices(),
                    value=lambda: _binding.selected_checkpoint.display_name,
                    type="index",
                    elem_id="remote_model_dropdown")

                refresh_button = ToolButton(
                    value=refresh_symbol, elem_id="Refresh")

            with gr.Column():
                _binding.remote_lora_dropdown = gr.Dropdown(
                    choices=_binding.get_model_loras_cohices(),
                    label="Lora",
                    elem_id="remote_lora_dropdown", multiselect=True)

                _binding.remote_model_dropdown.select(
                    fn=_binding.update_selected_model,
                    inputs=[
                        _binding.remote_model_dropdown,
                        _binding.remote_lora_dropdown,
                        _binding.suggest_prompts_checkbox,
                        _binding.txt2img_prompt, _binding.txt2img_neg_prompt
                    ],
                    outputs=[
                        _binding.remote_model_dropdown,
                        _binding.remote_lora_dropdown,
                        _binding.txt2img_prompt, _binding.txt2img_neg_prompt
                    ])
                _binding.remote_lora_dropdown.select(
                    fn=lambda x, y: _binding.update_selected_lora(x, y),
                    inputs=[
                        _binding.remote_lora_dropdown,
                        _binding.txt2img_prompt
                    ],
                    outputs=_binding.txt2img_prompt,
                )

                def _model_refresh():
                    api.get_instance().refresh_models()
                    _binding.remote_sd_models = api.get_instance().list_models()
                    return gr.update(choices=[_.display_name for _ in _binding.remote_sd_models]), gr.update(choices=[_.name for _ in _binding.remote_sd_models if _.kind == 'lora'])

                refresh_button.click(
                    fn=_model_refresh,
                    inputs=[],
                    outputs=[_binding.remote_model_dropdown,
                             _binding.remote_lora_dropdown]
                )

        if not DEMO_MODE:
            enable_remote_inference = None
            if is_img2img:
                enable_remote_inference = _binding.img2img_enable_remote_inference
            else:
                enable_remote_inference = _binding.txt2img_enable_remote_inference

            return [enable_remote_inference, _binding.remote_model_dropdown]

        return [enable_remote_inference, _binding.remote_model_dropdown]


_binding = None
if _binding is None:
    _binding = DataBinding()
    from scripts.proxy import _proxy
    _proxy._binding = _binding
    _proxy.monkey_patch()


print('Loading extension: sd-webui-cloud-inference')


def on_after_component_callback(component, **_kwargs):

    if type(component) is gr.Button and getattr(component, 'elem_id',
                                                None) == 'txt2img_generate':
        _binding.txt2img_generate = component
        # _binding.txt2img_generate.render()
    if type(component) is gr.Button and getattr(component, 'elem_id',
                                                None) == 'img2img_generate':
        _binding.img2img_generate = component
        print("[cloud-inference] img2img_generate", component)

    if type(component) is gr.Textbox and getattr(component, 'elem_id',
                                                 None) == 'txt2img_prompt':
        _binding.txt2img_prompt = component
    if type(component) is gr.Textbox and getattr(component, 'elem_id',
                                                 None) == 'txt2img_neg_prompt':
        _binding.txt2img_neg_prompt = component


script_callbacks.on_after_component(on_after_component_callback)
