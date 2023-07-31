import modules.scripts as scripts
import gradio as gr

from modules import script_callbacks, shared
from extension import api

import random
import os


refresh_symbol = '\U0001f504'  # 🔄
favorite_symbol = '\U0001f49e'  # 💞


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

        self.txt2img_cloud_inference_vae_dropdown = None
        self.img2img_cloud_inference_vae_dropdown = None

        self.txt2img_cloud_inference_suggest_prompts_checkbox = None
        self.img2img_cloud_inference_suggest_prompts_checkbox = None

        self.remote_inference_enabled = False

        self.remote_models = None
        self.remote_model_checkpoints = None
        self.remote_model_loras = None
        self.remote_model_controlnet = None
        self.remote_model_vaes = None

        # third component
        self.txt2img_controlnet_model_dropdown = None
        self.img2img_controlnet_model_dropdown = None

        self.default_remote_model = None
        self.initialized = False

    def on_selected_model(self, name_index: int, suggest_prompts_enabled, prompt: str, neg_prompt: str):
        selected: api.StableDiffusionModel = self.find_model_by_display_name(name_index)
        selected_checkpoint = selected

        # name = self.remote_sd_models[name_index].name
        prompt = prompt
        neg_prompt = neg_prompt

        if selected.example is not None:
            if selected.example.prompts is not None and suggest_prompts_enabled:
                prompt = selected.example.prompts
                prompt = prompt.replace("\n", "")
            if selected.example.neg_prompt is not None and suggest_prompts_enabled:
                neg_prompt = selected.example.neg_prompt

        return gr.Dropdown.update(
            choices=[_.display_name for _ in self.remote_model_checkpoints],
            value=selected_checkpoint.display_name), gr.update(value=prompt), gr.update(value=neg_prompt)

    def update_models(self):
        _binding.remote_model_loras = _get_kind_from_remote_models(
            _binding.remote_models, "lora")
        _binding.remote_model_checkpoints = _get_kind_from_remote_models(
            _binding.remote_models, "checkpoint")
        _binding.remote_model_vaes = _get_kind_from_remote_models(
            _binding.remote_models, "vae")
        _binding.remote_model_controlnet = _get_kind_from_remote_models(
            _binding.remote_models, "controlnet")

    @staticmethod
    def _update_lora_in_prompt(prompt, _lora_names, weight=1):
        lora_names = []
        for lora_name in _lora_names:
            lora_names.append(
                _binding.find_model_by_display_name(lora_name).name)

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
        return gr.update(value=self._update_lora_in_prompt(prompt, lora_names))

    def update_cloud_api(self, v):
        self.cloud_api = v

    def find_model_by_display_name(self, choice):  # display_name -> sd_name
        for model in self.remote_models:
            if model.display_name == choice:
                return model


def _get_kind_from_remote_models(models, kind):
    t = []
    for model in models:
        if model.kind == kind:
            t.append(model)
    return t


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
        if _binding.remote_models is None or len(_binding.remote_models) == 0:
            _binding.remote_models = api.get_instance().list_models()
            _binding.update_models()

        top_n = min(len(_binding.remote_model_checkpoints), 50)
        if _binding.default_remote_model is None:
            _binding.default_remote_model = random.choice(
                _binding.remote_model_checkpoints[:top_n]).display_name if len(_binding.remote_model_checkpoints) > 0 else None

        default_enabled = shared.opts.data.get(
            "cloud_inference_default_enabled", False)
        if default_enabled:
            _binding.remote_inference_enabled = True

        # define ui layouts
        with gr.Accordion('Cloud Inference', open=True):
            with gr.Row():
                cloud_inference_checkbox = gr.Checkbox(
                    label="Enable Cloud Inference",
                    value=lambda: default_enabled,
                    visible=not shared.opts.data.get(
                        "cloud_inference_checkbox_hidden", False),
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
                    label="Checkpoint",
                    choices=[
                        _.display_name for _ in _binding.remote_model_checkpoints],
                    value=lambda: _binding.default_remote_model,
                    elem_id="{}_cloud_inference_model_dropdown".format(tabname))

                refresh_button = ToolButton(
                    value=refresh_symbol, elem_id="{}_cloud_inference_refersh_button".format(tabname))
                # favorite_button = ToolButton(
                #     value=favorite_symbol, elem_id="{}_cloud_inference_favorite_button".format(tabname))

            with gr.Row():
                cloud_inference_lora_dropdown = gr.Dropdown(
                    choices=[_.display_name for _ in _binding.remote_model_loras],
                    label="Lora",
                    elem_id="{}_cloud_inference_lora_dropdown", multiselect=True, scale=4)

                cloud_inference_extra_checkbox = gr.Checkbox(
                    label="Extra",
                    value=False,
                    elem_id="{}_cloud_inference_extra_subseed_show",
                    scale=1
                )

            with gr.Row(visible=False) as extra_row:
                cloud_inference_vae_dropdown = gr.Dropdown(
                    choices=["Automatic", "None"] + [
                        _.name for _ in _binding.remote_model_vaes],
                    value="Automatic",
                    label="VAE",
                    elme_id="{}_cloud_inference_vae_dropdown".format(tabname),
                )

                cloud_inference_extra_checkbox.change(lambda x: gr.update(visible=x), inputs=[
                                                      cloud_inference_extra_checkbox], outputs=[extra_row])

            # define events of components.
            # auto fill prompt after select model
            cloud_inference_model_dropdown.select(
                fn=_binding.on_selected_model,
                inputs=[
                    cloud_inference_model_dropdown,
                    cloud_inference_suggest_prompts_checkbox,
                    getattr(_binding, "{}_prompt".format(tabname)),
                    getattr(_binding, "{}_neg_prompt".format(tabname))

                ],
                outputs=[
                    cloud_inference_model_dropdown,
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
                _binding.remote_models = api.get_instance().list_models()
                _binding.update_models()

                return gr.update(choices=[_.display_name for _ in _binding.remote_model_checkpoints]), gr.update(choices=[_.display_name for _ in _binding.remote_model_loras]), gr.update(choices=["Automatic", "None"] + [_.name for _ in _binding.remote_model_vaes])

            refresh_button.click(
                fn=_model_refresh,
                inputs=[],
                outputs=[cloud_inference_model_dropdown,
                         cloud_inference_lora_dropdown,
                         cloud_inference_vae_dropdown
                         ])

        return [cloud_inference_checkbox, cloud_inference_model_dropdown, cloud_inference_vae_dropdown]


_binding = None
if _binding is None:
    _binding = DataBinding()
    from scripts.hijack import _hijack_manager
    _hijack_manager._binding = _binding
    _hijack_manager._apply_xyz()  # TOOD


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

    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'txt2img_controlnet_ControlNet-0_controlnet_model_dropdown':
        _binding.txt2img_controlnet_model_dropdown = component
    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'img2img_controlnet_ControlNet-0_controlnet_model_dropdown':
        _binding.img2img_controlnet_model_dropdown = component

    if _binding.txt2img_cloud_inference_checkbox and \
            _binding.img2img_cloud_inference_checkbox and \
            _binding.txt2img_cloud_inference_model_dropdown and \
            _binding.img2img_cloud_inference_model_dropdown and \
            _binding.txt2img_cloud_inference_suggest_prompts_checkbox and \
            _binding.img2img_cloud_inference_suggest_prompts_checkbox and \
            _binding.txt2img_generate and \
            _binding.img2img_generate and \
            _binding.txt2img_controlnet_model_dropdown and \
            _binding.img2img_controlnet_model_dropdown and \
            not _binding.initialized:

        sync_cloud_model(_binding.txt2img_cloud_inference_model_dropdown,
                         _binding.img2img_cloud_inference_model_dropdown)

        sync_two_component(_binding.txt2img_cloud_inference_suggest_prompts_checkbox,
                           _binding.img2img_cloud_inference_suggest_prompts_checkbox, 'change')

        sync_cloud_inference_checkbox(_binding.txt2img_cloud_inference_checkbox,
                                      _binding.img2img_cloud_inference_checkbox, _binding.txt2img_generate, _binding.img2img_generate, _binding.txt2img_controlnet_model_dropdown, _binding.img2img_controlnet_model_dropdown)

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
        return a, b
    getattr(a, "select")(fn=mirror, inputs=[a, b], outputs=[a, b])
    getattr(b, "select")(fn=mirror, inputs=[b, a], outputs=[b, a])


def sync_cloud_inference_checkbox(txt2img_checkbox, img2img_checkbox, txt2img_generate_button, img2img_generate_button, txt2img_controlnet_model_dropdown, img2img_controlnet_model_dropdown):
    def mirror(source, target):
        enabled = source

        if source != target:
            target = source

        button_text = "Generate"
        if enabled:
            _binding.remote_inference_enabled = True
            button_text = "Generate (cloud)"
        else:
            _binding.remote_inference_enabled = False

        controlnet_models = ["None"] + \
            [_.name for _ in _binding.remote_model_controlnet]

        if not enabled:
            return source, target, button_text, button_text, None, None

        return source, target, button_text, button_text, gr.update(value=controlnet_models[0], choices=controlnet_models), gr.update(value=controlnet_models[0], choices=controlnet_models)

    txt2img_checkbox.change(fn=mirror, inputs=[txt2img_checkbox, img2img_checkbox], outputs=[
                            txt2img_checkbox, img2img_checkbox, txt2img_generate_button, img2img_generate_button, txt2img_controlnet_model_dropdown, img2img_controlnet_model_dropdown])
    img2img_checkbox.change(fn=mirror, inputs=[img2img_checkbox, txt2img_checkbox], outputs=[
                            img2img_checkbox, txt2img_checkbox, txt2img_generate_button, img2img_generate_button, txt2img_controlnet_model_dropdown, img2img_controlnet_model_dropdown])


def on_ui_settings():
    section = ('cloud_inference', "Cloud Inference")
    shared.opts.add_option("cloud_inference_default_enabled", shared.OptionInfo(
        False, "Cloud Inference Default Enabled", component=gr.Checkbox, section=section))
    shared.opts.add_option("cloud_inference_checkbox_hidden", shared.OptionInfo(
        False, "Cloud Inference Checkbox Hideen", component=gr.Checkbox, section=section))


script_callbacks.on_after_component(on_after_component_callback)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(_hijack_manager.hijack_all)
