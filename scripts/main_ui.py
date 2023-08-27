import modules.scripts as scripts
import gradio as gr
import os

from modules import script_callbacks, shared, paths_internal, ui_common
from extension import api

from collections import Counter
import random


refresh_symbol = '\U0001f504'  # 🔄
favorite_symbol = '\U0001f49e'  # 💞
model_browser_symbol = '\U0001f50d'  # 🔍


class FormComponent:
    def get_expected_parent(self):
        return gr.components.Form


class FormButton(FormComponent, gr.Button):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_block_name(self):
        return "button"


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
        self.remote_models_aliases = {}
        self.remote_model_checkpoints = None
        self.remote_model_embeddings = None
        self.remote_model_loras = None
        self.remote_model_controlnet = None
        self.remote_model_vaes = None
        self.remote_model_upscalers = None

        # refiner
        self.txt2img_checkpoint = None
        self.img2img_checkpoint = None
        self.txt2img_checkpoint_refresh = None
        self.img2img_checkpoint_refresh = None

        # third component
        self.txt2img_controlnet_model_dropdown_units = []
        self.img2img_controlnet_model_dropdown_units = []

        # upscale
        self.extras_upscaler_1 = None
        self.extras_upscaler_2 = None
        self.txt2img_hr_upscaler = None

        # backup config
        self.txt2img_hr_upscaler_original = None
        self.extras_upscaler_1_original = None
        self.extras_upscaler_2_original = None
        self.txt2img_controlnet_model_dropdown_original_units = []
        self.img2img_controlnet_model_dropdown_original_units = []
        self.txt2img_checkpoint_original = None
        self.img2img_checkpoint_original = None

        self.default_remote_model = None
        self.initialized = False

        self.bultin_refiner_supported = False
        self.ext_controlnet_installed = False

    def on_selected_model(self, name_index: int, selected_loras: list[str], selected_embedding: list[str], suggest_prompts_enabled, prompt: str, neg_prompt: str):
        selected: api.StableDiffusionModel = self.find_model_by_alias(name_index)
        selected_checkpoint = selected

        # name = self.remote_sd_models[name_index].name
        prompt = prompt
        neg_prompt = neg_prompt

        if len(selected.examples) > 0:
            example = random.choice(selected.examples)
            if suggest_prompts_enabled and example.prompts:
                prompt = example.prompts
                prompt = prompt.replace("\n", "")
                if len(selected_loras) > 0:
                    prompt = self._update_lora_in_prompt(prompt, selected_loras)
            if suggest_prompts_enabled and example.neg_prompt:
                neg_prompt = example.neg_prompt
                neg_prompt = neg_prompt.replace("\n", "")
                if len(selected_embedding) > 0:
                    neg_prompt = self._update_embedding_in_neg_prompt(neg_prompt, selected_embedding)

        return gr.Dropdown.update(
            choices=[_.alias for _ in self.remote_model_checkpoints], value=selected_checkpoint.alias),  gr.update(value=prompt), gr.update(value=neg_prompt)

    def update_models(self):
        for model in self.remote_models:
            self.remote_models_aliases[model.alias] = model

        _binding.remote_model_loras = _get_kind_from_remote_models(_binding.remote_models, "lora")
        _binding.remote_model_embeddings = _get_kind_from_remote_models(_binding.remote_models, "textualinversion")
        _binding.remote_model_checkpoints = _get_kind_from_remote_models(_binding.remote_models, "checkpoint")
        _binding.remote_model_vaes = _get_kind_from_remote_models(_binding.remote_models, "vae")
        _binding.remote_model_controlnet = _get_kind_from_remote_models(_binding.remote_models, "controlnet")
        _binding.remote_model_upscalers = _get_kind_from_remote_models(_binding.remote_models, "upscaler")

    @staticmethod
    def _update_lora_in_prompt(prompt, _lora_names, weight=1):
        lora_names = []
        for lora_name in _lora_names:
            lora_names.append(_binding.find_model_by_alias(lora_name).name)

        prompt = prompt
        add_lora_prompts = []

        prompt_split = [_.strip() for _ in prompt.split(',') if _.strip() != ""]

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

    @staticmethod
    def _update_embedding_in_neg_prompt(neg_prompt, _embedding_names):
        embedding_names = []
        for embedding_name in _embedding_names:
            name = _binding.find_model_by_alias(embedding_name).name.rsplit(".", 1)[0]  # remove extension
            embedding_names.append(name)

        neg_prompt = neg_prompt
        add_embedding_prompts = []

        neg_prompt_split = [_.strip() for _ in neg_prompt.split(',') if _.strip() != ""]

        # add
        for embedding_name in embedding_names:
            if embedding_name not in neg_prompt:
                add_embedding_prompts.append(embedding_name)
        # delete
        for prompt_item in neg_prompt_split:
            if prompt_item in embedding_names:
                neg_prompt_split.remove(prompt_item)

        neg_prompt_split.extend(add_embedding_prompts)

        return ", ".join(neg_prompt_split)

    def update_selected_lora(self, lora_names, prompt):
        return gr.update(value=self._update_lora_in_prompt(prompt, lora_names))

    def update_selected_embedding(self, embedding_names, neg_prompt):
        return gr.update(value=self._update_embedding_in_neg_prompt(neg_prompt, embedding_names))

    def update_cloud_api(self, v):
        self.cloud_api = v

    def find_model_by_alias(self, choice):  # alias -> sd_name
        for model in self.remote_models:
            if model.alias == choice:
                return model

    def find_name_by_alias(self, choice):
        for model in self.remote_models:
            if model.alias == choice:
                return model.name

    # def update_model_favorite(self, alias):
    #     model = self.find_model_by_alias(alias)
    #     if model is not None:
    #         if "favorite" in model.tags:
    #             model.tags.remove("favorite")
    #         else:
    #             model.tags.append("favorite")
    #         return gr.update(value=build_model_browser_html_for_checkpoint("txt2img", _binding.remote_model_checkpoints)), \
    #             gr.update(value=build_model_browser_html_for_loras("txt2img", _binding.remote_model_loras)), \
    #             gr.update(value=build_model_browser_html_for_embeddings("txt2img", _binding.remote_model_embeddings)), \


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
            _binding.default_remote_model = random.choice(_binding.remote_model_checkpoints[:top_n]).alias if len(_binding.remote_model_checkpoints) > 0 else None

        default_enabled = shared.opts.data.get("cloud_inference_default_enabled", False)
        if default_enabled:
            _binding.remote_inference_enabled = True

        default_suggest_prompts_enabled = shared.opts.data.get("cloud_inference_suggest_prompts_default_enabled", True)

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
                    value=lambda: default_suggest_prompts_enabled,
                    label="Suggest Prompts",
                    elem_id="{}_cloud_inference_suggest_prompts_checkbox".format(tabname))

            with gr.Row():
                gr.Dropdown(
                    label="Service Provider",
                    choices=["Omniinfer"],
                    value="Omniinfer",
                    elem_id="{}_cloud_api_dropdown".format(tabname),
                    scale=1
                )

                cloud_inference_model_dropdown = gr.Dropdown(
                    label="Checkpoint",
                    choices=[_.alias for _ in _binding.remote_model_checkpoints],
                    value=lambda: _binding.default_remote_model,
                    elem_id="{}_cloud_inference_model_dropdown".format(tabname), scale=2)

                model_browser_button = FormButton(value="{} Browser".format(model_browser_symbol), elem_classes='model-browser-button',
                                                  elem_id="{}_cloud_inference_browser_button".format(tabname), scale=0)
                refresh_button = ToolButton(value=refresh_symbol, elem_id="{}_cloud_inference_refersh_button".format(tabname))

                # model_browser_button = ToolButton(model_browser_symbol,  elem_id="{}_cloud_inference_browser_button".format(tabname))
                # favorite_button = ToolButton(
                #     value=favorite_symbol, elem_id="{}_cloud_inference_favorite_button".format(tabname))

            with gr.Row():
                cloud_inference_lora_dropdown = gr.Dropdown(
                    choices=[_.alias for _ in _binding.remote_model_loras],
                    label="Lora",
                    elem_id="{}_cloud_inference_lora_dropdown", multiselect=True, scale=4)
                cloud_inference_embedding_dropdown = gr.Dropdown(
                    choices=[_.alias for _ in _binding.remote_model_embeddings],
                    label="Embedding",
                    elem_id="{}_cloud_inference_embedding_dropdown", multiselect=True, scale=4)

                cloud_inference_extra_checkbox = gr.Checkbox(
                    label="Extra",
                    value=False,
                    elem_id="{}_cloud_inference_extra_subseed_show",
                    scale=1
                )

                # functionally
                hide_button_change_checkpoint = gr.Button('Change Cloud checkpoint', elem_id='{}_change_cloud_checkpoint'.format(tabname), visible=False)
                hide_button_change_lora = gr.Button('Change Cloud LORA', elem_id='{}_change_cloud_lora'.format(tabname), visible=False)
                hide_button_change_embedding = gr.Button('Change Cloud Embedding', elem_id='{}_change_cloud_embedding'.format(tabname), visible=False)
                # hide_button_favorite = gr.Button('Favorite', elem_id='{}_favorite'.format(tabname), visible=False)

            with gr.Box(elem_id='{}_model_browser'.format(tabname), elem_classes="popup-model-browser", visbile=False) as checkpoint_model_browser_dialog:
                with gr.Tab(label="Checkpoint", elem_id='{}_model_browser_checkpoint_tab'.format(tabname)):
                    model_checkpoint_browser_dialog_html = gr.HTML(build_model_browser_html_for_checkpoint(tabname, _binding.remote_model_checkpoints))
                with gr.Tab(label="LORA", elem_id='{}_model_browser_lora_tab'.format(tabname)):
                    model_lora_browser_dialog_html = gr.HTML(build_model_browser_html_for_loras(tabname, _binding.remote_model_loras))
                with gr.Tab(label="Embedding", elem_id='{}_model_browser_embedding_tab'.format(tabname)):
                    model_embedding_browser_dialog_html = gr.HTML(build_model_browser_html_for_embeddings(tabname, _binding.remote_model_embeddings))

            checkpoint_model_browser_dialog.visible = False
            model_browser_button.click(fn=lambda: gr.update(visible=True), inputs=[], outputs=[checkpoint_model_browser_dialog],).\
                then(fn=None, _js="function(){ modelBrowserPopup('" + tabname + "', gradioApp().getElementById('" + checkpoint_model_browser_dialog.elem_id + "')); }", show_progress=True)

            with gr.Row(visible=False) as extra_row:
                cloud_inference_vae_dropdown = gr.Dropdown(
                    choices=["Automatic", "None"] + [_.name for _ in _binding.remote_model_vaes],
                    value="Automatic",
                    label="VAE",
                    elme_id="{}_cloud_inference_vae_dropdown".format(tabname),
                )

                cloud_inference_extra_checkbox.change(lambda x: gr.update(visible=x), inputs=[
                                                      cloud_inference_extra_checkbox], outputs=[extra_row])

            # lora
            # define events of components.
            # auto fill prompt after select model
            hide_button_change_checkpoint.click(
                fn=_binding.on_selected_model,
                _js="function(a, b, c, d, e, f){ var res = desiredCloudInferenceCheckpointName; desiredCloudInferenceCheckpointName = ''; return [res, b, c, d, e, f]; }",
                inputs=[
                    cloud_inference_model_dropdown,
                    cloud_inference_lora_dropdown,
                    cloud_inference_embedding_dropdown,
                    cloud_inference_suggest_prompts_checkbox,
                    getattr(_binding, "{}_prompt".format(tabname)),
                    getattr(_binding, "{}_neg_prompt".format(tabname))
                ],
                outputs=[
                    cloud_inference_model_dropdown,
                    getattr(_binding, "{}_prompt".format(tabname)),
                    getattr(_binding, "{}_neg_prompt".format(tabname))
                ]
            )
            # dummy_component = gr.Label(visible=False)
            # hide_button_favorite.click(
            #     fn=_binding.update_model_favorite,
            #     _js='''function(){ name = desciredCloudInferenceFavoriteModelName; desciredCloudInferenceFavoriteModelName = ""; return [name]; }''',
            #     inputs=[dummy_component],
            #     outputs=[
            #         model_checkpoint_browser_dialog_html,
            #         model_lora_browser_dialog_html,
            #         model_embedding_browser_dialog_html,
            #     ],
            # )

            hide_button_change_lora.click(
                fn=lambda x, y: _binding.update_selected_lora(x, y),
                _js="function(a, b){ a.includes(desiredCloudInferenceLoraName) || a.push(desiredCloudInferenceLoraName); desiredCloudInferenceLoraName = ''; return [a, b]; }",
                inputs=[
                    cloud_inference_lora_dropdown,
                    getattr(_binding, "{}_prompt".format(tabname))
                ],
                outputs=getattr(_binding, "{}_prompt".format(tabname)),
            )
            # auto fill prompt after select lora
            cloud_inference_lora_dropdown.select(
                fn=lambda x, y: _binding.update_selected_lora(x, y),
                inputs=[
                    cloud_inference_lora_dropdown,
                    getattr(_binding, "{}_prompt".format(tabname))
                ],
                outputs=getattr(_binding, "{}_prompt".format(tabname)),
            )

            hide_button_change_embedding.click(
                fn=lambda x, y: _binding.update_selected_embedding(x, y),
                _js="function(a, b){ a.includes(desiredCloudInferenceEmbeddingName) || a.push(desiredCloudInferenceEmbeddingName); desiredCloudInferenceEmbeddingName = ''; return [a, b]; }",
                inputs=[
                    cloud_inference_embedding_dropdown,
                    getattr(_binding, "{}_neg_prompt".format(tabname))
                ],
                outputs=getattr(_binding, "{}_neg_prompt".format(tabname)),
            )
            # embeddings
            cloud_inference_embedding_dropdown.select(
                fn=lambda x, y: _binding.update_selected_embedding(x, y),
                inputs=[
                    cloud_inference_embedding_dropdown,
                    getattr(_binding, "{}_neg_prompt".format(tabname))
                ],
                outputs=[
                    getattr(_binding, "{}_neg_prompt".format(tabname)),
                ]
            )

            cloud_inference_model_dropdown.select(
                fn=_binding.on_selected_model,
                inputs=[
                    cloud_inference_model_dropdown,
                    cloud_inference_lora_dropdown,
                    cloud_inference_embedding_dropdown,
                    cloud_inference_suggest_prompts_checkbox,
                    getattr(_binding, "{}_prompt".format(tabname)),
                    getattr(_binding, "{}_neg_prompt".format(tabname))
                ],
                outputs=[
                    cloud_inference_model_dropdown,
                    getattr(_binding, "{}_prompt".format(tabname)),
                    getattr(_binding, "{}_neg_prompt".format(tabname))
                ])

            def _model_refresh(tab):
                def wrapper():
                    api.get_instance().refresh_models()
                    _binding.remote_models = api.get_instance().list_models()
                    _binding.update_models()

                    return gr.update(choices=[_.alias for _ in _binding.remote_model_checkpoints]), \
                        gr.update(choices=[_.alias for _ in _binding.remote_model_loras]), \
                        gr.update(choices=["Automatic", "None"] + [_.name for _ in _binding.remote_model_vaes]), \
                        gr.update(choices=[_.alias for _ in _binding.remote_model_embeddings]), \
                        gr.update(value=build_model_browser_html_for_checkpoint(tab, _binding.remote_model_checkpoints)), \
                        gr.update(value=build_model_browser_html_for_loras(tab, _binding.remote_model_loras)), \
                        gr.update(value=build_model_browser_html_for_embeddings(tab, _binding.remote_model_embeddings))
                return wrapper

            refresh_button.click(
                fn=_model_refresh(tabname),
                inputs=[],
                outputs=[cloud_inference_model_dropdown,
                         cloud_inference_lora_dropdown,
                         cloud_inference_embedding_dropdown,
                         cloud_inference_vae_dropdown,

                         model_checkpoint_browser_dialog_html,
                         model_lora_browser_dialog_html,
                         model_embedding_browser_dialog_html,
                         ])

        return [cloud_inference_checkbox, cloud_inference_model_dropdown, cloud_inference_vae_dropdown]


# TODO: refactor this
_binding = None
if _binding is None:
    _binding = DataBinding()
    if shared.opts.data.get("cloud_inference_default_enabled", False):
        _binding.remote_inference_enabled = True

    if os.path.isdir(os.path.join(paths_internal.extensions_dir, "sd-webui-controlnet")) and 'sd-webui-controlnet' not in shared.opts.data.get('disabled_extensions', []):
        _binding.ext_controlnet_installed = True

    try:
        import modules.processing_scripts.refiner
        _binding.bultin_refiner_supported = True
    except:
        pass

    from scripts.hijack import _hijack_manager
    _hijack_manager._binding = _binding
    _hijack_manager.hijack_onload()

    _binding.remote_models = api.get_instance().list_models()
    _binding.update_models()


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

    # example: txt2img_controlnet_ControlNet_controlnet_model_dropdown and img2img_controlnet_ControlNet-0_controlnet_model_dropdown
    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) != None and component.elem_id.startswith('txt2img_controlnet_ControlNet') and component.elem_id.endswith('_model_dropdown'):
        _binding.txt2img_controlnet_model_dropdown_units.append(component)
        _binding.txt2img_controlnet_model_dropdown_original_units.append(component.get_config())

        if _binding.remote_inference_enabled:
            component.choices = ['None'] + [_.alias for _ in _binding.remote_model_controlnet]
            component.value = component.choices[0]

    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) != None and component.elem_id.startswith('img2img_controlnet_ControlNet') and component.elem_id.endswith('_model_dropdown'):
        _binding.img2img_controlnet_model_dropdown_units.append(component)
        _binding.img2img_controlnet_model_dropdown_original_units.append(component.get_config())

        if _binding.remote_inference_enabled:
            component.choices = ['None'] + [_.alias for _ in _binding.remote_model_controlnet]
            component.value = component.choices[0]

    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'extras_upscaler_1':
        _binding.extras_upscaler_1 = component
        _binding.extras_upscaler_1_original = component.get_config()

        if _binding.remote_inference_enabled:
            component.choices = [_.alias for _ in _binding.remote_model_upscalers]
            component.value = component.choices[0]

    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'extras_upscaler_2':
        _binding.extras_upscaler_2 = component
        _binding.extras_upscaler_2_original = component.get_config()

        if _binding.remote_inference_enabled:
            component.choices = ['None'] + [_.alias for _ in _binding.remote_model_upscalers]
            component.value = component.choices[0]

    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'txt2img_hr_upscaler':
        _binding.txt2img_hr_upscaler = component
        _binding.txt2img_hr_upscaler_original = component.get_config()

        if _binding.remote_inference_enabled:
            component.choices = [_.alias for _ in _binding.remote_model_upscalers]
            component.value = component.choices[0]

    # txt2img refiner
    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'txt2img_checkpoint':
        _binding.txt2img_checkpoint = component
        _binding.txt2img_checkpoint_original = component.get_config()

        if _binding.remote_inference_enabled:
            component.choices = ["None"] + [_.name for _ in _binding.remote_model_checkpoints if 'refiner' in _.name]  # TODO
            component.value = component.choices[0]
    if gr.Dropdown and getattr(component, 'elem_id', None) == 'txt2img_checkpoint_refresh':
        _binding.txt2img_checkpoint_refresh = component
        if _binding.remote_inference_enabled:
            component.visible = False

    # img2img refiner
    if type(component) is gr.Dropdown and getattr(component, 'elem_id', None) == 'img2img_checkpoint':
        _binding.img2img_checkpoint = component
        _binding.img2img_checkpoint_original = component.get_config()

        if _binding.remote_inference_enabled:
            component.choices = ["None"] + [_.name for _ in _binding.remote_model_checkpoints if 'refiner' in _.name]  # TODO
            component.value = component.choices[0]

    if gr.Dropdown and getattr(component, 'elem_id', None) == 'img2img_checkpoint_refresh':
        _binding.img2img_checkpoint_refresh = component
        if _binding.remote_inference_enabled:
            component.visible = False

    if _binding.txt2img_cloud_inference_checkbox and \
            _binding.img2img_cloud_inference_checkbox and \
            _binding.txt2img_cloud_inference_model_dropdown and \
            _binding.img2img_cloud_inference_model_dropdown and \
            _binding.txt2img_cloud_inference_suggest_prompts_checkbox and \
            _binding.img2img_cloud_inference_suggest_prompts_checkbox and \
            _binding.txt2img_generate and \
            _binding.img2img_generate and \
            _binding.extras_upscaler_1 and \
            _binding.extras_upscaler_2 and \
            _binding.txt2img_hr_upscaler and \
            not _binding.initialized:

        if _binding.ext_controlnet_installed:
            expect_unit_amount = shared.opts.data.get("control_net_max_models_num", 1)
            if expect_unit_amount != len(_binding.txt2img_controlnet_model_dropdown_units):
                return

        if _binding.bultin_refiner_supported:
            if _binding.txt2img_checkpoint is None or _binding.img2img_checkpoint is None:
                return

        sync_cloud_model(_binding.txt2img_cloud_inference_model_dropdown,
                         _binding.img2img_cloud_inference_model_dropdown)
        sync_two_component(_binding.txt2img_cloud_inference_suggest_prompts_checkbox,
                           _binding.img2img_cloud_inference_suggest_prompts_checkbox, 'change')
        on_cloud_inference_checkbox_change(_binding)

        _binding.initialized = True


def build_model_browser_html_for_checkpoint(tab, checkpoints):
    column_html = ""
    column_size = 5
    column_items = [[] for _ in range(column_size)]
    tag_counter = Counter()
    kind = "checkpoint"
    for i, model in enumerate(checkpoints):
        trimed_tags = [_.replace(" ", "_") for _ in model.tags]
        tag_counter.update(trimed_tags)
        if model.preview_url is None or not model.preview_url.startswith("http"):
            model.preview_url = "https://via.placeholder.com/512x512.png?text=Preview+Not+Available"
        model_html = f"""<div class="image-item" data-kind="{kind}" data-tags="{" ".join(trimed_tags)}" data-search-terms="{" ".join(model.search_terms)}">
          <img src="{model.preview_url}" loading="lazy">
          <div class="title-container">
            <div class="title" style="color: var(--checkbox-label-text-color)" data-alias="{model.alias}">{model.name.rsplit(".", 1)[0]}</div>
          </div>
          <div class="overlay">
            <div class="buttons">
            </div>
              <button id="select-button">Select</button>
          </div>
        </div>"""
        column_index = i % column_size
        column_items[column_index].append(model_html)

    for i in range(column_size):
        column_image_items_html = ""
        for item in column_items[i]:
            column_image_items_html += item
        column_html += """<div class="column">{}</div>""".format(column_image_items_html)

    tag_html = f"""<div class="filter-buttons">
                <button class="btn filter-btn" data-tab="{tab}" data-kind="{kind}" data-tag="all">ALL</button>
                """
    tag_html += """{}</div>"""
    tag_html = tag_html.format("\n".join([f"""<button class="btn filter-btn" data-tab="{tab}" data-kind="{kind}" data-tag="{_[0]}">{_[0].upper()}</button>""" for _ in tag_counter.most_common()]))

    return f"""<h1 class="heading-text" id="{kind}-browser">{kind.upper()} Browser</h1>{tag_html}
            <div class="search-bar"><input type="text" id="{tab}-{kind}-filter-search-input" class="filter-search-input" style="color: var(--checkbox-label-text-color); background-color: transparent" placeholder="Search models..."></div>
            <div class="image-gallery">{column_html}</div>"""


def build_model_browser_html_for_loras(tab, loras):
    column_html = ""
    column_size = 5
    column_items = [[] for _ in range(column_size)]
    tag_counter = Counter()
    kind = "lora"
    for i, model in enumerate(loras):
        trimed_tags = [_.replace(" ", "_") for _ in model.tags]
        tag_counter.update(trimed_tags)
        model_html = f"""<div class="image-item" data-kind="{kind}" data-tags="{" ".join(trimed_tags)}" data-search-terms="{" ".join(model.search_terms)}">
          <img src="{model.preview_url}" loading="lazy">
          <div class="title-container">
            <div class="title" style="color: var(--checkbox-label-text-color); background-color: transparent" data-alias="{model.alias}">{model.name.rsplit(".", 1)[0]}</div>
          </div>
          <div class="overlay">
            <div class="buttons">
            </div>
              <button id="select-button">Select</button>
          </div>
        </div>"""
        column_index = i % column_size
        column_items[column_index].append(model_html)

    for i in range(column_size):
        column_image_items_html = ""
        for item in column_items[i]:
            column_image_items_html += item
        column_html += """<div class="column">{}</div>""".format(column_image_items_html)

    tag_html = f"""<div class="filter-buttons">
                <button class="btn filter-btn" data-tab="{tab}" data-kind="{kind}" data-tag="all">ALL</button>
                """
    tag_html += """{}</div>"""
    tag_html = tag_html.format("\n".join([f"""<button class="btn filter-btn" data-tab="{tab}" data-kind="{kind}" data-tag="{_[0]}">{_[0].upper()}</button>""" for _ in tag_counter.most_common()]))

    return f"""<h1 class="heading-text" id="{kind}-browser">{kind.upper()} Browser</h1>{tag_html}<div class="search-bar"><input type="text" id="{tab}-{kind}-filter-search-input" class="filter-search-input" style="color: var(--checkbox-label-text-color); background-color: transparent" placeholder="Search models..."></div><div class="image-gallery">{column_html}</div>"""


def build_model_browser_html_for_embeddings(tab, embeddings):
    column_html = ""
    column_size = 5
    column_items = [[] for _ in range(column_size)]
    tag_counter = Counter()
    kind = "embedding"
    for i, model in enumerate(embeddings):
        trimed_tags = [_.replace(" ", "_") for _ in model.tags]
        tag_counter.update(trimed_tags)
        model_html = f"""<div class="image-item" data-kind="{kind}" data-tags="{" ".join(trimed_tags)}" data-search-terms="{" ".join(model.search_terms)}">
          <img src="{model.preview_url}" loading="lazy">
          <div class="title-container">
            <div class="title" style="color: var(--checkbox-label-text-color)" data-alias="{model.alias}">{model.name.rsplit(".", 1)[0]}</div>
          </div>
          <div class="overlay">
            <div class="buttons">
            </div>
              <button id="select-button">Select</button>
          </div>
        </div>"""
        column_index = i % column_size
        column_items[column_index].append(model_html)

    for i in range(column_size):
        column_image_items_html = ""
        for item in column_items[i]:
            column_image_items_html += item
        column_html += """<div class="column">{}</div>""".format(column_image_items_html)

    tag_html = f"""<div class="filter-buttons">
                <button class="btn filter-btn" data-tab="{tab}" data-kind="{kind}" data-tag="all">ALL</button>
                """
    tag_html += """{}</div>"""
    tag_html = tag_html.format("\n".join([f"""<button class="btn filter-btn" data-tab="{tab}" data-kind="{kind}" data-tag="{_[0]}">{_[0].upper()}</button>""" for _ in tag_counter.most_common()]))

    return f"""<h1 class="heading-text" id="{kind}-browser">{kind.upper()} Browser</h1>{tag_html}<div class="search-bar"><input type="text" id="{tab}-{kind}-filter-search-input" class="filter-search-input" style="color: var(--checkbox-label-text-color)" placeholder="Search models..."></div><div class="image-gallery">{column_html}</div>"""


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
    getattr(a, "change")(fn=mirror, inputs=[a, b], outputs=[a, b])
    getattr(b, "change")(fn=mirror, inputs=[b, a], outputs=[b, a])


def on_cloud_inference_checkbox_change(binding: DataBinding):
    def mirror(source, target):
        enabled = source

        if source != target:
            target = source

        button_text = "Generate"
        if enabled:
            binding.remote_inference_enabled = True
            button_text = "Generate (cloud)"
        else:
            binding.remote_inference_enabled = False

        controlnet_models = ["None"] + [_.name for _ in binding.remote_model_controlnet]
        upscale_models_with_none = ["None"] + [_.alias for _ in binding.remote_model_upscalers]
        upscale_models = [_.alias for _ in binding.remote_model_upscalers]
        refiner_models = ["None"] + [_.name for _ in binding.remote_model_checkpoints if 'refiner' in _.name]  # TODO

        update_components = (
            source,
            target,
            button_text,
            button_text,
        )

        def back_to_original(origin_config):
            allow_update_fields = ['value', 'choices']
            return {k: v for k, v in origin_config.items() if k in allow_update_fields}

        if not enabled:
            update_components += (
                gr.update(**back_to_original(binding.extras_upscaler_1_original)),
                gr.update(**back_to_original(binding.extras_upscaler_2_original)),
                gr.update(**back_to_original(binding.txt2img_hr_upscaler_original))
            )
            if binding.ext_controlnet_installed:
                update_components += (
                    *[gr.update(**back_to_original(_)) for _ in binding.txt2img_controlnet_model_dropdown_original_units],
                    *[gr.update(**back_to_original(_)) for _ in binding.img2img_controlnet_model_dropdown_original_units],
                )
            if binding.bultin_refiner_supported:
                update_components += (
                    gr.update(**back_to_original(binding.txt2img_checkpoint_original)),
                    gr.update(**back_to_original(binding.img2img_checkpoint_original)),
                    gr.update(visible=True),
                    gr.update(visible=True),
                )

            return update_components

        update_components += (
            gr.update(value=upscale_models[0], choices=upscale_models),
            gr.update(value=upscale_models_with_none[0], choices=upscale_models_with_none),
            gr.update(value=upscale_models[0], choices=upscale_models),
        )
        if binding.ext_controlnet_installed:
            update_components += (
                *[gr.update(value=controlnet_models[0], choices=controlnet_models) for _ in binding.txt2img_controlnet_model_dropdown_units],
                *[gr.update(value=controlnet_models[0], choices=controlnet_models) for _ in binding.img2img_controlnet_model_dropdown_units],
            )
        if binding.bultin_refiner_supported:
            update_components += (
                gr.update(value=refiner_models[0], choices=refiner_models),
                gr.update(value=refiner_models[0], choices=refiner_models),
                gr.update(visible=False),
                gr.update(visible=False),
            )

        return update_components

    expect_update_components = (
        _binding.txt2img_generate,
        _binding.img2img_generate,
        _binding.extras_upscaler_1,
        _binding.extras_upscaler_2,
        _binding.txt2img_hr_upscaler
    )
    if _binding.ext_controlnet_installed:
        expect_update_components += (
            *_binding.txt2img_controlnet_model_dropdown_units,
            *_binding.img2img_controlnet_model_dropdown_units,
        )
    if _binding.bultin_refiner_supported:
        expect_update_components += (
            _binding.txt2img_checkpoint,
            _binding.img2img_checkpoint,
            _binding.txt2img_checkpoint_refresh,
            _binding.img2img_checkpoint_refresh,
        )

    _binding.txt2img_cloud_inference_checkbox.change(fn=mirror,
                                                     inputs=[_binding.txt2img_cloud_inference_checkbox,
                                                             _binding.img2img_cloud_inference_checkbox,
                                                             ],
                                                     outputs=[
                                                         _binding.img2img_cloud_inference_checkbox,
                                                         _binding.txt2img_cloud_inference_checkbox,
                                                         *expect_update_components])
    _binding.img2img_cloud_inference_checkbox.change(fn=mirror,
                                                     inputs=[_binding.img2img_cloud_inference_checkbox,
                                                             _binding.txt2img_cloud_inference_checkbox],
                                                     outputs=[
                                                         _binding.img2img_cloud_inference_checkbox,
                                                         _binding.txt2img_cloud_inference_checkbox,
                                                         *expect_update_components
                                                     ])


def on_ui_settings():
    section = ('cloud_inference', "Cloud Inference")
    shared.opts.add_option("cloud_inference_default_enabled", shared.OptionInfo(
        False, "Cloud Inference Default Enabled", component=gr.Checkbox, section=section))
    shared.opts.add_option("cloud_inference_checkbox_hidden", shared.OptionInfo(
        False, "Cloud Inference Checkbox Hideen", component=gr.Checkbox, section=section))
    shared.opts.add_option("cloud_inference_suggest_prompts_default_enabled", shared.OptionInfo(
        True, "Cloud Inference Suggest Prompts Default Enabled", component=gr.Checkbox, section=section))


script_callbacks.on_after_component(on_after_component_callback)
script_callbacks.on_ui_settings(on_ui_settings)
script_callbacks.on_app_started(_hijack_manager.hijack_on_app_started)
