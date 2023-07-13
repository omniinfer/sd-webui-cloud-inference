import modules.scripts as scripts
import sys
import gradio as gr
import importlib

from modules import images, script_callbacks, processing, ui
from modules import images, script_callbacks
from modules import processing, shared
from modules.processing import Processed, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state, prompt_styles
from extension import api

from inspect import getmembers, isfunction
import random
import traceback


class _Proxy(object):

    def __init__(self, fn):
        self._fn = fn
        self._patched = False

    def _apply_xyz(self):

        def find_module(module_names):
            if isinstance(module_names, str):
                module_names = [s.strip() for s in module_names.split(",")]
            for data in scripts.scripts_data:
                if data.script_class.__module__ in module_names and hasattr(
                        data, "module"):
                    return data.module
            return None

        xyz_grid = find_module("xyz_grid.py, xy_grid.py")
        if xyz_grid:

            def xyz_model_apply(p, opt, v):
                m = _binding.choice_to_model(opt)
                if m.kind == 'lora':
                    p._remote_model_name = m.dependency_model_name
                    p.prompt = _binding._add_lora_in_prompt(p.prompt, m.name)
                else:
                    p._remote_model_name = m.name

            def xyz_model_confirm(p, opt):
                return

            def xyz_model_format(p, opt, v):
                return _binding.choice_to_model(v).name.rsplit(".", 1)[0]

            xyz_grid.axis_options.append(
                xyz_grid.AxisOption('[Cloud Inference] Model Name',
                                    str,
                                    apply=xyz_model_apply,
                                    confirm=xyz_model_confirm,
                                    format_value=xyz_model_format,
                                    choices=_binding.get_model_choices))

    def monkey_patch(self):
        if self._patched:
            return

        processing.process_images = self

        keys = list(sys.modules.keys())
        for name in keys:
            # if (name.startswith('modules')
            # or name.startswith('scripts')) and name != 'modules.processing':
            if name.startswith('modules') and name != 'modules.processing':
                if 'process_images' in dict(
                        getmembers(sys.modules[name], isfunction)).keys():
                    print('[cloud-inference] reloading', name)
                    importlib.reload(sys.modules[name])

        from modules.scripts import scripts_data
        for script in scripts_data:
            if hasattr(script.module, 'process_images'):
                script.module.process_images = self
            if hasattr(
                    script.module, 'processing'
            ) and script.module.processing.__name__ == 'modules.processing':
                script.module.processing.process_images = self

        self._apply_xyz()
        print('[cloud-inference] monkey patched')

        self._patched = True

    def __call__(self, *args, **kwargs) -> Processed:
        if not _binding.remote_inference_enabled:
            return self._fn(*args, **kwargs)

        if len(args) > 0 and isinstance(args[0],
                                        processing.StableDiffusionProcessing):
            p = args[0]
        else:
            raise Exception(
                'process_images: first argument must be a processing object')

        # random seed locally if not specified
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.begin()
        state.sampling_steps = p.steps
        state.job_count = p.n_iter

        state.textinfo = "remote inferencing ({})".format(
            api.get_instance().__class__.__name__)
        if not getattr(p, '_remote_model_name', None):  # xyz_grid
            p._remote_model_name = _binding.selected_checkpoint.name

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            generated_images = api.get_instance().txt2img(p)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            generated_images = api.get_instance().img2img(p)
        else:
            return self._fn(p)

        # compatible with old version
        if hasattr(p, 'setup_prompts'):
            p.setup_prompts()
        else:
            if type(p.prompt) == list:
                p.all_prompts = p.prompt
            else:
                p.all_prompts = p.batch_size * p.n_iter * [p.prompt]
            if type(p.negative_prompt) == list:
                p.all_negative_prompts = p.negative_prompt
            else:
                p.all_negative_prompts = p.batch_size * p.n_iter * [
                    p.negative_prompt
                ]
            p.all_prompts = [
                prompt_styles.apply_styles_to_prompt(x, p.styles)
                for x in p.all_prompts
            ]
            p.all_negative_prompts = [
                prompt_styles.apply_negative_styles_to_prompt(x, p.styles)
                for x in p.all_negative_prompts
            ]

            # TODO: img2img hr prompts

        p.all_seeds = [p.seed for _ in range(len(generated_images))]
        p.seeds = p.all_seeds

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(
            generated_images) < 2 and opts.grid_only_if_multiple

        comments = {}
        infotexts = []

        def infotext(iteration=0, position_in_batch=0):
            return create_infotext(p, p.all_prompts, p.all_seeds,
                                   p.all_subseeds, comments, iteration,
                                   position_in_batch)

        for i, image in enumerate(generated_images):
            if opts.enable_pnginfo:
                image.info["parameters"] = infotext()
                infotexts.append(infotext())

            seed = None
            if len(p.all_seeds) > i:
                seed = p.all_seeds[i]
            prompt = None
            if len(p.all_prompts) > i:
                prompt = p.all_seeds[i]

            if opts.samples_save and not p.do_not_save_samples:
                images.save_image(image,
                                  p.outpath_samples,
                                  "",
                                  seed,
                                  prompt,
                                  opts.samples_format,
                                  info=infotext(),
                                  p=p)
        if (
                opts.return_grid or opts.grid_save
        ) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
            grid = images.image_grid(generated_images, p.batch_size)

            if opts.return_grid:
                text = infotext()
                infotexts.insert(0, text)
                if opts.enable_pnginfo:
                    grid.info["parameters"] = text

                generated_images.insert(0, grid)
                index_of_first_image = 1

            if opts.grid_save:
                images.save_image(
                    grid,
                    p.outpath_grids,
                    "grid",
                    p.all_seeds[0],
                    p.all_prompts[0],
                    opts.grid_format,
                    info=infotext(),
                    short_filename=not opts.grid_extended_filename,
                    p=p,
                    grid=True)
        p = Processed(
            p,
            generated_images,
            all_seeds=[p.seed for _ in range(len(generated_images))],
            all_prompts=[p.prompt for _ in range(len(generated_images))],
            comments="".join(f"{comment}\n" for comment in comments),
            index_of_first_image=index_of_first_image,
            infotexts=infotexts)
        state.end()
        return p


def create_infotext(p,
                    all_prompts,
                    all_seeds,
                    all_subseeds,
                    comments=None,
                    iteration=0,
                    position_in_batch=0):
    index = position_in_batch + iteration * p.batch_size

    clip_skip = getattr(p, 'clip_skip', opts.CLIP_stop_at_last_layers)
    enable_hr = getattr(p, 'enable_hr', False)

    # compatible with old version
    token_merging_ratio = None
    token_merging_ratio_hr = None
    if hasattr(p, 'get_token_merging_ratio'):
        token_merging_ratio = p.get_token_merging_ratio()
        token_merging_ratio_hr = p.get_token_merging_ratio(for_hr=True)

    uses_ensd = opts.eta_noise_seed_delta != 0
    if uses_ensd:
        uses_ensd = processing.sd_samplers_common.is_sampler_using_eta_noise_seed_delta(
            p)

    generation_params = {
        "Steps":
        p.steps,
        "Sampler":
        p.sampler_name,
        "CFG scale":
        p.cfg_scale,
        "Image CFG scale":
        getattr(p, 'image_cfg_scale', None),
        "Seed":
        all_seeds[index],
        "Face restoration":
        (opts.face_restoration_model if p.restore_faces else None),
        "Size":
        f"{p.width}x{p.height}",
        "Model":
        (None if not opts.add_model_name_to_info or not p._remote_model_name
         else p._remote_model_name.replace(',', '').replace(':', '')),
        "Variation seed":
        (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength":
        (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from":
        (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else
         f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength":
        getattr(p, 'denoising_strength', None),
        "Conditional mask weight":
        getattr(p, "inpainting_mask_weight", opts.inpainting_mask_weight)
        if p.is_using_inpainting_conditioning else None,
        "Clip skip":
        None if clip_skip <= 1 else clip_skip,
        "ENSD":
        getattr(opts, 'eta_noise_seed_delta', None) if uses_ensd else None,
        "Token merging ratio":
        None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr":
        None if not enable_hr or token_merging_ratio_hr == 0 else
        token_merging_ratio_hr,
        "Init image hash":
        getattr(p, 'init_img_hash', None),
        "RNG":
        None,
        "NGMS":
        None,
        "Version":
        None,
        **p.extra_generation_params,
    }

    # compatible with old version
    if getattr(p, 's_min_ucond', None) is not None:
        if p.s_min_uncond != 0:
            generation_params["NGMS"] = p.s_min_uncond
    if getattr(opts, 'randn_source',
               None) is not None and opts.randn_source != "GPU":
        generation_params["RNG"] = opts.randn_source

    if getattr(opts, 'add_version_to_infotext', None):
        if opts.add_version_to_infotext:
            generation_params['Version'] = processing.program_version()

    generation_params_text = ", ".join([
        k if k == v else
        f'{k}: {processing.generation_parameters_copypaste.quote(v)}'
        for k, v in generation_params.items() if v is not None
    ])

    negative_prompt_text = f"\nNegative prompt: {p.all_negative_prompts[index]}" if p.all_negative_prompts[
        index] else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip(
    )


class DataBinding:

    def __init__(self):
        self.enable_remote_inference = None
        self.txt2img_enable_remote_inference = None
        self.img2img_enable_remote_inference = None

        self.remote_inference_enabled = False
        self.txt2img_prompt = None
        self.txt2img_neg_prompt = None
        self.txt2img_generate = None
        self.img2img_generate = None
        self.remote_sd_models = None
        self.remote_model_dropdown = None
        self.remote_lora_checkbox_group = None
        self.selected_checkpoint = None  # checkpoint object
        self.cloud_api_dropdown = None
        self.update_suggest_prompts_checkbox = None
        self.suggest_prompts_enabled = True

        self.initialized = False

    def update_suggest_prompts_enabled(self, v):
        print("[cloud-inference] set_suggest_prompts_enabled", v)
        self.suggest_prompts_enabled = v

    def set_remote_inference_enabled(self, v):
        print("[cloud-inference] set_remote_inference_enabled", v)
        return v

    def update_remote_inference_enabled(self, v):
        print("[cloud-infernece] set_remote_inference_enabled", v)
        self.remote_inference_enabled = v

        if v:
            return v, "Generate (cloud)"
        return v, "Generate"

    def update_selected_model(self, name_index: int, prompt: str,
                              neg_prompt: str):
        selected: api.StableDiffusionModel = self.remote_sd_models[name_index]
        selected_checkpoint: api.StableDiffusionModel = None
        selected_checkpoint_index: int = 0
        selected_loras = []

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

        self.selected_checkpoint = selected_checkpoint

        name = self.remote_sd_models[name_index].name
        print("[cloud-inference] set_selected_model", name)

        prompt = prompt
        neg_prompt = neg_prompt

        if selected.example is not None:
            if selected.example.prompts is not None and self.suggest_prompts_enabled:
                prompt = selected.example.prompts
                prompt = prompt.replace("\n", "")
                if len(selected_loras) > 0:
                    prompt = self._add_lora_in_prompt(selected.example.prompts,
                                                      selected_loras)
                prompt = prompt.replace("\n", "")
            if selected.example.neg_prompt is not None and self.suggest_prompts_enabled:
                neg_prompt = selected.example.neg_prompt

        return gr.Dropdown.update(
            choices=[_.display_name for _ in self.remote_sd_models],
            value=selected_checkpoint.display_name), gr.update(
                choices=[_ for _ in selected_checkpoint.child],
                value=selected_loras), gr.update(value=prompt), gr.update(
                    value=neg_prompt)

    @staticmethod
    def _add_lora_in_prompt(prompt, lora_names, weight=1):
        prompt = prompt
        add_lora_prompts = []

        for lora_name in lora_names:
            if '<lora:{}:'.format(lora_name) not in prompt:
                add_lora_prompts.append("<lora:{}:{}>".format(
                    lora_name, weight))

        if len(add_lora_prompts) > 0 and (not prompt.endswith(", ")
                                          and not prompt.endswith(",")):
            prompt = prompt + ", "

        return prompt + ", ".join(add_lora_prompts)

    @staticmethod
    def _del_lora_in_prompt(self, prompt, lora_name):
        pass

    def update_selected_lora(self, lora_names, prompt):
        print("[cloud-inference] set_selected_lora", lora_names)
        return gr.update(value=self._add_lora_in_prompt(prompt, lora_names))

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

    def choice_to_model(self, choice):
        for model in self.remote_sd_models:
            if model.display_name == choice:
                return model

    def get_model_choices(self):
        return [_.display_name for _ in self.remote_sd_models]


class CloudInferenceScript(scripts.Script):
    # Extension title in menu UI
    def title(self):
        return "Cloud Inference"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def event_handler(value):
        v = "Generate"
        if value:
            v = "Generate (cloud)"

        _binding.img2img_generate.update(v)
        _binding.txt2img_generate.update(v)

    def ui(self, is_img2img):
        with gr.Accordion('Cloud Inference', open=True):
            with gr.Row():

                if _binding.enable_remote_inference is None:
                    _binding.enable_remote_inference = gr.Checkbox(
                        value=False,
                        visible=False,
                        label="Enable",
                        elem_id="enable_remote_inference")

                if is_img2img:
                    _binding.img2img_enable_remote_inference = gr.Checkbox(
                        value=lambda: _binding.remote_inference_enabled,
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
                        fn=lambda x: _binding.set_remote_inference_enabled(x),
                        inputs=[_binding.img2img_enable_remote_inference],
                        outputs=[_binding.enable_remote_inference])

                else:
                    _binding.txt2img_enable_remote_inference = gr.Checkbox(
                        value=lambda: _binding.remote_inference_enabled,
                        label="Enable",
                        elem_id="txt2img_enable_remote_inference")

                    _binding.enable_remote_inference.change(
                        fn=lambda x: _binding.update_remote_inference_enabled(
                            x),
                        inputs=[_binding.enable_remote_inference],
                        outputs=[
                            _binding.txt2img_enable_remote_inference,
                            _binding.txt2img_generate
                        ])

                    _binding.txt2img_enable_remote_inference.change(
                        fn=lambda x: _binding.set_remote_inference_enabled(x),
                        inputs=[_binding.txt2img_enable_remote_inference],
                        outputs=[_binding.enable_remote_inference])

                _binding.enable_suggest_prompts_checkbox = gr.Checkbox(
                    value=_binding.suggest_prompts_enabled,
                    label="Suggest Prompts",
                    elem_id="enable_suggest_prompts_checkbox")
                _binding.enable_suggest_prompts_checkbox.change(
                    fn=lambda x: _binding.update_suggest_prompts_enabled(x),
                    inputs=[_binding.enable_suggest_prompts_checkbox],
                )

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
                        _binding.remote_sd_models = api.get_instance(
                        ).list_models()
                        if _binding.remote_sd_models is None or len(
                                _binding.remote_sd_models) == 0:
                            api.get_instance().refresh_models()
                            _binding.remote_sd_models = api.get_instance(
                            ).list_models()

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
                    label="Cloud Models (ckpt/lora)",
                    choices=_binding.get_model_choices(),
                    value=lambda: _binding.selected_checkpoint.display_name,
                    type="index",
                    elem_id="remote_model_dropdown")

                def _refresh():
                    _binding.remote_sd_models = api.get_instance().list_models(
                    )
                    return {
                        "choices":
                        [_.display_name for _ in _binding.remote_sd_models]
                    }

                ui.create_refresh_button(_binding.remote_model_dropdown,
                                         api.get_instance().refresh_models,
                                         _refresh, "Refresh")

            with gr.Column():
                # remote lora
                _binding.remote_lora_checkbox_group = gr.CheckboxGroup(
                    _binding.get_selected_model_loras(),
                    label="Lora",
                    elem_id="remote_lora_dropdown")

                _binding.remote_model_dropdown.select(
                    fn=_binding.update_selected_model,
                    inputs=[
                        _binding.remote_model_dropdown,
                        _binding.txt2img_prompt, _binding.txt2img_neg_prompt
                    ],
                    outputs=[
                        _binding.remote_model_dropdown,
                        _binding.remote_lora_checkbox_group,
                        _binding.txt2img_prompt, _binding.txt2img_neg_prompt
                    ])
                _binding.remote_lora_checkbox_group.select(
                    fn=lambda x, y: _binding.update_selected_lora(x, y),
                    inputs=[
                        _binding.remote_lora_checkbox_group,
                        _binding.txt2img_prompt
                    ],
                    outputs=_binding.txt2img_prompt,
                )

        enable_remote_inference = None
        if is_img2img:
            enable_remote_inference = _binding.img2img_enable_remote_inference
        else:
            enable_remote_inference = _binding.txt2img_enable_remote_inference

        return [
            _binding.remote_model_dropdown,
            enable_remote_inference,
            _binding.remote_lora_checkbox_group,
        ]


_binding = None
if _binding is None:
    _binding = DataBinding()

_proxy = _Proxy(processing.process_images)
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
