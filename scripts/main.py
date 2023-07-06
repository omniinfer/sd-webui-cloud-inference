import modules.scripts as scripts
import sys
import gradio as gr
import importlib

from modules import images, script_callbacks, processing, ui
from modules import images, script_callbacks
from modules import processing
from modules.processing import Processed, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.shared import opts, state
from extension import api

from inspect import getmembers, isfunction
import random
import traceback


class _Proxy(object):

    def __init__(self, fn):
        self._fn = fn
        self._patched = False

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
        p._remote_model_name = _binding.selected_checkpoint

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            generated_images = api.get_instance().txt2img(p)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            generated_images = api.get_instance().img2img(p)
        else:
            return self._fn(p)

        p.setup_prompts()
        p.all_seeds = [p.seed for _ in range(len(generated_images))]

        index_of_first_image = 0
        unwanted_grid_because_of_img_count = len(
            generated_images) < 2 and opts.grid_only_if_multiple

        comments = {}
        infotexts = []

        def infotext(iteration=0, position_in_batch=0):
            return create_infotext(p, p.all_prompts, p.all_seeds,
                                   p.all_subseeds, comments, iteration,
                                   position_in_batch)

        for image in generated_images:
            if opts.enable_pnginfo:
                image.info["parameters"] = infotext()
                infotexts.append(infotext())

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
        opts.eta_noise_seed_delta if uses_ensd else None,
        "Token merging ratio":
        None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr":
        None if not enable_hr or token_merging_ratio_hr == 0 else
        token_merging_ratio_hr,
        "Init image hash":
        getattr(p, 'init_img_hash', None),
        "RNG":
        opts.randn_source if opts.randn_source != "GPU" else None,
        "NGMS":
        None if p.s_min_uncond == 0 else p.s_min_uncond,
        **p.extra_generation_params,
        "Version":
        processing.program_version() if opts.add_version_to_infotext else None,
    }

    generation_params_text = ", ".join([
        k if k == v else
        f'{k}: {processing.generation_parameters_copypaste.quote(v)}'
        for k, v in generation_params.items() if v is not None
    ])

    negative_prompt_text = f"\nNegative prompt: {p.all_negative_prompts[index]}" if p.all_negative_prompts[
        index] else ""

    return f"{all_prompts[index]}{negative_prompt_text}\n{generation_params_text}".strip(
    )


_proxy = _Proxy(processing.process_images)
_proxy.monkey_patch()
print('Loading extension: sd-webui-cloud-inference')


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
        self.remote_checkpoints = None
        self.remote_checkpoint_dropdown = None
        self.remote_lora_checkbox_group = None
        self.selected_checkpoint = None
        self.cloud_api_dropdown = None
        self.update_suggest_prompts_checkbox = None
        self.suggest_prompts_enabled = True

    def update_suggest_prompts_enabled(self, v):
        print("[cloud-inference] set_suggest_prompts_enabled", v)
        self.suggest_prompts_enabled = v

    def set_remote_inference_enabled(self, v):
        print("set",v)
        return v

    def update_remote_inference_enabled(self, v):
        print("[cloud-infernece] set_remote_inference_enabled", v)
        self.remote_inference_enabled = v

        if v:
            return v,"Generate (cloud)"
        return v,"Generate"

    def update_selected_model(self, name, prompt, neg_prompt):
        print("[cloud-inference] set_selected_model", name)
        self.selected_checkpoint = name

        prompt = prompt
        neg_prompt = neg_prompt

        for ckpt in self.remote_checkpoints:
            if ckpt.name == name:
                if ckpt.example is not None:
                    if ckpt.example.prompts is not None and self.suggest_prompts_enabled:
                        prompt = ckpt.example.prompts
                    if ckpt.example.neg_prompt is not None and self.suggest_prompts_enabled:
                        neg_prompt = ckpt.example.neg_prompt
                return gr.update(choices=ckpt.loras), gr.update(
                    value=prompt), gr.update(value=neg_prompt)
        return gr.update(choices=[]), gr.update(value=prompt), gr.update(
            value=neg_prompt)

    def update_selected_lora(self, lora_names, prompt):
        print("[cloud-inference] set_selected_lora")
        # return prompt + ", " + lora_name

        add_lora_prompts = []

        for lora in lora_names:
            if '<lora:{}:'.format(lora) not in prompt:
                add_lora_prompts.append("<lora:{}:1>".format(lora))

        if len(add_lora_prompts) > 0 and (not prompt.endswith(", ")
                                          and not prompt.endswith(",")):
            prompt = prompt + ", "

        return gr.update(
            value="{}{}".format(prompt, ", ".join(add_lora_prompts)))

    def update_cloud_api(self, v):
        # TODO: support multiple cloud api provider
        print("[cloud-inference] set_cloud_api", v)
        self.cloud_api = v


class CloudInferenceScript(scripts.Script):
    # Extension title in menu UI
    def title(self):
        return "Cloud Inference"

    def show(self, is_img2img):
        return scripts.AlwaysVisible

    def event_handler(value):
        
        v = "Generate"
        if value:
            v= "Generate (cloud)"
        
        print(v)
        
        _binding.img2img_generate.update(v)
        _binding.txt2img_generate.update(v)
    
    def ui(self, is_img2img):
        with gr.Accordion('Cloud Inference', open=True):
            with gr.Row():
                
                if _binding.enable_remote_inference is None:
                    _binding.enable_remote_inference = gr.Checkbox(
                    value=False,
                    label="Enable",
                    elem_id="enable_remote_inference")

                
                if is_img2img:
                    _binding.img2img_enable_remote_inference = gr.Checkbox(
                        value=False,
                        label="Enable",
                        elem_id="img2img_enable_remote_inference")
                    
                    
                    _binding.enable_remote_inference.change(
                        fn=lambda x: _binding.update_remote_inference_enabled(x),
                        inputs=[_binding.enable_remote_inference],
                        outputs=[_binding.img2img_enable_remote_inference,_binding.img2img_generate]
                    )
                    
                    _binding.img2img_enable_remote_inference.change(
                        fn=lambda x: _binding.set_remote_inference_enabled(x),
                        inputs=[_binding.img2img_enable_remote_inference],
                        outputs=[_binding.enable_remote_inference]
                    )

                else:
                    _binding.txt2img_enable_remote_inference = gr.Checkbox(
                        value=False,
                        label="Enable",
                        elem_id="txt2img_enable_remote_inference")
                    
                    _binding.enable_remote_inference.change(
                        fn=lambda x: _binding.update_remote_inference_enabled(x),
                        inputs=[_binding.enable_remote_inference],
                        outputs=[_binding.txt2img_enable_remote_inference,_binding.txt2img_generate]
                    )
                    
                    _binding.txt2img_enable_remote_inference.change(
                        fn=lambda x: _binding.set_remote_inference_enabled(x),
                        inputs=[_binding.txt2img_enable_remote_inference],
                        outputs=[_binding.enable_remote_inference]
                    )


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
                try:
                    _binding.remote_checkpoints = api.get_instance(
                    ).list_models()
                    if _binding.remote_checkpoints is None or len(
                            _binding.remote_checkpoints) == 0:
                        api.get_instance().refresh_models()
                        _binding.remote_checkpoints = api.get_instance(
                        ).list_models()

                    _checkpoint_choices = [
                        m.name for m in _binding.remote_checkpoints
                    ]
                except Exception as e:
                    print(traceback.format_exc())
                    _checkpoint_choices = []

                _binding.selected_checkpoint = _checkpoint_choices[0] if len(
                    _checkpoint_choices) > 0 else None
                print("[cloud-inference] default checkpoint {}".format(
                    _binding.selected_checkpoint))

                _binding.remote_checkpoint_dropdown = gr.Dropdown(
                    label="Checkpoint",
                    choices=_checkpoint_choices,
                    value=_binding.selected_checkpoint,
                    elem_id="remote_checkpoint_dropdown")

                ui.create_refresh_button(
                    _binding.remote_checkpoint_dropdown,
                    api.get_instance().refresh_models, lambda: {
                        "choices":
                        [_.name for _ in api.get_instance().list_models()]
                    }, "Refresh")

                with gr.Column():
                    # remote lora
                    _binding.remote_lora_checkbox_group = gr.CheckboxGroup(
                        choices=[],
                        label="Lora",
                        elem_id="remote_lora_dropdown")

                    _binding.remote_checkpoint_dropdown.select(
                        fn=_binding.update_selected_model,
                        inputs=[
                            _binding.remote_checkpoint_dropdown,
                            _binding.txt2img_prompt,
                            _binding.txt2img_neg_prompt
                        ],
                        outputs=[
                            _binding.remote_lora_checkbox_group,
                            _binding.txt2img_prompt,
                            _binding.txt2img_neg_prompt
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
            _binding.remote_checkpoint_dropdown,
            enable_remote_inference,
            _binding.remote_lora_checkbox_group,
        ]


_binding = None
if _binding is None:
    _binding = DataBinding()



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

    # is_txt2img_gallery = type(component) is gr.Gallery and getattr(
    #     component, 'elem_id', None) == 'txt2img_gallery'
    # is_txt2img_generation_info = type(component) is gr.Textbox and getattr(
    #     component, 'elem_id', None) == 'generation_info_txt2img'
    # is_txt2img_html_info = type(component) is gr.HTML and getattr(
    #     component, 'elem_id', None) == 'html_info_txt2img'

    # is_global_checkpoint_dropdown = type(component) == gr.Dropdown and getattr(
    #     component, "elem_id", None) == "setting_sd_model_checkpoint"
    # is_remote_checkpoint_list = type(component) == gr.Dropdown and getattr(
    #     component, "elem_id", None) == "remote_checkpoint_list"
    # is_txt2img_batch_size = type(component) == gr.Slider and getattr(
    #     component, "elem_id", None) == "txt2img_batch_size"
    # is_txt2img_batch_count = type(component) == gr.Slider and getattr(
    #     component, "elem_id", None) == "txt2img_batch_count"
    # is_txt2img_sampling = type(component) == gr.Dropdown and getattr(
    #     component, "elem_id", None) == "txt2img_sampling"
    # is_txt2img_steps = type(component) == gr.Slider and getattr(
    #     component, "elem_id", None) == "txt2img_steps"
    # is_txt2img_cfg_scale = type(component) == gr.Slider and getattr(
    #     component, "elem_id", None) == "txt2img_cfg_scale"
    # is_txt2img_seed = type(component) == gr.Number and getattr(
    #     component, "elem_id", None) == "txt2img_seed"
    # is_txt2img_height = type(component) == gr.Slider and getattr(
    #     component, "elem_id", None) == "txt2img_height"
    # is_txt2img_width = type(component) == gr.Slider and getattr(
    #     component, "elem_id", None) == "txt2img_width"

    # if is_txt2img_prompt:
    #     _binding.txt2img_prompt = component
    # if is_txt2img_neg_prompt:
    #     _binding.txt2img_neg_prompt = component
    # if is_txt2img_gallery:
    #     _binding.txt2img_gallery = component
    # if is_remote_checkpoint_list:
    #     _binding.remote_models = component
    # if is_txt2img_batch_size:
    #     _binding.txt2img_batch_size = component
    # if is_txt2img_batch_count:
    #     _binding.txt2img_batch_count = component
    # if is_txt2img_sampling:
    #     _binding.txt2img_sampling = component
    # if is_txt2img_steps:
    #     _binding.txt2img_steps = component
    # if is_txt2img_cfg_scale:
    #     _binding.txt2img_cfg_scale = component
    # if is_txt2img_seed:
    #     _binding.txt2img_seed = component
    # if is_txt2img_height:
    #     _binding.txt2img_height = component
    # if is_txt2img_width:
    #     _binding.txt2img_width = component

    # # print(component, _kwargs)
    # # if txt2img_gallery is not None and not button_bound:
    # if _binding.txt2img_gallery is not None \
    #     and _binding.txt2img_prompt is not None \
    #     and _binding.txt2img_neg_prompt is not None \
    #     and _binding.remote_models is not None \
    #     and _binding.txt2img_sampling is not None \
    #     and _binding.txt2img_batch_size is not None \
    #     and _binding.txt2img_batch_count is not None \
    #     and _binding.txt2img_steps is not None \
    #     and _binding.txt2img_cfg_scale is not None \
    #     and _binding.txt2img_seed is not None \
    #     and _binding.txt2img_height is not None \
    #     and _binding.txt2img_width is not None \
    #     and not _binding.button_bound:

    #     # print(sd_prompt)
    #     _binding.generate_button.click(
    #         # fn=_api.get_instance().txt2img,
    #         _download_wrapper(
    #             api.get_instance().txt2img, "/stable-diffusion-webui/outputs/{}".format(
    #                 datetime.datetime.now().strftime("%Y-%m-%d"))),
    #         inputs=[
    #             _binding.remote_models, _binding.txt2img_prompt,
    #             _binding.txt2img_neg_prompt, _binding.txt2img_sampling,
    #             _binding.txt2img_batch_size, _binding.txt2img_steps,
    #             _binding.txt2img_batch_count, _binding.txt2img_cfg_scale,
    #             _binding.txt2img_seed, _binding.txt2img_height,
    #             _binding.txt2img_width
    #         ],
    #         outputs=[_binding.txt2img_gallery])
    #     _binding.button_bound = True


script_callbacks.on_after_component(on_after_component_callback)
