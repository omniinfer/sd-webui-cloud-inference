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


class _Proxy(object):

    def __init__(self, fn):
        self._fn = fn
        self._patched = False
        self._binding = None

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

            def xyz_model_apply(p: StableDiffusionProcessing, opt, v):
                m = self._binding.choice_to_model(opt)
                if m.kind == 'lora':
                    p._remote_model_name = m.dependency_model_name
                    p.prompt = self._binding._update_lora_in_prompt(
                        p.prompt, m.name)
                else:
                    p._remote_model_name = m.name

            def xyz_model_confirm(p, opt):
                return

            def xyz_model_format(p, opt, v):
                return self._binding.choice_to_model(v).name.rsplit(".", 1)[0]

            xyz_grid.axis_options.append(
                xyz_grid.AxisOption('[Cloud Inference] Model Name',
                                    str,
                                    apply=xyz_model_apply,
                                    confirm=xyz_model_confirm,
                                    format_value=xyz_model_format,
                                    choices=self._binding.get_model_ckpt_choices))

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

        if len(args) > 0 and isinstance(args[0],
                                        processing.StableDiffusionProcessing):
            p = args[0]
        else:
            raise Exception(
                'process_images: first argument must be a processing object')

        remote_inference_enabled, selected_model_index = get_visible_extension_args(p, 'cloud inference')
        if not remote_inference_enabled:
            return self._fn(*args, **kwargs)

        # random seed locally if not specified
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.begin()
        state.sampling_steps = p.steps
        state.job_count = p.n_iter

        state.textinfo = "remote inferencing ({})".format(
            api.get_instance().__class__.__name__)
        if not getattr(p, '_remote_model_name', None):  # xyz_grid
            p._remote_model_name = self._binding.remote_sd_models[selected_model_index].name

        if isinstance(p, StableDiffusionProcessingTxt2Img):
            generated_images = api.get_instance().txt2img(p)
        elif isinstance(p, StableDiffusionProcessingImg2Img):
            generated_images = api.get_instance().img2img(p)
        else:
            return self._fn(*args, **kwargs)

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
                prompt = p.all_prompts[i]

            if opts.samples_save and not p.do_not_save_samples:
                images.save_image(image,
                                  p.outpath_samples,
                                  "",
                                  seed,
                                  prompt,
                                  opts.samples_format,
                                  info=infotext(),
                                  p=p)
        if (opts.return_grid or opts.grid_save) and not p.do_not_save_grid and not unwanted_grid_because_of_img_count:
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
        "Steps": p.steps,
        "Sampler": p.sampler_name,
        "CFG scale": p.cfg_scale,
        "Image CFG scale": getattr(p, 'image_cfg_scale', None),
        "Seed": all_seeds[index],
        "Face restoration": (opts.face_restoration_model if p.restore_faces else None),
        "Size": f"{p.width}x{p.height}",
        "Model": (None if not opts.add_model_name_to_info or not p._remote_model_name else p._remote_model_name.replace(',', '').replace(':', '')),
        "Variation seed": (None if p.subseed_strength == 0 else all_subseeds[index]),
        "Variation seed strength": (None if p.subseed_strength == 0 else p.subseed_strength),
        "Seed resize from": (None if p.seed_resize_from_w == 0 or p.seed_resize_from_h == 0 else f"{p.seed_resize_from_w}x{p.seed_resize_from_h}"),
        "Denoising strength": getattr(p, 'denoising_strength', None),
        "Conditional mask weight": getattr(p, "inpainting_mask_weight", opts.inpainting_mask_weight) if p.is_using_inpainting_conditioning else None,
        "Clip skip": None if clip_skip <= 1 else clip_skip,
        "ENSD": getattr(opts, 'eta_noise_seed_delta', None) if uses_ensd else None,
        "Token merging ratio": None if token_merging_ratio == 0 else token_merging_ratio,
        "Token merging ratio hr": None if not enable_hr or token_merging_ratio_hr == 0 else token_merging_ratio_hr,
        "Init image hash": getattr(p, 'init_img_hash', None),
        "RNG": None,
        "NGMS": None,
        "Version": None,
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


def get_visible_extension_args(p: processing.StableDiffusionProcessing, name):
    for s in p.scripts.alwayson_scripts:
        if s.name == name:
            return p.script_args[s.args_from:s.args_to]
    return []


_proxy = _Proxy(processing.process_images)
