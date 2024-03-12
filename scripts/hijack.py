import modules.scripts as scripts
import os
import sys
import importlib

from modules import images, script_callbacks, errors, processing, ui, shared, scripts_postprocessing, ui_common
try:
    from modules import generation_parameters_copypaste
except ImportError:
    pass
    from modules import infotext_utils as generation_parameters_copypaste

from modules.processing import Processed, StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img, StableDiffusionProcessing
from modules.shared import opts, state, prompt_styles
from extension import api

from inspect import getmembers, isfunction, ismodule
import random

from PIL import Image


class _HijackManager(object):

    def __init__(self):
        self._hijacked_onload = False
        self._hijacked_on_app_started = False
        self._binding = None

        self.hijack_map = {}

    def hijack_one(self, name, new_fn):
        tmp = name.rsplit('.', 1)
        if len(tmp) < 2:
            raise Exception('invalid module.func name: {}'.format(name))

        module_name, func_name = tmp
        old_fn = _hijack_func(module_name, func_name, new_fn)
        if old_fn is None:
            print('[cloud-inference] hijack failed: {}'.format(name))
            return False

        self.hijack_map[name] = {
            'old': old_fn,
            'new': new_fn,
        }

        print('[cloud-inference] hijack {}, old: <{}>, new: <{}>'.format(
            name, old_fn.__module__ + '.' + old_fn.__name__, new_fn.__module__ + '.' + new_fn.__name__))

    def hijack_onload(self):
        if self._hijacked_onload:
            return
        self.hijack_one('modules.processing.process_images', self._hijack_process_images)
        self.hijack_one('modules.postprocessing.run_postprocessing', self._hijack_run_postprocessing)
        self._apply_xyz()
        self._hijacked_onload = True

    def hijack_on_app_started(self, *args, **kwargs):
        if self._hijacked_on_app_started:
            return

        self.hijack_one('extensions.sd-webui-controlnet.scripts.global_state.update_cn_models', self._hijack_update_cn_models)
        self._hijack_update_cn_models()  # update once

        self._hijacked_on_app_started = True

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
            def xyz_checkpoint_apply(p: StableDiffusionProcessing, opt, v):
                if '_cloud_inference_settings' not in p.__dict__:
                    p._cloud_inference_settings = {}

                m = self._binding.find_model_by_alias(opt)
                p._cloud_inference_settings['sd_checkpoint'] = m.name

            def xyz_checkpoint_confirm(p, opt):
                return

            def xyz_checkpoint_format(p, opt, v):
                return self._binding.find_model_by_alias(v).name.rsplit(".", 1)[0]

            def xyz_vae_apply(p: StableDiffusionProcessing, opt, v):
                if '_cloud_inference_settings' not in p.__dict__:
                    p._cloud_inference_settings = {}

                p._cloud_inference_settings['sd_vae'] = opt

            def xyz_vae_confirm(p, opt):
                return

            def xyz_vae_format(p, opt, v):
                return v

            print('[cloud-inference] hijack xyz_grid')
            xyz_grid.axis_options.append(
                xyz_grid.AxisOption('[Cloud Inference] Checkpoint',
                                    str,
                                    apply=xyz_checkpoint_apply,
                                    confirm=xyz_checkpoint_confirm,
                                    format_value=xyz_checkpoint_format,
                                    choices=lambda: [_.alias for _ in self._binding.remote_model_checkpoints]))
            xyz_grid.axis_options.append(
                xyz_grid.AxisOption('[Cloud Inference] VAE',
                                    str,
                                    apply=xyz_vae_apply,
                                    confirm=xyz_vae_confirm,
                                    format_value=xyz_vae_format,
                                    choices=lambda: ["Automatic", "None"] + [_.name for _ in self._binding.remote_model_vaes]))
        self._xyz_hijacked = True

    def _hijack_update_cn_models(self):
        from modules.scripts import scripts_data
        for script in scripts_data:
            if script.module.__name__ == 'controlnet.py':
                if self._binding.remote_inference_enabled:
                    script.module.global_state.cn_models.clear()
                    cn_models_keys = ["None"] + [_.name for _ in self._binding.remote_model_controlnet]
                    cn_models_dict = {k: None for k in cn_models_keys}

                    script.module.global_state.cn_models.update(cn_models_dict)
                    script.module.global_state.cn_models_names.clear()
                    script.module.global_state.cn_models_names.update(cn_models_dict)
                    break
                else:
                    self.hijack_map['extensions.sd-webui-controlnet.scripts.global_state.update_cn_models']['old']()

    def _hijack_process_images(self, *args, **kwargs) -> Processed:
        if len(args) > 0 and isinstance(args[0],
                                        processing.StableDiffusionProcessing):
            p = args[0]
        else:
            raise Exception(
                'process_images: first argument must be a processing object')

        remote_inference_enabled, selected_checkpoint_name, selected_vae_name = get_visible_extension_args(p, 'cloud inference')

        if not remote_inference_enabled:
            return self.hijack_map['modules.processing.process_images']['old'](*args, **kwargs)

        # random seed locally if not specified
        if p.seed == -1:
            p.seed = int(random.randrange(4294967294))

        state.begin()
        state.sampling_steps = p.steps
        state.job_count = p.n_iter

        state.textinfo = "remote inferencing ({})".format(api.get_instance().__class__.__name__)

        if '_cloud_inference_settings' not in p.__dict__:
            p._cloud_inference_settings = {}

        if 'sd_checkpoint' not in p._cloud_inference_settings:
            p._cloud_inference_settings['sd_checkpoint'] = self._binding.find_model_by_alias(selected_checkpoint_name).name
        if 'sd_vae' not in p._cloud_inference_settings:
            p._cloud_inference_settings['sd_vae'] = selected_vae_name

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

    def _hijack_run_postprocessing(self, *args, **kwargs):
        if not self._binding.remote_inference_enabled:
            return self.hijack_map['modules.postprocessing.run_postprocessing']['old'](*args, **kwargs)

        shared.state.begin()
        shared.state.textinfo = "remote inferencing ({})".format(api.get_instance().__class__.__name__)

        image_data = []
        image_names = []
        outputs = []

        # extras_mode, image, image_folder, input_dir, output_dir, show_extras_results, *args, save_output: bool = True
        extras_mode = args[0]
        image = args[1]
        image_folder = args[2]
        input_dir = args[3]
        output_dir = args[4]
        show_extras_results = args[5]
        if len(args) > 6:
            args = args[6:]
        save_output = kwargs.get('save_output', True)

        if extras_mode == 1:
            for img in image_folder:
                if isinstance(img, Image.Image):
                    image = img
                    fn = ''
                else:
                    image = Image.open(os.path.abspath(img.name))
                    fn = os.path.splitext(img.orig_name)[0]
                image_data.append(image)
                image_names.append(fn)
        elif extras_mode == 2:
            assert not shared.cmd_opts.hide_ui_dir_config, '--hide-ui-dir-config option must be disabled'
            assert input_dir, 'input directory not selected'

            image_list = shared.listfiles(input_dir)
            for filename in image_list:
                try:
                    image = Image.open(filename)
                except Exception:
                    continue
                image_data.append(image)
                image_names.append(filename)
        else:
            assert image, 'image not selected'

            image_data.append(image)
            image_names.append(None)

        if extras_mode == 2 and output_dir != '':
            outpath = output_dir
        else:
            outpath = opts.outdir_samples or opts.outdir_extras_samples

        infotext = ''

        for image, name in zip(image_data, image_names):
            shared.state.textinfo = name

            parameters, existing_pnginfo = images.read_info_from_image(image)
            if parameters:
                existing_pnginfo["parameters"] = parameters

            # api.get_instance().txt2img
            # scripts.scripts_postproc.run(pp, args)
            resize_mode, \
                upscaling_resize, \
                upscaling_resize_w, \
                upscaling_resize_h, \
                upscaling_crop, \
                extras_upscaler_1, \
                extras_upscaler_2, \
                extras_upscaler_2_visibility, \
                gfpgan_visibility, codeformer_visibility, \
                codeformer_weight, *extra_args = args

            if extras_upscaler_1 and extras_upscaler_1 != 'None':
                extras_upscaler_1 = self._binding.find_name_by_alias(extras_upscaler_1)
            if extras_upscaler_2 and extras_upscaler_2 != 'None':
                extras_upscaler_2 = self._binding.find_name_by_alias(extras_upscaler_2)

            imgs = api.get_instance().upscale(image,
                                              resize_mode,
                                              upscaling_resize,
                                              upscaling_resize_w,
                                              upscaling_resize_h,
                                              upscaling_crop,
                                              extras_upscaler_1,
                                              extras_upscaler_2,
                                              extras_upscaler_2_visibility,
                                              gfpgan_visibility,
                                              codeformer_visibility,
                                              codeformer_weight,
                                              *extra_args)
            pp = scripts_postprocessing.PostprocessedImage(imgs[0].convert("RGB"))

            if opts.use_original_name_batch and name is not None:
                basename = os.path.splitext(os.path.basename(name))[0]
            else:
                basename = ''

            infotext = ", ".join([k if k == v else f'{k}: {generation_parameters_copypaste.quote(v)}' for k, v in pp.info.items() if v is not None])

            if opts.enable_pnginfo:
                pp.image.info = existing_pnginfo
                pp.image.info["postprocessing"] = infotext

            if save_output:
                images.save_image(pp.image, path=outpath, basename=basename, seed=None, prompt=None, extension=opts.samples_format, info=infotext,
                                  short_filename=True, no_prompt=True, grid=False, pnginfo_section_name="extras", existing_info=existing_pnginfo, forced_filename=None)

            if extras_mode != 2 or show_extras_results:
                outputs.append(pp.image)

        return outputs, ui_common.plaintext_to_html(infotext), ''


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
        "Model": (None if not opts.add_model_name_to_info or not p._cloud_inference_settings['sd_checkpoint'] else p._cloud_inference_settings['sd_checkpoint'].replace(',', '').replace(':', '')),
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
        f'{k}: {generation_parameters_copypaste.quote(v)}'
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


def _hijack_func(module_name, func_name, new_func):
    old_func = None
    extension_mode = False
    extension_prefix = ""
    if module_name.startswith('extensions.'):
        extension_mode = True

    # from modules.processing import process_images
    search_names = [module_name]
    search_names.append(func_name)
    tmp = module_name.split(".")
    if len(tmp) >= 2:
        search_names.append(".".join(tmp[-2:]))  # import modules.processing
        search_names.append(tmp[-1])  # from modules import processing
        # from modules import processing.process_images
        search_names.append("{}.{}".format(tmp[-1], func_name))

    if not extension_mode:
        # hajiack for normal module

        # case 1: import module, replace function, return old function
        module = importlib.import_module(module_name)
        old_func = getattr(module, func_name)
        setattr(module, func_name, new_func)

        # case 2: from module import func_name
        keys = list(sys.modules.keys())
        for name in keys:
            # if (name.startswith('modules')
            # or name.startswith('scripts')) and name != 'modules.processing':
            if name.startswith('modules') and name != module_name:
                members = getmembers(sys.modules[name], isfunction)
                if func_name in dict(members):
                    # func_fullname = '{}.{}'.format(members[func_name].__module__, members[func_name].__name__)
                    # print(func_fullname, '{}.{}'.format(module_name, func_name))
                    # if func_fullname == '{}.{}'.format(module_name, func_name):
                    print('[cloud-inference] reloading', name)
                    importlib.reload(sys.modules[name])

        from modules.scripts import scripts_data
        for script in scripts_data:
            for name in search_names:
                if name in script.module.__dict__:
                    obj = script.module.__dict__[name]
                    replace = False
                    if ismodule(obj) and obj.__file__ == module.__file__:
                        replace = True
                    elif isfunction(obj) and obj.__module__ == module_name:  # ??
                        replace = True

                    if replace:
                        if name == func_name:
                            print('[cloud-inference] reloading {} - {}'.format(script.module.__name__, func_name))
                            setattr(script.module, name, new_func)
                        else:
                            print('[cloud-inference] reloading {} - {}'.format(script.module.__name__, name))
                            t = getattr(script.module, name)
                            setattr(t, func_name, new_func)
                            # setattr(script.module, name, t)  # ?
        return old_func
    else:
        # hijack for extension module

        from modules.scripts import scripts_data
        tmp1, tmp2, extension_suffix = module_name.split(
            '.', 2)  # scripts internal module name
        extension_prefix = "{}.{}".format(tmp1, tmp2)
        module_name = "modules.{}".format(extension_suffix)

        for script in scripts_data:
            if extension_mode and os.path.join(*extension_prefix.split('.')) not in script.basedir:
                continue
            for name in search_names:
                if name in script.module.__dict__:
                    obj = script.module.__dict__[name]
                    replace = False
                    if ismodule(obj) and module_name.endswith(obj.__name__):
                        replace = True
                    elif isfunction(obj) and obj.__module__ == module_name:  # ??
                        replace = True

                    if replace:
                        # import pkgutil
                        # s = [_ for _ in pkgutil.iter_modules([os.path.dirname(script.module.__file__)])]
                        if name == func_name:
                            print(
                                '[cloud-inference] reloading {} - {}'.format(script.module.__name__, func_name))
                            old_func = getattr(script.module, name)
                            setattr(script.module, name, new_func)
                        else:
                            print(
                                '[cloud-inference] reloading {} - {}'.format(script.module.__name__, name))
                            t = getattr(script.module, name)
                            old_func = getattr(t, func_name)
                            setattr(t, func_name, new_func)
                            setattr(script.module, name, t)
                    # print(importlib.reload(importlib.import_module('extensions.sd-webui-controlnet.scripts.xyz_grid_support')))
        return old_func


_hijack_manager = _HijackManager()
