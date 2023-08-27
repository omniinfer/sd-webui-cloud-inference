import requests
import io
import base64
from modules import processing
from modules.shared import opts, state
from PIL import Image, ImageFilter, ImageOps
from multiprocessing.pool import ThreadPool
import importlib

from omniinfer_client import *
from dataclass_wizard import JSONWizard, DumpMeta
from dataclasses import dataclass, field

from typing import Dict

import numpy as np

import os
import copy
import json


from .utils import image_to_base64, read_image_files
from .version import __version__

OMNIINFER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '.omniinfer.json')

OMNIINFER_API_ENDPOINT = "https://api.omniinfer.io"


def _user_agent(model_name=None):
    if model_name:
        return 'sd-webui-cloud-inference/{} (model_name: {})'.format(__version__, model_name)
    return 'sd-webui-cloud-inference/{}'.format(__version__)


class BaseAPI(object):
    def txt2img(self, p: processing.StableDiffusionProcessingTxt2Img):
        pass

    def img2img(self, p: processing.StableDiffusionProcessingImg2Img):
        pass

    def list_models():
        pass

    def refresh_models():
        pass


class UpscaleAPI(object):
    def upscale(self, *args, **kwargs):
        pass


class JSONe(JSONWizard):
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        DumpMeta(key_transform='SNAKE').bind_to(cls)


@dataclass
class StableDiffusionModelExample(JSONe):
    prompts: Optional[str] = None
    neg_prompt: Optional[str] = None
    sampler_name: Optional[str] = None
    steps: Optional[int] = None
    seed: Optional[int] = None
    height: Optional[int] = None
    width: Optional[int] = None
    preview: Optional[str] = None
    cfg_scale: Optional[float] = None


@dataclass
class StableDiffusionModel(JSONe):
    kind: str
    name: str
    rating: int = 0
    tags: List[str] = None
    child: Optional[List[str]] = field(default_factory=lambda: [])
    examples: Optional[List[StableDiffusionModelExample]] = field(default_factory=lambda: [])
    user_tags: Optional[List[str]] = field(default_factory=lambda: [])
    preview_url: Optional[str] = None
    search_terms: Optional[List[str]] = field(default_factory=lambda: [])
    origin_url: Optional[str] = None

    @property
    def alias(self):
        # format -> [<ckpt/lora>] [<tag>] <name>
        if self.kind in ["upscaler", "controlnet"]:
            return self.name
            # return "cloud://{}".format(self.name)

        n = ""
        if len(self.tags) > 0:
            n = "[{}] ".format(",".join(self.tags))
        return n + os.path.splitext(self.name)[0]

    def add_user_tag(self, tag):
        if tag not in self.user_tags:
            self.user_tags.append(tag)


# class StableDiffusionModelExample(object):

#     def __init__(self,
#                  prompts=None,
#                  neg_prompt=None,
#                  sampler_name=None,
#                  steps=None,
#                  cfg_scale=None,
#                  seed=None,
#                  height=None,
#                  width=None,
#                  preview=None,
#                  ):
#         self.prompts = prompts
#         self.neg_prompt = neg_prompt
#         self.sampler_name = sampler_name
#         self.steps = steps
#         self.cfg_scale = cfg_scale
#         self.seed = seed
#         self.height = height
#         self.width = width
#         self.preview = preview


class OmniinferAPI(BaseAPI, UpscaleAPI):

    def __init__(self, api_key=None):
        self._api_key = api_key
        self._client: OmniClient = None

        if self._api_key is not None:
            self.update_client()

        self._models: List[StableDiffusionModel] = []

    def update_client(self):
        self._client = OmniClient(self._api_key)
        self._client.set_extra_headers({'User-Agent': _user_agent()})

    @classmethod
    def load_from_config(cls):
        config = {}
        try:
            with open(OMNIINFER_CONFIG, 'r') as f:
                config = json.load(f)
        except Exception as exp:
            pass

        o = OmniinferAPI()
        if config.get('key') is not None:
            o._api_key = config['key']
            o.update_client()
        else:
            # if no key, we will set it to NONE
            o._api_key = 'NONE'
            o.update_client()

        if config.get('models') is not None:
            try:
                o._models = [StableDiffusionModel.from_dict(m) for m in config['models']]
            except Exception as exp:
                print('[cloud-inference] failed to load models from config file {}, we will create a new one'.format(exp))
                o._models = []

        return o

    @classmethod
    def update_key_to_config(cls, key):
        config = {}
        if os.path.exists(OMNIINFER_CONFIG):
            with open(OMNIINFER_CONFIG, 'r') as f:
                try:
                    config = json.load(f)
                except:
                    print(
                        '[cloud-inference] failed to load config file, we will create a new one'
                    )
                    pass

        config['key'] = key
        with open(OMNIINFER_CONFIG, 'wb+') as f:
            f.write(
                json.dumps(config, ensure_ascii=False, indent=2,
                           default=vars).encode('utf-8'))

    @classmethod
    def update_models_to_config(cls, models):
        config = {}
        if os.path.exists(OMNIINFER_CONFIG):
            with open(OMNIINFER_CONFIG, 'r') as f:
                try:
                    config = json.load(f)
                except:
                    print(
                        '[cloud-inference] failed to load config file, we will create a new one'
                    )
                    pass

        config['models'] = models
        with open(OMNIINFER_CONFIG, 'wb+') as f:
            f.write(
                json.dumps(config, ensure_ascii=False, indent=2,
                           default=vars).encode('utf-8'))

    @classmethod
    def test_connection(cls, api_key: str):
        client = OmniClient(api_key)
        try:
            res = client.progress("sd-webui-test")
        except Exception as e:
            raise Exception("Failed to connect to Omniinfer API: {}".format(e))
        if res.code == ProgressResponseCode.INVALID_AUTH:
            raise Exception("Invalid API key")
        return "✅ Omniinfer Ready... now you can inference on cloud"

    def _update_state(self, progress: ProgressResponse):
        # queue(0-10), generating(10-90), downloading(90-100)
        if state.skipped or state.interrupted:
            raise Exception("Interrupted")

        progress_data = progress.data

        if progress_data.status == ProgressResponseStatusCode.RUNNING:
            global_progress = (0.7 * progress_data.progress)
            if global_progress < 0.1:
                global_progress = 0.1

            if global_progress >= 0.9:
                global_progress = 0.9  # reverse download time
        if progress_data.status == ProgressResponseStatusCode.INITIALIZING:
            global_progress = 0.1
        elif progress_data.status == ProgressResponseStatusCode.SUCCESSFUL:
            global_progress = 0.9
        elif progress_data.status == ProgressResponseStatusCode.TIMEOUT:
            raise Exception("failed to generate image: timeout")
        elif progress_data.status == ProgressResponseStatusCode.FAILED:
            raise Exception("failed to generate image({}): {}", progress.data.failed_reason)

        state.sampling_step = int(state.sampling_steps * state.job_count * global_progress)

    def img2img(
        self,
        p: processing.StableDiffusionProcessingImg2Img,
    ):
        controlnet_batchs = get_controlnet_arg(p)

        live_previews_image_format = "png"
        if getattr(opts, 'live_previews_image_format', None):
            live_previews_image_format = opts.live_previews_image_format

        images_base64 = []
        for i in p.init_images:
            if live_previews_image_format == "png":
                # using optimize for large images takes an enormous amount of time
                if max(*i.size) <= 256:
                    save_kwargs = {"optimize": True}
                else:
                    save_kwargs = {"optimize": False, "compress_level": 1}

            else:
                save_kwargs = {}

            with io.BytesIO() as buffered:
                i.save(buffered, format=live_previews_image_format, **save_kwargs)
                base64_image = base64.b64encode(buffered.getvalue()).decode('ascii')
                images_base64.append(base64_image)

        def _req(p: processing.StableDiffusionProcessingImg2Img, controlnet_units):
            req = Img2ImgRequest(
                model_name=p._cloud_inference_settings['sd_checkpoint'],
                sampler_name=p.sampler_name,
                init_images=images_base64,
                mask=image_to_base64(p.image_mask) if p.image_mask else None,
                resize_mode=p.resize_mode,
                denoising_strength=p.denoising_strength,
                cfg_scale=p.image_cfg_scale,
                mask_blur=p.mask_blur_x,
                inpaint_full_res=bool2int(p.inpaint_full_res),
                inpaint_full_res_padding=p.inpaint_full_res_padding,
                initial_noise_multiplier=p.initial_noise_multiplier,
                inpainting_mask_invert=bool2int(p.inpainting_mask_invert),
                prompt=p.prompt,
                seed=int(p.seed) or -1,
                negative_prompt=p.negative_prompt,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                width=p.width,
                height=p.height,
                restore_faces=p.restore_faces,
                clip_skip=opts.CLIP_stop_at_last_layers,
            )
            if 'CLIP_stop_at_last_layers' in p.override_settings:
                req.clip_skip = p.override_settings['CLIP_stop_at_last_layers']

            if 'sd_vae' in p._cloud_inference_settings:
                req.sd_vae = p._cloud_inference_settings['sd_vae']

            if hasattr(p, 'refiner_checkpoint') and p.refiner_checkpoint is not None and p.refiner_checkpoint != "None":
                req.sd_refiner = Refiner(
                    checkpoint=p.refiner_checkpoint,
                    switch_at=p.refiner_switch_at,
                )

            if len(controlnet_units) > 0:
                req.controlnet_units = controlnet_units
                if opts.data.get("control_net_no_detectmap", False):
                    req.controlnet_no_detectmap = True

            res = self._client.sync_img2img(req, download_images=False, callback=self._update_state)
            return res.data.imgs

        controlnet_batchs = get_controlnet_arg(p)

        imgs = []
        if len(controlnet_batchs) > 0:
            for c in controlnet_batchs:
                imgs.extend(_req(p, c))
        else:
            imgs.extend(_req(p, []))
        return retrieve_images(imgs)

    def txt2img(self, p: processing.StableDiffusionProcessingTxt2Img):
        controlnet_batchs = get_controlnet_arg(p)

        def _req(p: processing.StableDiffusionProcessingTxt2Img, controlnet_units):
            req = Txt2ImgRequest(
                model_name=p._cloud_inference_settings['sd_checkpoint'],
                sampler_name=p.sampler_name,
                prompt=p.prompt,
                negative_prompt=p.negative_prompt,
                batch_size=p.batch_size,
                n_iter=p.n_iter,
                steps=p.steps,
                cfg_scale=p.cfg_scale,
                seed=int(p.seed) or -1,
                height=p.height,
                width=p.width,
                restore_faces=p.restore_faces,
                clip_skip=opts.CLIP_stop_at_last_layers,
            )

            if p.enable_hr:
                req.enable_hr = True
                req.hr_upscaler = p.hr_upscaler
                req.hr_scale = p.hr_scale
                req.hr_resize_x = p.hr_resize_x
                req.hr_resize_y = p.hr_resize_y

            if 'CLIP_stop_at_last_layers' in p.override_settings:
                req.clip_skip = p.override_settings['CLIP_stop_at_last_layers']
            if 'sd_vae' in p._cloud_inference_settings:
                req.sd_vae = p._cloud_inference_settings['sd_vae']

            if len(controlnet_units) > 0:
                req.controlnet_units = controlnet_units
                if opts.data.get("control_net_no_detectmap", False):
                    req.controlnet_no_detectmap = True

            if hasattr(p, 'refiner_checkpoint') and p.refiner_checkpoint is not None and p.refiner_checkpoint != "None":
                req.sd_refiner = Refiner(
                    checkpoint=p.refiner_checkpoint,
                    switch_at=p.refiner_switch_at,
                )

            res = self._client.sync_txt2img(req, download_images=False, callback=self._update_state)
            if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
                raise Exception(res.data.failed_reason)

            return res.data.imgs

        imgs = []
        if len(controlnet_batchs) > 0:
            for c in controlnet_batchs:
                imgs.extend(_req(p, c))
        else:
            imgs.extend(_req(p, []))

        state.textinfo = "downloading images..."

        return retrieve_images(imgs)

    def upscale(self, image,
                resize_mode: int,
                upscaling_resize: float,
                upscaling_resize_w: int,
                upscaling_resize_h: int,
                upscaling_crop: bool,
                extras_upscaler_1: str,
                extras_upscaler_2: str,
                extras_upscaler_2_visibility: float,
                gfpgan_visibility: float,
                codeformer_visibility: float,
                codeformer_weight: float,
                *args,
                **kwargs
                ):
        req = UpscaleRequest(
            image=image_to_base64(image),
            upscaler_1=extras_upscaler_1,
            resize_mode=resize_mode,
            upscaling_resize=upscaling_resize,
            upscaling_resize_w=upscaling_resize_w,
            upscaling_resize_h=upscaling_resize_h,
            upscaling_crop=upscaling_crop,
            upscaler_2=extras_upscaler_2,
            extras_upscaler_2_visibility=extras_upscaler_2_visibility,
            gfpgan_visibility=gfpgan_visibility,
            codeformer_visibility=codeformer_visibility,
            codeformer_weight=codeformer_weight
        )

        res = self._client.sync_upscale(req, download_images=False, callback=self._update_state)
        if res.data.status != ProgressResponseStatusCode.SUCCESSFUL:
            raise Exception(res.data.failed_reason)
        return retrieve_images(res.data.imgs)

    def list_models(self):
        if self._models is None or len(self._models) == 0:
            self._models = self.refresh_models()
        return sorted(self._models, key=lambda x: x.rating, reverse=True)

    def refresh_models(self):

        def get_models(type_):
            ret = []
            models = self._client.models(refresh=True).filter_by_type(type_)
            for item in models:
                model = StableDiffusionModel(kind=item.type.value,
                                             name=item.sd_name)
                model.search_terms = [
                    item.sd_name,
                    item.name,
                    str(item.civitai_version_id)
                ]
                model.rating = item.civitai_download_count
                civitai_tags = item.civitai_tags.split(",") if item.civitai_tags is not None else []

                if model.tags is None:
                    model.tags = []

                if len(civitai_tags) > 0:
                    model.tags.append(civitai_tags[0])

                if item.civitai_nsfw or item.civitai_image_nsfw:
                    model.tags.append("nsfw")

                if item.civitai_image_url:
                    model.preview_url = item.civitai_image_url

                model.examples = []
                if item.civitai_images:
                    for img in item.civitai_images:
                        if img.meta.prompt:
                            model.examples.append(StableDiffusionModelExample(
                                prompts=img.meta.prompt,
                                neg_prompt=img.meta.negative_prompt,
                                width=img.meta.width,
                                height=img.meta.height,
                                sampler_name=img.meta.sampler_name,
                                cfg_scale=img.meta.cfg_scale,
                            ))

                ret.append(model)
            return ret

        sd_models = []
        print("[cloud-inference] refreshing models...")

        sd_models.extend(get_models(ModelType.CHECKPOINT))
        sd_models.extend(get_models(ModelType.LORA))
        sd_models.extend(get_models(ModelType.CONTROLNET))
        sd_models.extend(get_models(ModelType.VAE))
        sd_models.extend(get_models(ModelType.UPSCALER))
        sd_models.extend(get_models(ModelType.TEXT_INVERSION))

        # build lora and checkpoint relationship

        merged_models = {}
        origin_models = {}
        for model in self._models:
            origin_models[model.name] = model
        for model in sd_models:
            if model.name in origin_models:
                # save user tags
                merged_models[model.name] = model
                merged_models[model.name].user_tags = origin_models[model.name].user_tags
            else:
                merged_models[model.name] = model

        self._models = [v for k, v in merged_models.items()]
        self.update_models_to_config(self._models)
        return self._models


_instance = None


def get_instance():
    global _instance
    if _instance is not None:
        return _instance
    _instance = OmniinferAPI.load_from_config()
    return _instance


def refresh_instance():
    global _instance
    _instance = OmniinferAPI.load_from_config()
    return _instance


def get_visible_extension_args(p: processing.StableDiffusionProcessing, name):
    for s in p.scripts.alwayson_scripts:
        if s.name == name:
            return p.script_args[s.args_from:s.args_to]
    return []


def get_controlnet_arg(p: processing.StableDiffusionProcessing):
    controlnet_batchs = []

    # controlnet_units = get_visible_extension_args(p, 'controlnet')
    try:
        external_code = importlib.import_module('extensions.sd-webui-controlnet.scripts.external_code', 'external_code')
    except ModuleNotFoundError:
        return []

    controlnet_units = external_code.get_all_units_in_processing(p)

    for c in controlnet_units:
        if c.enabled == False:
            continue

        controlnet_arg = {}
        controlnet_arg['weight'] = c.weight
        controlnet_arg['model'] = c.model
        controlnet_arg['module'] = c.module
        if c.resize_mode == "Just Resize":
            controlnet_arg['resize_mode'] = 0
        elif c.resize_mode == "Resize and Crop":
            controlnet_arg['resize_mode'] = 1
        elif c.resize_mode == "Envelope (Outer Fit)":
            controlnet_arg['resize_code'] = 2

        if 'processor_res' in c.__dict__:
            if c.processor_res > 0:
                controlnet_arg['processor_res'] = c.processor_res

        if 'threshold_a' in c.__dict__:
            controlnet_arg['threshold_a'] = int(c.threshold_a)
        if 'threshold_b' in c.__dict__:
            controlnet_arg['threshold_b'] = int(c.threshold_b)
        if 'guidance_start' in c.__dict__:
            controlnet_arg['guidance_start'] = c.guidance_start
        if 'guidance_end' in c.__dict__:
            controlnet_arg['guidance_end'] = c.guidance_end

        if c.control_mode == "Balanced":
            controlnet_arg['control_mode'] = 0
        elif c.control_mode == "My prompt is more important":
            controlnet_arg['control_mode'] = 1
        elif c.control_mode == "ControlNet is more important":
            controlnet_arg['control_mode'] = 2
        else:
            return

        img2img = isinstance(p, processing.StableDiffusionProcessingImg2Img)
        if img2img and not c.image:
            c.image = {}
            init_image = getattr(p, "init_images", [None])[0]
            if init_image is not None:
                c.image['image'] = np.asarray(init_image)

            a1111_i2i_resize_mode = getattr(p, "resize_mode", None)
            # TODO: mask

            if a1111_i2i_resize_mode is not None:
                controlnet_arg['resize_mode'] = a1111_i2i_resize_mode

        if getattr(c.input_mode, 'value', '') == "simple":
            if c.image is not None:
                c.image = image_dict_from_any(c.image)
                if "mask" in c.image:
                    mask = Image.fromarray(c.image["mask"])
                    controlnet_arg['mask'] = image_to_base64(mask)

                controlnet_arg['input_image'] = image_to_base64(Image.fromarray(c.image["image"]))

                if len(controlnet_batchs) == 0:
                    controlnet_batchs = [[]]

                controlnet_batchs[0].append(controlnet_arg)

        elif getattr(c.input_mode, 'value', '') == "batch":
            if c.batch_images != "" and c.batch_images != None:
                images = read_image_files(c.batch_images)
                for i, img in enumerate(images):
                    if len(controlnet_batchs) <= i:
                        controlnet_batchs.append([])

                    controlnet_new_arg = copy.deepcopy(controlnet_arg)
                    controlnet_new_arg['input_image'] = img

                    controlnet_batchs[i].append(controlnet_new_arg)
            else:
                print("batch_images is empty")

        else:
            print("input_mode is empty")

    return controlnet_batchs


def image_has_mask(input_image: np.ndarray) -> bool:
    return (
        input_image.ndim == 3 and
        input_image.shape[2] == 4 and
        np.max(input_image[:, :, 3]) > 127
    )


def prepare_mask(
    mask: Image.Image, p: processing.StableDiffusionProcessing
) -> Image.Image:
    mask = mask.convert("L")
    if getattr(p, "inpainting_mask_invert", False):
        mask = ImageOps.invert(mask)
    if getattr(p, "mask_blur", 0) > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(p.mask_blur))
    return mask


def image_dict_from_any(image) -> Optional[Dict[str, np.ndarray]]:
    if image is None:
        return None

    if isinstance(image, (tuple, list)):
        image = {'image': image[0], 'mask': image[1]}
    elif not isinstance(image, dict):
        image = {'image': image, 'mask': None}
    else:  # type(image) is dict
        # copy to enable modifying the dict and prevent response serialization error
        image = dict(image)

    if isinstance(image['image'], str):
        if os.path.exists(image['image']):
            image['image'] = np.array(Image.open(image['image'])).astype('uint8')
        elif image['image']:
            image['image'] = external_code.to_base64_nparray(image['image'])
        else:
            image['image'] = None

    # If there is no image, return image with None image and None mask
    if image['image'] is None:
        image['mask'] = None
        return image

    if 'mask' not in image:
        image['mask'] = None

    if isinstance(image['mask'], str):
        if os.path.exists(image['mask']):
            image['mask'] = np.array(Image.open(image['mask'])).astype('uint8')
        elif image['mask']:
            image['mask'] = external_code.to_base64_nparray(image['mask'])
        else:
            image['mask'] = np.zeros_like(image['image'], dtype=np.uint8)
    elif image['mask'] is None:
        image['mask'] = np.zeros_like(image['image'], dtype=np.uint8)

    return image


def retrieve_images(img_urls):
    def _download(img_url):
        attempts = 5
        while attempts > 0:
            try:
                response = requests.get(img_url, timeout=2)
                with io.BytesIO(response.content) as fp:
                    return Image.open(fp).copy()
            except Exception:
                print("[cloud-inference] failed to download image, retrying...")
            attempts -= 1
        return None

    pool = ThreadPool()
    applied = []
    for img_url in img_urls:
        applied.append(pool.apply_async(_download, (img_url, )))
    ret = [r.get() for r in applied]
    return [_ for _ in ret if _ is not None]


def bool2int(b):
    if isinstance(b, bool):
        return 1 if b else 0
    return b
