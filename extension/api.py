import requests
import time
import io
import base64
from modules import sd_samplers, processing
from modules.shared import opts, state
from PIL import Image
from multiprocessing.pool import ThreadPool
import random
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


class StableDiffusionModel(object):
    def __init__(self,
                 kind,
                 name,
                 rating=0,
                 tags=None,
                 child=None,
                 example=None,
                 dependency_model_name=None,
                 user_tags=None):
        self.kind = kind  # checkpoint, lora
        self.name = name
        self.rating = rating
        self.tags = tags
        if self.tags is None:
            self.tags = []

        self.user_tags = user_tags
        if self.user_tags is None:
            self.user_tags = []

        self.child = child
        if self.child is None:
            self.child = []

        self.example = example
        self.dependency_model_name = dependency_model_name

    def append_child(self, child):
        self.child.append(child)

    @property
    def display_name(self):
        # format -> [<ckpt/lora>] [<tag>] <name>
        n = ""
        if len(self.tags) > 0:
            n = "[{}] ".format(",".join(self.tags))
        return n + os.path.splitext(self.name)[0]

    def to_json(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, StableDiffusionModelExample):
                d[k] = v.__dict__
            else:
                d[k] = v
        return d


class StableDiffusionModelExample(object):

    def __init__(self,
                 prompts=None,
                 neg_prompt=None,
                 sampler_name=None,
                 steps=None,
                 cfg_scale=None,
                 seed=None,
                 height=None,
                 width=None,
                 preview=None,
                 ):
        self.prompts = prompts
        self.neg_prompt = neg_prompt
        self.sampler_name = sampler_name
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.height = height
        self.width = width
        self.preview = preview


class OmniinferAPI(BaseAPI):

    def __init__(self, token=None):
        self._token = None
        if self._token is not None:
            self._token = token
        self._models = None
        self._session = requests.Session()
        self._session.headers.update({'User-Agent': _user_agent()})

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
            o._token = config['key']
        try:
            if config.get('models') is not None:
                o._models = []
                for model in config['models']:
                    if model.get('example'):
                        model['example'] = StableDiffusionModelExample(
                            **model['example'])
                    o._models.append(StableDiffusionModel(**model))
        except Exception as e:
            print(
                '[cloud-inference] failed to load models from config file, we will create a new one'
            )
        return o

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

        config['models'] = []
        with open(OMNIINFER_CONFIG, 'wb+') as f:
            for model in models:
                config['models'].append(model.to_json())

            f.write(
                json.dumps(config, ensure_ascii=False, indent=2,
                           default=vars).encode('utf-8'))

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
    def test_connection(cls, token):
        if token == "":
            raise Exception("Token is empty")
        res = requests.get('{}/v2/progress'.format(OMNIINFER_API_ENDPOINT),
                           params={'key': token})
        if res.status_code >= 400:
            raise Exception("Request failed: {}".format(res.text))
        if res.json()['code'] == 4:
            raise Exception("Request failed: {}".format(res.text))

        return "Omniinfer Ready... now you can inference on cloud"

    def _wait_task_completed(self, task_id):
        STATUS_CODE_PENDING = 0
        STATUS_CODE_PROGRESSING = 1
        STATUS_CODE_SUCCESS = 2
        STATUS_CODE_FAILED = 3
        STATUS_CODE_TIMEOUT = 4

        attempts = 300

        # queue(0-20), generating(20-90), downloading(90-100)
        global_progress = 0
        while attempts > 0:
            if state.skipped or state.interrupted:
                raise Exception("Interrupted")

            task_res = self._session.get(
                "{}/v2/progress".format(OMNIINFER_API_ENDPOINT),
                params={
                    "key": self._token,
                    "task_id": task_id,
                    'Accept-Encoding': 'gzip, deflate',
                },
                headers={"X-OmniInfer-Source": "sd-webui"})

            task_res_json = task_res.json()
            generate_progress = task_res_json["data"]["progress"]

            status_code = task_res_json["data"]["status"]

            if status_code == STATUS_CODE_PROGRESSING:
                global_progress += (0.7 * generate_progress)
                if global_progress >= 0.9:
                    global_progress = 0.9  # reverse download time
            if status_code == STATUS_CODE_PENDING and global_progress < 0.2:
                global_progress += 0.05
            elif status_code == STATUS_CODE_SUCCESS:
                return task_res_json["data"]["imgs"]
            elif status_code == STATUS_CODE_TIMEOUT:
                raise Exception("failed to generate image({}): timeout",
                                task_id)
            elif status_code == STATUS_CODE_FAILED:
                raise Exception("failed to generate image({}): {}",
                                task_res_json["data"]["failed_reason"])

            state.sampling_step = int(state.sampling_steps * state.job_count *
                                      global_progress)

            attempts -= 1
            time.sleep(0.5)

        raise Exception("failed to generate image({}): timeout", task_id)

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

            buffered = io.BytesIO()
            i.save(buffered, format=live_previews_image_format, **save_kwargs)
            base64_image = base64.b64encode(
                buffered.getvalue()).decode('ascii')
            images_base64.append(base64_image)

        def _req(p: processing.StableDiffusionProcessingImg2Img, controlnet_units):
            req = {
                "model_name": p._cloud_inference_settings['sd_checkpoint'],
                "init_images": [image_to_base64(_) for _ in p.init_images],
                "mask": image_to_base64(p.image_mask) if p.image_mask else None,
                "resize_mode": p.resize_mode,
                "denoising_strength": p.denoising_strength,
                "cfg_scale": p.image_cfg_scale,
                "mask_blur": p.mask_blur_x,
                "inpainting_fill": p.inpainting_fill,
                "inpaint_full_res": bool2int(p.inpaint_full_res),
                "inpaint_full_res_padding": p.inpaint_full_res_padding,
                "inpainting_mask_invert": p.inpainting_mask_invert,
                "initial_noise_multiplier": p.initial_noise_multiplier,
                "prompt": p.prompt,
                "seed": int(p.seed) or -1,
                "negative_prompt": p.negative_prompt,
                "batch_size": p.batch_size,
                "n_iter": p.n_iter,
                "steps": p.steps,
                "width": p.width,
                "height": p.height,
                "restore_faces": p.restore_faces,
                "clip_skip": opts.CLIP_stop_at_last_layers,
            }
            if 'CLIP_stop_at_last_layers' in p.override_settings:
                req['clip_skip'] = p.override_settings['CLIP_stop_at_last_layers']

            if 'sd_vae' in p._cloud_inference_settings:
                req['sd_vae'] = p._cloud_inference_settings['sd_vae']

            if len(controlnet_units) > 0:
                req['controlnet_units'] = controlnet_units

            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                'Accept-Encoding': 'gzip, deflate',
                "X-OmniInfer-Source": _user_agent(p._cloud_inference_settings['sd_checkpoint']),
                "X-OmniInfer-Key": self._token,
                "User-Agent": _user_agent(p._cloud_inference_settings['sd_checkpoint'])
            }

            res = self._session.post("{}/v2/img2img".format(OMNIINFER_API_ENDPOINT),
                                     json=req,
                                     headers=headers,
                                     params={"key": self._token})

            try:
                json_data = res.json()
            except Exception:
                raise Exception("Request failed: {}".format(res.text))

            if json_data['code'] != 0:
                raise Exception("Request failed: {}".format(res.text))

            return self._wait_task_completed(json_data['data']['task_id'])

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
            req = {
                "model_name": p._cloud_inference_settings['sd_checkpoint'],
                "prompt": p.prompt,
                "negative_prompt": p.negative_prompt,
                "sampler_name": p.sampler_name or "Euler a",
                "batch_size": p.batch_size or 1,
                "n_iter": p.n_iter or 1,
                "steps": p.steps or 30,
                "cfg_scale": p.cfg_scale or 7.5,
                "seed": int(p.seed) or -1,
                "height": p.height or 512,
                "width": p.width or 512,
                "restore_faces": p.restore_faces,
                "clip_skip": opts.CLIP_stop_at_last_layers,
            }

            if 'CLIP_stop_at_last_layers' in p.override_settings:
                req['clip_skip'] = p.override_settings['CLIP_stop_at_last_layers']

            if 'sd_vae' in p._cloud_inference_settings:
                req['sd_vae'] = p._cloud_inference_settings['sd_vae']

            if len(controlnet_units) > 0:
                req['controlnet_units'] = controlnet_units

            headers = {
                "accept": "application/json",
                "content-type": "application/json",
                'Accept-Encoding': 'gzip, deflate',
                "X-OmniInfer-Source": _user_agent(p._cloud_inference_settings['sd_checkpoint']),
                "X-OmniInfer-Key": self._token,
                "User-Agent": _user_agent(p._cloud_inference_settings['sd_checkpoint'])
            }

            res = self._session.post("{}/v2/txt2img".format(OMNIINFER_API_ENDPOINT),
                                     json=req,
                                     headers=headers,
                                     params={"key": self._token})
            try:
                json_data = res.json()
            except Exception:
                raise Exception("Request failed: {}".format(res.text))

            if json_data['code'] != 0:
                raise Exception("Request failed: {}".format(res.text))

            return self._wait_task_completed(json_data['data']['task_id'])

        imgs = []
        if len(controlnet_batchs) > 0:
            for c in controlnet_batchs:
                imgs.extend(_req(p, c))
        else:
            imgs.extend(_req(p, []))

        state.textinfo = "downloading images..."

        return retrieve_images(imgs)

    def list_models(self):
        if self._models is None or len(self._models) == 0:
            self._models = self.refresh_models()
        return sorted(self._models, key=lambda x: x.rating, reverse=True)

    def refresh_models(self):

        def get_models(kind):
            url = "{}/v2/models".format(OMNIINFER_API_ENDPOINT)
            headers = {
                "accept": "application/json",
                'Accept-Encoding': 'gzip, deflate',
                "X-OmniInfer-Source": _user_agent(),
                "User-Agent": _user_agent()
            }

            res = requests.get(url, headers=headers, params={"type": kind})
            if res.status_code >= 400:
                return []

            models = []
            if res.json()["data"]["models"] is not None:
                models = res.json()["data"]["models"]

            for item in models:
                model = StableDiffusionModel(kind=item["type"],
                                             name=item["sd_name"])
                model.rating = item.get("civitai_download_count", 0)
                civitai_tags = item["civitai_tags"].split(",") if item.get(
                    "civitai_tags", None) is not None else []

                if model.tags is None:
                    model.tags = []

                if len(civitai_tags) > 0:
                    model.tags.append(civitai_tags[0])

                if item.get('civitai_nsfw', False):
                    model.tags.append("nsfw")

                if len(item.get('civitai_images',
                                [])) > 0 and item['civitai_images'][0]['meta'].get(
                                    'prompt') is not None:
                    first_image = item['civitai_images'][0]
                    first_image_meta = item['civitai_images'][0]['meta']
                    model.example = StableDiffusionModelExample(
                        prompts=first_image_meta['prompt'],
                        neg_prompt=first_image_meta.get(
                            'negative_prompt', None),
                        width=first_image_meta.get('width', None),
                        height=first_image_meta.get('height', None),
                        sampler_name=first_image_meta.get(
                            'sampler_name', None),
                        cfg_scale=first_image_meta.get('cfg_scale', None),
                        seed=first_image_meta.get('seed', None),
                        preview=first_image.get('url', None)
                    )

                if item['type'] == 'lora':
                    civitai_dependency_model_name = item.get(
                        'civitai_dependency_model_name', None)
                    if civitai_dependency_model_name is not None:
                        model.dependency_model_name = civitai_dependency_model_name
                sd_models.append(model)

        m = {}
        sd_models = []
        print("[cloud-inference] refreshing models...")

        get_models("checkpoint")
        get_models("lora")
        get_models("controlnet")
        get_models("vae")

        # build lora and checkpoint relationship
        for model in sd_models:
            m[model.name] = model

        for _, model in m.items():
            if model.dependency_model_name is not None:
                if m.get(model.dependency_model_name) is not None:
                    m[model.dependency_model_name].append_child(model.name)

        self.__class__.update_models_to_config(sd_models)
        self._models = sd_models
        return sd_models


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
    controlnet_units = get_visible_extension_args(p, 'controlnet')
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

        if getattr(c.input_mode, 'value', '') == "simple":
            if c.image:
                if "mask" in c.image:
                    mask = Image.fromarray(c.image["mask"])
                    controlnet_arg['mask'] = image_to_base64(mask)

                controlnet_arg['input_image'] = image_to_base64(
                    Image.fromarray(c.image["image"]))

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


def retrieve_images(img_urls):
    def _download(img_url):
        attempts = 5
        while attempts > 0:
            try:
                response = requests.get(img_url, timeout=2)
                return Image.open(io.BytesIO(response.content))
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
