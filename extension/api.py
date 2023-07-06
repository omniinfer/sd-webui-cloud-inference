import requests
import time
import io
import base64
from modules import sd_samplers
from modules.shared import opts, state
from PIL import Image
from multiprocessing.pool import ThreadPool
import random
import os
import copy
import json
from types import SimpleNamespace
from .utils import image_to_base64, read_image_files

OMNIINFER_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                '.omniinfer.json')


class BaseAPI(object):

    def txt2img(self, p) -> list[Image.Image]:
        pass

    def img2img(self, p) -> list[Image.Image]:
        pass

    def list_models() -> list[str]:
        pass

    def refresh_models() -> list[str]:
        pass


class CheckpointExample(object):

    def __init__(self,
                 prompts=None,
                 neg_prompt=None,
                 sampler_name=None,
                 steps=None,
                 cfg_scale=None,
                 seed=None,
                 height=None,
                 width=None):
        self.prompts = prompts
        self.neg_prompt = neg_prompt
        self.sampler_name = sampler_name
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.seed = seed
        self.height = height
        self.width = width


class Checkpoint(object):

    def __init__(self,
                 name: str,
                 rating: int = None,
                 loras: list[str] = None,
                 example: CheckpointExample = None):
        self.name = name
        self.rating = rating
        self.loras = loras
        if self.loras is None:
            self.loras = []
        self.example = example

    def to_json(self):
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, CheckpointExample):
                d[k] = v.__dict__
            else:
                d[k] = v
        return d

    def add_lora(self, lora):
        self.loras.append(lora)


class OmniinferAPI(BaseAPI):

    def __init__(self, token=None):
        self._endpoint = 'https://api.omniinfer.io'
        self._token = None
        if self._token is not None:
            self._token = token
        self._models = None
        # self._txt2img_timeout = 30

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
        if config.get('models') is not None:
            o._models = []
            for model in config['models']:
                if model.get('example'):
                    model['example'] = CheckpointExample(**model['example'])
                o._models.append(Checkpoint(**model))
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
        res = requests.get('https://api.omniinfer.io/v2/progress',
                           params={'key': token})
        if res.status_code >= 400:
            raise Exception("Request failed: {}".format(res.text))
        if res.json()['code'] == 4:
            raise Exception("Request failed: {}".format(res.text))

        return "Omniinfer Ready... now you can inference on cloud"

    def _txt2img(self, model_name, prompts, neg_prompts, sampler_name,
                 batch_size, steps, n_iter, cfg_scale, seed, height, width,
                 controlnet_args):

        if self._token is None:
            raise Exception(
                "Please configure your omniinfer key in the `Cloud Inference` Tab"
            )

        # TODO: workaround
        if isinstance(sampler_name, int):
            sampler_name = sd_samplers[sampler_name]
        payload = {
            "prompt": prompts,
            "negative_prompt": neg_prompts,
            "sampler_name": sampler_name or "Euler a",
            "batch_size": batch_size or 1,
            "n_iter": n_iter or 1,
            "steps": steps or 30,
            "cfg_scale": cfg_scale or 7.5,
            "seed": int(seed) or -1,
            "height": height or 512,
            "width": width or 512,
            # "model_name": "AnythingV5_v5PrtRE.safetensors",
            "model_name": model_name,
            "controlnet_units": controlnet_args
        }

        print(
            '[cloud-inference] call api txt2img: payload: {}'.format({
                key: value
                for key, value in payload.items() if key != "controlnet_units"
            }), )

        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            'Accept-Encoding': 'gzip, deflate',
            "X-OmniInfer-Source": "source"
        }

        try:
            res = requests.post("http://api.omniinfer.io/v2/txt2img",
                                json=payload,
                                headers=headers,
                                params={"key": self._token})
        except Exception as exp:
            raise Exception("Request failed: {}, res: {}".format(
                exp, res.text if res is not None else ""))

        json_data = res.json()

        if json_data['code'] != 0:
            raise Exception("Request failed: {}".format(res.text))

        return json_data['data']['task_id']

    def _img2img(self, model_name, prompts, neg_prompts, sampler_name,
                 batch_size, steps, n_iter, cfg_scale, seed, height, width,
                 restore_faces, denoising_strength, init_images,
                 controlnet_args):

        if self._token is None:
            raise Exception(
                "Please configure your omniinfer key in the `Cloud Inference` Tab"
            )

        if isinstance(sampler_name, int):
            sampler_name = sd_samplers[sampler_name]

        payload = {
            "prompt": prompts,
            "negative_prompt": neg_prompts,
            "sampler_name": sampler_name or "Euler a",
            "batch_size": batch_size or 1,
            "n_iter": n_iter or 1,
            "steps": steps or 30,
            "cfg_scale": cfg_scale or 7.5,
            "seed": int(seed) or -1,
            "height": height or 512,
            "width": width or 512,
            "model_name": model_name,
            "restore_faces": restore_faces,
            "denoising_strength": denoising_strength,
            "init_images": init_images,
            "controlnet_units": controlnet_args
        }
        headers = {
            "accept": "application/json",
            "content-type": "application/json",
            'Accept-Encoding': 'gzip, deflate',
            "X-OmniInfer-Source": "sd-webui"
        }

        res = requests.post("http://api.omniinfer.io/v2/img2img",
                            json=payload,
                            headers=headers,
                            params={"key": self._token})

        json_data = res.json()

        return json_data['data']['task_id']

    def _wait_task_completed(self, task_id):
        STATUS_CODE_PENDING = 0
        STATUS_CODE_PROGRESSING = 1
        STATUS_CODE_SUCCESS = 2
        STATUS_CODE_FAILED = 3
        STATUS_CODE_TIMEOUT = 4

        attempts = 300
        while attempts > 0:
            if state.skipped or state.interrupted:
                raise Exception("Interrupted")

            task_res = requests.get("http://api.omniinfer.io/v2/progress",
                                    params={
                                        "key": self._token,
                                        "task_id": task_id
                                    },
                                    headers={"X-OmniInfer-Source": "sd-webui"})
            task_res_json = task_res.json()
            progress = task_res_json["data"]["progress"]
            if progress == 1:
                progress = 0.9  # reverse 10% progress bar for download
            state.sampling_step = state.sampling_steps * state.job_count * progress

            status_code = task_res_json["data"]["status"]

            if status_code == STATUS_CODE_SUCCESS:
                return task_res_json["data"]["imgs"]
            elif status_code == STATUS_CODE_TIMEOUT:
                raise Exception("failed to generate image({}): timeout",
                                task_id)
            elif status_code == STATUS_CODE_FAILED:
                raise Exception("failed to generate image({}): {}",
                                task_res_json["data"]["failed_reason"])
            attempts -= 1
            time.sleep(0.5)

        raise Exception("failed to generate image({}): timeout", task_id)

    def img2img(
        self,
        p,
    ):

        controlnet_batchs = self.check_controlnet_arg(p)

        images_base64 = []
        for i in p.init_images:
            if opts.live_previews_image_format == "png":
                # using optimize for large images takes an enormous amount of time
                if max(*i.size) <= 256:
                    save_kwargs = {"optimize": True}
                else:
                    save_kwargs = {"optimize": False, "compress_level": 1}

            else:
                save_kwargs = {}

            buffered = io.BytesIO()
            i.save(buffered,
                   format=opts.live_previews_image_format,
                   **save_kwargs)
            base64_image = base64.b64encode(
                buffered.getvalue()).decode('ascii')
            images_base64.append(base64_image)

        img_urls = []
        if len(controlnet_batchs) > 0:
            for c in controlnet_batchs:
                img_urls.extend(
                    self._wait_task_completed(
                        self._img2img(model_name=p._remote_model_name,
                                      prompts=p.prompt,
                                      neg_prompts=p.negative_prompt,
                                      sampler_name=p.sampler_name,
                                      batch_size=p.batch_size,
                                      steps=p.steps,
                                      n_iter=p.n_iter,
                                      cfg_scale=p.cfg_scale,
                                      seed=p.seed,
                                      height=p.height,
                                      width=p.width,
                                      restore_faces=p.restore_faces,
                                      denoising_strength=p.denoising_strength,
                                      init_images=images_base64,
                                      controlnet_args=c)))
        else:
            img_urls.extend(
                self._wait_task_completed(
                    self._img2img(model_name=p._remote_model_name,
                                  prompts=p.prompt,
                                  neg_prompts=p.negative_prompt,
                                  sampler_name=p.sampler_name,
                                  batch_size=p.batch_size,
                                  steps=p.steps,
                                  n_iter=p.n_iter,
                                  cfg_scale=p.cfg_scale,
                                  seed=p.seed,
                                  height=p.height,
                                  width=p.width,
                                  restore_faces=p.restore_faces,
                                  denoising_strength=p.denoising_strength,
                                  init_images=images_base64,
                                  controlnet_args=[])))
        return retrieve_images(img_urls)

    def txt2img(self, p):
        controlnet_batchs = self.check_controlnet_arg(p)

        img_urls = []
        if len(controlnet_batchs) > 0:
            for c in controlnet_batchs:
                img_urls.extend(
                    self._wait_task_completed(
                        self._txt2img(model_name=p._remote_model_name,
                                      prompts=p.prompt,
                                      neg_prompts=p.negative_prompt,
                                      sampler_name=p.sampler_name,
                                      batch_size=p.batch_size,
                                      steps=p.steps,
                                      n_iter=p.n_iter,
                                      cfg_scale=p.cfg_scale,
                                      seed=p.seed,
                                      height=p.height,
                                      width=p.width,
                                      controlnet_args=c)))
        else:
            img_urls.extend(
                self._wait_task_completed(
                    self._txt2img(model_name=p._remote_model_name,
                                  prompts=p.prompt,
                                  neg_prompts=p.negative_prompt,
                                  sampler_name=p.sampler_name,
                                  batch_size=p.batch_size,
                                  steps=p.steps,
                                  n_iter=p.n_iter,
                                  cfg_scale=p.cfg_scale,
                                  seed=p.seed,
                                  height=p.height,
                                  width=p.width,
                                  controlnet_args=[])))

        state.textinfo = "downloading images..."
        return retrieve_images(img_urls)

    def check_controlnet_arg(self, p):

        controlnet_batchs = []
        for s in p.scripts.alwayson_scripts:

            if s.filename.endswith("controlnet.py"):

                script_args = p.script_args[s.args_from:s.args_to]
                image = ""

                for c in script_args:

                    if c.enabled == False:
                        continue

                    print(c.input_mode)

                    controlnet_arg = {}
                    controlnet_arg['weight'] = c.weight
                    controlnet_arg['model'] = "control_v11f1e_sd15_tile"
                    controlnet_arg['module'] = c.module

                    if c.control_mode == "Balanced":
                        controlnet_arg['control_mode'] = 1
                    elif c.control_mode == "My prompt is more important,":
                        controlnet_arg['control_mode'] = 2
                    elif c.control_mode == "ControlNet is more important":
                        controlnet_arg['control_mode'] = 3
                    else:
                        return

                    if getattr(c.input_mode, 'value', '') == "simple":

                        base64_str = ""
                        if script_args[0].image is not None:
                            image = Image.fromarray(
                                script_args[0].image["image"])
                            base64_str = image_to_base64(image)

                        controlnet_arg['input_image'] = base64_str

                        if len(controlnet_batchs) <= 1:
                            controlnet_batchs.append([])

                        controlnet_batchs[0].append(controlnet_arg)

                    elif getattr(c.input_mode, 'value', '') == "batch":
                        if c.batch_images != "" and c.batch_images != None:
                            images = read_image_files(c.batch_images)
                            for i, img in enumerate(images):
                                if len(controlnet_batchs) <= i:
                                    controlnet_batchs.append([])

                                controlnet_new_arg = copy.deepcopy(
                                    controlnet_arg)
                                controlnet_new_arg['input_image'] = img

                                controlnet_batchs[i].append(controlnet_new_arg)
                        else:
                            print("batch_images is empty")

                    else:
                        print("input_mode is empty")

        return controlnet_batchs

    def list_models(self):
        if self._models != None:
            return self._models

    def list_models(self):
        if self._models is None or len(self._models) == 0:
            self._models = self.refresh_models()
        return sorted(self._models, key=lambda x: x.rating, reverse=True)

    def refresh_models(self):
        url = "http://api.omniinfer.io/v2/models"
        headers = {
            "accept": "application/json",
            'Accept-Encoding': 'gzip, deflate',
            "X-OmniInfer-Source": "sd-webui"
        }

        print("[cloud-inference] refreshing models...")
        results = []
        res = requests.get(url, headers=headers)
        if res.status_code >= 400:
            return []
        # return [_["name"] for _ in res.json()["data"]["models"]]
        m = {}
        for item in res.json()["data"]["models"]:
            if item['type'] == 'checkpoint':
                ckpt = Checkpoint(
                    name=item["sd_name"],
                    rating=item.get("civitai_download_count", 0),
                )

                if len(item.get(
                        'civitai_images',
                    [])) > 0 and item['civitai_images'][0]['meta'].get(
                        'prompt') is not None:
                    first_image = item['civitai_images'][0]['meta']
                    ckpt.example = CheckpointExample(
                        prompts=first_image['prompt'],
                        neg_prompt=first_image.get('negative_prompt', None),
                        width=first_image.get('width', None),
                        height=first_image.get('height', None),
                        sampler_name=first_image.get('sampler_name', None),
                        cfg_scale=first_image.get('cfg_scale', None),
                        seed=first_image.get('seed', None),
                    )
                m[item['sd_name']] = ckpt
        for item in res.json()["data"]["models"]:
            if item['type'] == 'lora':
                civitai_dependency_model_name = item.get(
                    'civitai_dependency_model_name', None)
                if civitai_dependency_model_name is not None:
                    if m.get(civitai_dependency_model_name) is not None:
                        m[civitai_dependency_model_name].add_lora(
                            item['sd_name'])

        for _, ckpt in m.items():
            results.append(ckpt)

        self.__class__.update_models_to_config(results)
        return results


def retrieve_images(img_urls) -> list[Image.Image]:

    def _download(img_url):
        response = requests.get(img_url)
        return Image.open(io.BytesIO(response.content))

    pool = ThreadPool()
    applied = []
    for img_url in img_urls:
        applied.append(pool.apply_async(_download, (img_url, )))
    return [r.get() for r in applied]


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
