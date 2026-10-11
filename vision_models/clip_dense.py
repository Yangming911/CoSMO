# numpy
import numpy as np

# modeling
from vision_models.base_model import BaseModel
from detectron2.data.detection_utils import read_image
import detectron2.data.transforms as T
from torch.nn import functional as F
from fvcore.common.checkpoint import Checkpointer
import open_clip
import torch

# Visualization
import matplotlib.pyplot as plt

# typing
from typing import List, Optional

try:
    import tensorrt as trt
except:
    print("TensorRT not available, cannot use Jetson")

class ClipModel(torch.nn.Module, BaseModel):
    def __init__(self,
                 path: str,
                 jetson: bool = False
                 ):
        super(ClipModel, self).__init__()
        self.jetson = jetson

        self.input_format = "RGB"

        self.aug = T.ResizeShortestEdge(
            [640, 640], 2560
        )
        self.tokenizer = open_clip.get_tokenizer('convnext_large_d_320')
        self.feature_dim = 768
        self.clip_resolution = (768, 768)

        if self.jetson:
            logger = trt.Logger(trt.Logger.WARNING)
            # Load jetson specific model
            runtime = trt.Runtime(logger)
            with open("trt/clip_model_trt.engine", "rb") as f:
                self.engine = runtime.deserialize_cuda_engine(f.read())
            self.context = self.engine.create_execution_context()
            self.stream = torch.cuda.current_stream(torch.device("cuda"))
            with open("trt/clip_text_model_trt.engine", "rb") as f:
                self.engine_txt = runtime.deserialize_cuda_engine(f.read())
            self.context_txt = self.engine_txt.create_execution_context()

        else:
            name, pretrain = ('convnext_large_d_320', 'laion2b_s29b_b131k_ft_soup')
            clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(
                name,
                pretrained=pretrain,
                device="cuda", )
            checkpointer = Checkpointer(clip_model)
            checkpointer.load(path)
            self.clip_model = clip_model.float()
            self.clip_model.eval()
            self.clip_mean = torch.Tensor([122.7709383, 116.7460125, 104.09373615]).to("cuda")
            self.clip_mean = self.clip_mean.unsqueeze(-1).unsqueeze(-1)
            self.clip_std = torch.Tensor([68.5005327, 66.6321579, 70.3231630]).to("cuda")
            self.clip_std = self.clip_std.unsqueeze(-1).unsqueeze(-1)

    def eval(self):
        super().eval()
        if self.jetson:
            pass
        else:
            self.clip_model.eval()
        return self

    def compute_similarity(self,
                           image_feats: torch.Tensor,
                           text_feats: torch.Tensor,
                           ) -> torch.Tensor:
        if len(image_feats.shape) == 3:
            return torch.einsum('bcx, bc -> bx', image_feats, text_feats)
        else:
            return torch.einsum('bchw, bc -> bhw', image_feats, text_feats)

    def forward_im(self, images: torch.Tensor):
        return self.image_forward_torch(images)

    def forward_text(self, text_tokenized):
        with torch.no_grad():
            class_embeddings = self.clip_model.encode_text(text_tokenized)
            return F.normalize(class_embeddings, dim=1)

    def image_forward_torch(self, clip_images: torch.Tensor):
        with torch.no_grad():
            # NOTE Moved normalization to the beginning to match the paper implementation
            clip_images = (clip_images - self.clip_mean) / self.clip_std
            clip_images = F.interpolate(clip_images, size=self.clip_resolution, mode='bilinear',
                                        align_corners=False, )
            clip_features = self.clip_model.encode_image(clip_images, dense=True)
            clip_vis_dense = clip_features["clip_vis_dense"]

            return F.normalize(clip_vis_dense, dim=1)

    def text_forward_trt(self, texts: torch.Tensor):
        output_shape = self.engine_txt.get_binding_shape(1)
        input_tensor = texts.cuda()
        output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
        d_input = input_tensor.data_ptr()
        d_output = output_tensor.data_ptr()

        self.context_txt.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=self.stream.cuda_stream)
        return F.normalize(output_tensor, dim=1).to(torch.float32)

    def image_forward_trt(self, input_tensor: torch.Tensor):
        output_shape = self.engine.get_binding_shape(1)
        output_tensor = torch.empty(*output_shape, dtype=torch.float32, device="cuda")
        d_output = output_tensor.data_ptr()
        d_input = input_tensor.data_ptr()
        bindings = [int(d_input)] + [int(d_output)]
        self.context.execute_async_v2(bindings=bindings, stream_handle=self.stream.cuda_stream)
        return F.normalize(output_tensor, dim=1).to(torch.float32)

    def get_image_features(self,
                           images: np.ndarray
                           ) -> torch.Tensor:
        # expects images in shape B C H W in BGR, expected to be a numpy array
        if len(images.shape) == 3:
            images = np.expand_dims(images, 0)
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "BGR":
                # whether the model expects BGR inputs or RGB
                original_image = images[:, ::-1, :, :]
            else:
                original_image = images
            to_transform_img = original_image.transpose(0, 2, 3, 1)
            transformed_0 = self.aug.get_transform(to_transform_img[0]).apply_image(to_transform_img[0]).transpose(2, 0,
                                                                                                                   1)
            transformed = np.zeros((to_transform_img.shape[0], *transformed_0.shape), dtype=transformed_0.dtype)
            transformed[0] = transformed_0
            for i in range(1, to_transform_img.shape[0]):
                transformed[i] = self.aug.get_transform(to_transform_img[i]).apply_image(to_transform_img[i]).transpose(
                    2, 0, 1)
            # After, differentiate between jetson and normal
            if self.jetson:
                transformed = F.interpolate(torch.as_tensor(transformed.astype("float32")).to("cuda"),
                                            size=self.clip_resolution, mode='bilinear',
                                            align_corners=False, )
                return self.image_forward_trt(transformed)
            else:
                transformed = torch.as_tensor(transformed.astype("float32")).to("cuda")

                return self.image_forward_torch(transformed)

    def get_text_features(self,
                          texts: List[str]
                          ) -> torch.Tensor:
        with torch.no_grad():
            texts = self.tokenizer(texts)

            # After, differentiate between jetson and normal

            if self.jetson:
                return self.text_forward_trt(texts)
            else:
                class_embeddings = self.clip_model.encode_text(texts.to("cuda"))
                return F.normalize(class_embeddings, dim=1)
