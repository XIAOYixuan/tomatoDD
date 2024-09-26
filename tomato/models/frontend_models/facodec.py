from pathlib import Path
import torch
from huggingface_hub import hf_hub_download

from tomato.utils import logger
from .base import BaseFrontEnd
from .ns3_codec import FACodecEncoderV2, FACodecDecoderV2

class FACodec(BaseFrontEnd):

    def __init__(self, device, args=None):
        super().__init__(device, args)

        self.fa_encoder = FACodecEncoderV2(
            ngf=32,
            up_ratios=[2, 4, 5, 5],
            out_channels=256,
        )

        output_latent = getattr(args, "output_latent", "pcr")
        self.fa_decoder = FACodecDecoderV2(
            in_channels=256,
            upsample_initial_channel=1024,
            ngf=32,
            up_ratios=[5, 5, 4, 2],
            vq_num_q_c=2,
            vq_num_q_p=1,
            vq_num_q_r=3,
            vq_dim=256,
            codebook_dim=8,
            codebook_size_prosody=10,
            codebook_size_content=10,
            codebook_size_residual=10,
            use_gr_x_timbre=True,
            use_gr_residual_f0=True,
            use_gr_residual_phone=True,
            output_latent=output_latent
        )

        if args is not None:
            encoder_path = getattr(args, "encoder_path", None)
            decoder_path = getattr(args, "decoder_path", None)
        else:
            encoder_path = None
            decoder_path = None

        if encoder_path is None or decoder_path is None:
            logger.info("Downloading pretrained model using huggingface_hub") 
            encoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_encoder_v2.bin")
            decoder_ckpt = hf_hub_download(repo_id="amphion/naturalspeech3_facodec", filename="ns3_facodec_decoder_v2.bin")
            self.fa_encoder.load_state_dict(torch.load(encoder_ckpt))
            self.fa_decoder.load_state_dict(torch.load(decoder_ckpt))
        else:
            encoder_path = Path(args.encoder_path)
            decoder_path = Path(args.decoder_path)
            self.fa_encoder.load_state_dict(torch.load(encoder_path))
            self.fa_decoder.load_state_dict(torch.load(decoder_path))
        

        self.device = device
        self.fa_encoder.to(device)
        self.fa_decoder.to(device)

    def extract_all_feat(self, input_data):
        #input_data = input_data.unsqueeze(1) # N, C, T

        enc_out = self.fa_encoder(input_data)
        prosody = self.fa_encoder.get_prosody_feature(input_data)
        vq_post_emb, vq_id, _, quantized, spk_embs = self.fa_decoder(enc_out, prosody, eval_vq=False, vq=True)
        
        prosody_code = vq_id[:1]
        content_code = vq_id[1:3]
        residual_code = vq_id[3:]

        return {
            "latent": vq_post_emb.unsqueeze(1), # N, C, F, T
            "prosody_code": prosody_code, # 1, N, T
            "content_code": content_code, # 2, N, T
            "residual_code": residual_code, # 3, N, T
        }

    def extract_feat(self, input_data):
        # input_data (audio): N, C, T
        output = self.extract_all_feat(input_data) 
        return output["latent"]

if __name__ == "__main__" :
    device = "cpu"
    facodec = FACodec(device)
    
    bs = 3
    c = 1
    wav_length = 64_000
    x = torch.rand(bs, c, wav_length)
    feat = facodec.extract_feat(x)
    print(f"feat shape {feat.shape}")
    all_feat = facodec.extract_all_feat(x)
    for key in all_feat:
        print(f"{key} shape {all_feat[key].shape}")