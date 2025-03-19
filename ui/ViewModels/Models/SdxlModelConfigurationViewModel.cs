using DiffusionPipeInterface.ViewModels;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.Models.Models
{
    public class SdxlModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public SdxlModelConfigurationViewModel()
        {
            Type = Enums.ModelType.SDXL;
        }

        [DataMember(Name = "unet_lr")]
        public double UnetLr { get; set; } = 4e-5;
        
        [DataMember(Name = "text_encoder_1_lr")]
        public double TextEncoder1Lr { get; set; } = 2e-5;

        [DataMember(Name = "text_encoder_2_lr")]
        public double TextEncoder2Lr { get; set; } = 2e-5;

        [DataMember(Name = "v_pred")]
        public bool? VPred { get; set; } = false;

        [DataMember(Name = "min_snr_gamma")]
        public double? MinSnrGamma { get; set; } = null;

        [DataMember(Name = "debiased_estimation_loss")]
        public bool? DebiasedEstimationLoss { get; set; } = false;
    }
}
