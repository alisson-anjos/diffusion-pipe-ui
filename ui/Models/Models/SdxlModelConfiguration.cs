namespace DiffusionPipeInterface.Models.Models
{
    public class SdxlModelConfiguration : ModelConfiguration
    {
        public SdxlModelConfiguration()
        {
            Type = Enums.ModelType.SDXL;
        }

        public float UnetLr { get; set; } = 4e-5f;
        public float TextEncoder1Lr { get; set; } = 2e-5f;
        public float TextEncoder2Lr { get; set; } = 2e-5f;
        public bool? VPred { get; set; } = false;
        public float? MinSnrGamma { get; set; } = null;
        public bool? DebiasedEstimationLoss { get; set; } = false;
    }
}
