using DiffusionPipeInterface.Enums;

namespace DiffusionPipeInterface.Models.Models
{
    public class FluxModelConfiguration : ModelConfiguration
    {
        public FluxModelConfiguration()
        {
            Type = Enums.ModelType.Flux;
            TransformerDType = Dtype.Float8;
        }

        public bool FluxShift { get; set; } = true;
        public bool? BypassGuidanceEmbedding { get; set; } = null;
    }
}
