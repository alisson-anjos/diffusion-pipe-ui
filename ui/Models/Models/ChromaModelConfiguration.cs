using DiffusionPipeInterface.Enums;

namespace DiffusionPipeInterface.Models.Models
{
    public class ChromaModelConfiguration : ModelConfiguration
    {
        public ChromaModelConfiguration()
        {
            Type = Enums.ModelType.Chroma;
            TransformerDType = Dtype.Float8;
        }

        public bool FluxShift { get; set; } = true;
    }
}
