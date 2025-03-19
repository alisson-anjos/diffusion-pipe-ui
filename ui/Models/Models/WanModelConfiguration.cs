using DiffusionPipeInterface.Enums;

namespace DiffusionPipeInterface.Models.Models
{
    public class WanModelConfiguration : ModelConfiguration
    {
        public WanModelConfiguration()
        {
            Type = Enums.ModelType.Wan21;
            TransformerDType = Dtype.Float8;
        }

        public string CkptPath { get; set; } = string.Empty;
    }
}
