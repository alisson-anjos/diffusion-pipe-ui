using DiffusionPipeInterface.Enums;

namespace DiffusionPipeInterface.Models.Models
{
    public class HunyuanModelConfiguration : ModelConfiguration
    {
        public HunyuanModelConfiguration()
        {
            Type = Enums.ModelType.Hunyuan;
            TransformerDType = Dtype.Float8;    
        }

        public string CkptPath { get; set; } = string.Empty;
    }
}
