using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.ViewModels;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.Models.ViewModels
{
    public class HunyuanModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public HunyuanModelConfigurationViewModel()
        {
            Type = Enums.ModelType.Hunyuan;
            TransformerDType = Dtype.Float8;
            TimestepSampleMethod = SampleMethod.LogitNormal;
        }

        [DataMember(Name = "ckpt_path")]
        public string CkptPath { get; set; } = string.Empty;
    }
}
