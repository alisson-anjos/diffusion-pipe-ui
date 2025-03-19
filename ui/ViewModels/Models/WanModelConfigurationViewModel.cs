using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.ViewModels;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.Models.Models
{
    public class WanModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public WanModelConfigurationViewModel()
        {
            Type = Enums.ModelType.Wan21;
            TransformerDType = Dtype.Float8;
            TimestepSampleMethod = SampleMethod.LogitNormal;
        }

        [DataMember(Name = "ckpt_path")]
        public string CkptPath { get; set; } = string.Empty;
    }
}
