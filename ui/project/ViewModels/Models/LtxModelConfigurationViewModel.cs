using DiffusionPipeInterface.ViewModels;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.Models.Models
{
    public class LtxModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public LtxModelConfigurationViewModel()
        {
            Type = Enums.ModelType.LTX;
            TimestepSampleMethod = Enums.SampleMethod.LogitNormal;
        }

        [DataMember(Name = "single_file_path")]
        public string SingleFilePath { get; set; } = string.Empty;
    }
}
