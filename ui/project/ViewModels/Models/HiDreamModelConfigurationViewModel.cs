using DiffusionPipeInterface.Enums;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.ViewModels.Models
{
    public class HiDreamModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public HiDreamModelConfigurationViewModel()
        {
            Type = ModelType.HiDream;
            TransformerDType = Dtype.Float8;
        }

        [DataMember(Name = "flux_shift")]
        public bool? FluxShift { get; set; } = null;

        [DataMember(Name = "llama3_path")]
        public string Llama3Path { get; set; } = string.Empty;

        [DataMember(Name = "llama3_4bit")]
        public bool Llama34bit { get; set; } = true;

        [DataMember(Name = "max_llama3_sequence_length")]
        public int MaxLlama3SequenceLength { get; set; } = 128;
    }
}
