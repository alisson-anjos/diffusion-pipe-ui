using DiffusionPipeInterface.Enums;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.ViewModels.Models
{
    public class ChromaModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public ChromaModelConfigurationViewModel()
        {
            Type = ModelType.Chroma;
            TransformerDType = Dtype.Float8;
        }

        [DataMember(Name = "flux_shift")]
        public bool FluxShift { get; set; } = true;
    }
}
