using System.Runtime.Serialization;

namespace DiffusionPipeInterface.ViewModels.Models
{
    public class FluxModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public FluxModelConfigurationViewModel()
        {
            Type = Enums.ModelType.Flux;
            TransformerDType = Enums.Dtype.Float8;
        }

        [DataMember(Name = "flux_shift")]
        public bool FluxShift { get; set; } = true;

        [DataMember(Name = "bypass_guidance_embedding")]
        public bool? BypassGuidanceEmbedding { get; set; } = false;
    }
}
