using DiffusionPipeInterface.ViewModels;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.Models.Models
{
    public class LuminaModelConfigurationViewModel : ModelConfigurationViewModel
    {
        public LuminaModelConfigurationViewModel()
        {
            Type = Enums.ModelType.Lumina;
        }

        [DataMember(Name = "lumina_shift")]
        public bool LuminaShift { get; set; } = true;
    }
}
