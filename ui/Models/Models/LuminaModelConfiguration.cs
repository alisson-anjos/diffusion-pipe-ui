namespace DiffusionPipeInterface.Models.Models
{
    public class LuminaModelConfiguration : ModelConfiguration
    {
        public LuminaModelConfiguration()
        {
            Type = Enums.ModelType.Lumina;
        }

        public bool LuminaShift { get; set; } = true;
    }
}
