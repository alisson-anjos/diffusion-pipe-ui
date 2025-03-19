namespace DiffusionPipeInterface.Models.Models
{
    public class LtxModelConfiguration : ModelConfiguration
    {
        public LtxModelConfiguration()
        {
            Type = Enums.ModelType.LTX;
        }

        public string SingleFilePath { get; set; } = string.Empty;
    }
}
