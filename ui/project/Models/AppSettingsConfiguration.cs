using DiffusionPipeInterface.Models.Models;

namespace DiffusionPipeInterface.Models
{
    public class AppSettingsConfiguration
    {
        public string DatasetsPath { get; set; } = null!;
        public string ConfigsPath { get; set; } = null!;
        public string OutputsPath { get; set; } = null!;
        public string ModelsPath { get; set; } = null!;
        public int UploadChunkFileSizeInMB { get; set; } = 10;
        public int MaxFileSizeInMB { get; set; } = 1024;
        public string NameDatasetDefault { get; set; } = "Default";
        public bool UsingDiffusionPipeFromFork  { get; set; } = true;
        public string EnvPath { get; set; } = null!;
        public string DiffusionPipePath { get; set; } = null!;
        public string StartTrainingCommand { get; set; } = null!;
        public HunyuanModelConfiguration HunyuanModelConfigurationDefault { get; set; } = new();
    }
}
