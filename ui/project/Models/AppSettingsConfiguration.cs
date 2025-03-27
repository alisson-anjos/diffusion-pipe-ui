using DiffusionPipeInterface.Models.Models;
using DiffusionPipeInterface.ViewModels.Models;

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
        public string ServerUrlBase { get;set; } = null!;
        public AppSettingsModelsConfiguration Models { get; set; } = new();
    }

    public class AppSettingsModelsConfiguration
    {
        public SdxlModelConfigurationViewModel SDXL { get; set; } = new();
        public ChromaModelConfigurationViewModel Chroma { get; set; } = new();
        public CosmosModelConfigurationViewModel Cosmos { get; set; } = new();
        public FluxModelConfigurationViewModel Flux { get; set; } = new();
        public LtxModelConfigurationViewModel LTX { get; set; } = new();
        public LuminaModelConfigurationViewModel Lumina { get; set; } = new();
        public WanModelConfigurationViewModel Wan21 { get; set; } = new();
        public HunyuanModelConfigurationViewModel Hunyuan { get; set; } = new();
    }
}
