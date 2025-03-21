using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.Models.Models;
using DiffusionPipeInterface.Utils;
using DiffusionPipeInterface.ViewModels.Models;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.ViewModels
{
    public class TrainConfigurationViewModel
    {
        public int Id { get; set; }

        [DataMember(Name = "name")]
        public string Name { get; set; } = null!;

        [DataMember(Name = "output_dir")]
        public string OutputDir { get; set; } = null!;

        [DataMember(Name = "dataset")]
        public string DatasetConfig { get; set; } = null!;

        [DataMember(Name = "epochs")]
        public int Epochs { get; set; } = 1000;

        [DataMember(Name = "micro_batch_size_per_gpu")]
        public int MicroBatchSizePerGPU { get; set; } = 1;

        [DataMember(Name = "pipeline_stages")]
        public int PipelineStages { get; set; } = 1;

        [DataMember(Name = "gradient_accumulation_steps")]
        public int GradientAccumulationSteps { get; set; } = 1;

        [DataMember(Name = "gradient_clipping")]
        public float GradientClipping { get; set; } = 1.0f;

        [DataMember(Name = "warmup_steps")]
        public int WarmupSteps { get; set; } = 100;

        [DataMember(Name = "eval_every_n_epochs")]
        public int EvalEveryNEpochs { get; set; } = 1;

        [DataMember(Name = "eval_before_first_step")]
        public bool EvalBeforeFirstSteps { get; set; } = true;

        [DataMember(Name = "eval_micro_batch_size_per_gpu")]
        public int EvalMicroBatchSizePerGPU { get; set; } = 1;

        [DataMember(Name = "eval_gradient_accumulation_steps")]
        public int EvalGradientAccumulationSteps { get; set; } = 1;
        
        [DataMember(Name = "save_every_n_epochs")]
        public int SaveEveryNEpochs { get; set; } = 2;

        [DataMember(Name = "checkpoint_every_n_minutes")]
        public int CheckpointEveryNMinutes { get; set; } = 120;
        
        [DataMember(Name = "activation_checkpointing")]
        public bool ActivationCheckpointing { get; set; } = true;

        [IgnoreDataMember]
        public PartitionMethod PartitionMethod { get; set; } = PartitionMethod.Parameters;

        [DataMember(Name = "partition_method")]
        public string PartitionMethodDescription
        {
            get => PartitionMethod.GetDescription();
            set => PartitionMethod = value.GetEnumFromDescription<PartitionMethod>();
        }

        [IgnoreDataMember]
        public Dtype SaveDType { get; set; } = Dtype.BFloat16;

        [DataMember(Name = "save_dtype")]
        public string SaveDTypeDescription
        {
            get => SaveDType.GetDescription();
            set => SaveDType = value.GetEnumFromDescription<Dtype>();
        }

        [DataMember(Name = "caching_batch_size")]
        public int CachingBatchSize { get; set; } = 1;

        [DataMember(Name = "steps_per_print")]
        public int StepsPerPrint { get; set; } = 1;

        [IgnoreDataMember]
        public VideoClipMode VideoClipMode { get; set; } = VideoClipMode.SingleBeginning;

        [DataMember(Name = "video_clip_mode")]
        public string VideoClipModeDescription
        {
            get => VideoClipMode.GetDescription();
            set => VideoClipMode = value.GetEnumFromDescription<VideoClipMode>();
        }

        [DataMember(Name = "blocks_to_swap")]
        public int? BlocksToSwap { get; set; } = null;

        [DataMember(Name = "model")]
        public virtual ModelConfigurationViewModel ModelConfiguration { get; set; } = new ModelConfigurationViewModel();

        [DataMember(Name = "adapter")]
        public virtual AdapterConfigurationViewModel AdapterConfiguration { get; set; } = new AdapterConfigurationViewModel();

        [DataMember(Name = "optimizer")]
        public virtual OptimizerConfigurationViewModel OptimizerConfiguration { get; set; } = new OptimizerConfigurationViewModel();

        [DataMember(Name = "monitoring")]
        public virtual MonitoringConfigurationViewModel MonitoringConfiguration { get; set; } = new MonitoringConfigurationViewModel();

        [IgnoreDataMember]
        public ChromaModelConfigurationViewModel ChromaModelConfiguration { get; set; } = new ChromaModelConfigurationViewModel();
        
        [IgnoreDataMember]
        public CosmosModelConfigurationViewModel CosmosModelConfiguration { get; set; } = new CosmosModelConfigurationViewModel();
        
        [IgnoreDataMember]
        public FluxModelConfigurationViewModel FluxModelConfiguration { get; set; } = new FluxModelConfigurationViewModel();
        
        [IgnoreDataMember]
        public HunyuanModelConfigurationViewModel HunyuanModelConfiguration { get; set; } = new HunyuanModelConfigurationViewModel();
        
        [IgnoreDataMember]
        public LtxModelConfigurationViewModel LtxModelConfiguration { get; set; } = new LtxModelConfigurationViewModel();
        
        [IgnoreDataMember]
        public LuminaModelConfigurationViewModel LuminaModelConfiguration { get; set; } = new LuminaModelConfigurationViewModel();
        
        [IgnoreDataMember]
        public SdxlModelConfigurationViewModel SdxlModelConfiguration { get; set; } = new SdxlModelConfigurationViewModel();
        
        [IgnoreDataMember]
        public WanModelConfigurationViewModel WanModelConfiguration { get; set; } = new WanModelConfigurationViewModel();

        public ModelConfigurationViewModel GetCurrentModelConfiguration()
        {
            return ModelConfiguration.Type switch
            {
                ModelType.Chroma => ChromaModelConfiguration,
                ModelType.Cosmos => CosmosModelConfiguration,
                ModelType.Flux => FluxModelConfiguration,
                ModelType.Hunyuan => HunyuanModelConfiguration,
                ModelType.LTX => LtxModelConfiguration,
                ModelType.Lumina => LuminaModelConfiguration,
                ModelType.SDXL => SdxlModelConfiguration,
                ModelType.Wan21 => WanModelConfiguration,
                _ => ModelConfiguration
            };
        }
    }
}
