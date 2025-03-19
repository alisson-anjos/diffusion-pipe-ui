using DiffusionPipeInterface.Enums;
using System.ComponentModel.DataAnnotations;

namespace DiffusionPipeInterface.Models
{
    public class TrainConfiguration
    {
        [Key]
        public int Id { get; set; }
        public string Name { get; set; } = null!;
        public string OutputDir { get; set; } = null!;
        public string DatasetConfig { get; set; } = null!;
        public int Epochs { get; set; } = 1000;
        public int MicroBatchSizePerGPU { get; set; } = 1;
        public int PipelineStages { get; set; } = 1;
        public int GradientAccumulationSteps { get; set; } = 4;
        public float GradientClipping { get; set; } = 1.0f;
        public int WarmupSteps { get; set; } = 100;

        public int EvalEveryNEpochs { get; set; } = 1;
        public bool EvalBeforeFirstSteps { get; set; } = true;
        public int EvalMicroBatchSizePerGPU { get; set; } = 1;
        public int EvalGradientAccumulationSteps { get; set; } = 1;

        public int SaveEveryNEpochs { get; set; } = 2;
        public int CheckpointEveryNMinutes { get; set; } = 120;
        public bool ActivationCheckpointing { get; set; } = true;
        public PartitionMethod PartitionMethod { get; set; } = PartitionMethod.Parameters;
        public Dtype SaveDType { get; set; } = Dtype.BFloat16;
        public int CachingBatchSize { get; set; } = 1;
        public int StepsPerPrint { get; set; } = 1;
        public VideoClipMode VideoClipMode { get; set; } = VideoClipMode.SingleMiddle;

        public virtual ModelConfiguration ModelConfiguration { get; set; } = new ModelConfiguration();
        public virtual AdapterConfiguration AdapterConfiguration { get; set; } = new AdapterConfiguration();
        public virtual OptimizerConfiguration OptimizerConfiguration { get; set; } = new OptimizerConfiguration();
    }
}
