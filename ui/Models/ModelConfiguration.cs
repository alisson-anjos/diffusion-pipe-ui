using DiffusionPipeInterface.Enums;
using System.ComponentModel.DataAnnotations;

namespace DiffusionPipeInterface.Models
{
    public class ModelConfiguration
    {
        [Key]
        public int Id { get; set; }
        public ModelType Type { get; set; }
        public string? CheckpointPath { get; set; } = null;
        public string? DiffusersPath { get; set; } = null;
        public string? TransformerPath { get; set; } = null;
        public string? VaePath { get; set; } = null;
        public string? TextEncoderPath { get; set; } = null;
        public string? LlmPath { get; set; } = null;
        public string? ClipPath { get; set; } = null;
        public Dtype Dtype { get; set; } = Dtype.BFloat16;
        public Dtype? TransformerDType { get; set; } = Dtype.Float8;
        public SampleMethod? TimestepSampleMethod { get; set; } = null;
        public bool? IsOfficialCheckpoint { get; set; } = null;
    }
}
