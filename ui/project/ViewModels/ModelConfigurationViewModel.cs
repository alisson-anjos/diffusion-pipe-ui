using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.Utils;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.ViewModels
{
    public class ModelConfigurationViewModel
    {
        [IgnoreDataMember]
        public ModelType Type { get; set; }

        [DataMember(Name = "type")]
        public string TypeDescription
        {
            get => Type.GetDescription();
            set => Type = value.GetEnumFromDescription<ModelType>();
        }

        [DataMember(Name = "checkpoint_path")]
        public string? CheckpointPath { get; set; } = null;

        [DataMember(Name = "diffusers_path")]
        public string? DiffusersPath { get; set; } = null;

        [DataMember(Name = "transformer_path")]
        public string? TransformerPath { get; set; } = null;

        [DataMember(Name = "vae_path")]
        public string? VaePath { get; set; } = null;
        
        [DataMember(Name = "text_encoder_path")] 
        public string? TextEncoderPath { get; set; } = null;

        [DataMember(Name = "llm_path")] 
        public string? LlmPath { get; set; } = null;
        
        [DataMember(Name = "clip_path")]
        public string? ClipPath { get; set; } = null;

        [IgnoreDataMember]
        public Dtype Dtype { get; set; } = Dtype.BFloat16;

        [DataMember(Name = "dtype")]
        public string DtypeDescription
        {
            get => Dtype.GetDescription();
            set => Dtype = value.GetEnumFromDescription<Dtype>();
        }

        [DataMember(Name = "transformer_dtype")]
        public string? TransformerDTypeDescription
        {
            get => TransformerDType?.GetDescription();
            set => TransformerDType = value != null ? value.GetEnumFromDescription<Dtype>() : null;
        }

        [IgnoreDataMember]
        public Dtype? TransformerDType { get; set; } = Dtype.Float8;


        [DataMember(Name = "timestep_sample_method")]
        public string? TimestepSampleMethodDescription
        {
            get => TimestepSampleMethod?.GetDescription();
            set => TimestepSampleMethod = value != null ? value.GetEnumFromDescription<SampleMethod>() : null;
        }

        [IgnoreDataMember]
        public SampleMethod? TimestepSampleMethod { get; set; } = null;
        
        public bool? IsOfficialCheckpoint { get; set; } = null;
    }
}
