using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.Utils;
using System.Runtime.Serialization;
using System.Text.Json;

namespace DiffusionPipeInterface.ViewModels
{
    public class OptimizerConfigurationViewModel
    {
        public OptimizerConfigurationViewModel()
        {
            BetasJson = "[0.9, 0.99]";
        }

        public int Id { get; set; }

        [IgnoreDataMember]
        public Optimizer Type { get; set; } = Optimizer.AdamwOptimi;

        [DataMember(Name = "type")]
        public string TypeDescription
        {
            get => Type.GetDescription();
            set => Type = value.GetEnumFromDescription<Optimizer>();
        }

        [DataMember(Name = "lr")]
        public double Lr { get; set; } = 0.00002;

        [IgnoreDataMember]
        private string? _betasJson;

        [DataMember(Name = "betas")]
        public double[]? Betas
        {
            get => _betasJson != null ? JsonSerializer.Deserialize<double[]>(_betasJson) : null;
            set => _betasJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        [DataMember(Name = "weight_decay")]
        public double WeightDecay { get; set; } = 0.01;

        [DataMember(Name = "eps")]
        public double Eps { get; set; } = 0.00000001;

        [IgnoreDataMember]
        public string? BetasJson
        {
            get => _betasJson;
            set => _betasJson = value;
        }
    }
}
