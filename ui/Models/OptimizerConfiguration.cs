using DiffusionPipeInterface.Enums;
using System.ComponentModel.DataAnnotations.Schema;
using System.ComponentModel.DataAnnotations;
using System.Text.Json;

namespace DiffusionPipeInterface.Models
{
    public class OptimizerConfiguration
    {
        [Key]
        public int Id { get; set; }

        public Optimizer Type { get; set; } = Optimizer.AdamwOptimi;

        public double Lr { get; set; } = 0.00002;

        private string? _betasJson;

        [NotMapped]
        public double[]? Betas
        {
            get => _betasJson != null ? JsonSerializer.Deserialize<double[]>(_betasJson) : null;
            set => _betasJson = value != null ? JsonSerializer.Serialize(value) : null;
        }

        public double WeightDecay { get; set; } = 0.01;

        public double Eps { get; set; } = 0.00000001;

        [Column("Betas")]
        public string? BetasJson
        {
            get => _betasJson;
            set => _betasJson = value;
        }
    }
}
