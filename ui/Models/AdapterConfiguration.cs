using DiffusionPipeInterface.Enums;
using System.ComponentModel.DataAnnotations;

namespace DiffusionPipeInterface.Models
{
    public class AdapterConfiguration
    {
        [Key]
        public int Id { get; set; }

        public string Type { get; set; } = "lora";
        public int Rank { get; set; } = 32;
        public Dtype Dtype { get; set; } = Dtype.BFloat16;
    }
}
