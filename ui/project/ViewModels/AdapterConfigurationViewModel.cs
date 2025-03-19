using DiffusionPipeInterface.Enums;
using DiffusionPipeInterface.Utils;
using System.Runtime.Serialization;

namespace DiffusionPipeInterface.ViewModels
{
    public class AdapterConfigurationViewModel
    {
        public int Id { get; set; }

        [DataMember(Name = "type")]
        public string Type { get; set; } = "lora";

        [DataMember(Name = "rank")]
        public int Rank { get; set; } = 32;

        [IgnoreDataMember]
        public Dtype Dtype { get; set; } = Dtype.BFloat16;

        [DataMember(Name = "dtype")]
        public string DtypeDescription
        {
            get => Dtype.GetDescription();
            set => Dtype = value.GetEnumFromDescription<Dtype>();
        }

        [DataMember(Name = "init_from_existing")]
        public string? InitFromExisting { get; set; } = null;
    }
}
