using System.Runtime.Serialization;

namespace DiffusionPipeInterface.ViewModels
{
    public class MonitoringConfigurationViewModel
    {
        [DataMember(Name = "log_dir")]
        public string LogDir { get; set; } = null!;

        [DataMember(Name = "enable_wandb")]
        public bool EnableWandb { get; set; } = false;

        [DataMember(Name = "wandb_run_name")]
        public string? WandbRunName { get; set; }

        [DataMember(Name = "wandb_tracker_name")]
        public string? WandbTrackerName { get; set; }

        [DataMember(Name = "wandb_api_key")]
        public string? WandbApiKey { get; set; }
    }
}
