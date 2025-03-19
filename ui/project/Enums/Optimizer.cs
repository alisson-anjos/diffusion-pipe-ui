using System.ComponentModel;

namespace DiffusionPipeInterface.Enums
{
    public enum Optimizer
    {
        [Description("adamw")]
        Adamw,
        [Description("adamw8bit")]
        Adamw8bit,
        [Description("adamw_optimi")]
        AdamwOptimi,
        [Description("stableadamw")]
        StableAdamw,
        [Description("sgd")]
        Sgd,
        [Description("adamw8bitKahan")]
        Adamw8bitKahan,
        [Description("offload")]
        Offload,
    }
}
